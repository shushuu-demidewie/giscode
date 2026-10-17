# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH PHÂN TÍCH VÀ MÔ HÌNH HÓA ĐA THỜI KỲ (CẢ 2 THÁNG: THÁNG 4 & THÁNG 11)
ĐÁNH GIÁ BIẾN ĐỘNG LỚP PHỦ VÀ SỰ DỊCH CHUYỂN NGUY CƠ TRƯỢT LỞ ĐẤT SAU BÃO YAGI
KHU VỰC: XÃ MƯỜNG LAI, HUYỆN LỤC YÊN, TỈNH YÊN BÁI (LANDSAT 8: 04/2024 & 11/2024)
================================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
import matplotlib.pyplot as plt

# Khắc phục PROJ trên Windows
try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    if os.path.exists(proj_dir):
        os.environ['PROJ_LIB'] = proj_dir
        os.environ['PROJ_DATA'] = proj_dir
except Exception:
    pass

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['SHAPE_RESTORE_SHX'] = 'YES'

try:
    from PIL import Image
except ImportError:
    Image = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score
)
import joblib

# Import các hàm trích xuất từ pipeline tháng 4 và tháng 11
from pipeline_thang_4 import extract_samples as extract_samples_th4, add_spectral_indices_to_features
from pipeline_thang_11 import extract_samples as extract_samples_th11

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU ĐẦU VÀO VÀ ĐẦU RA - CẢ 2 THÁNG
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_TH4_PATH = os.path.join(BASE_DIR, 'datath4', 'Extracth4.tif')
IMAGE_TH11_PATH = os.path.join(BASE_DIR, 'datath11', 'Extrac11.tif')

PLTH4_PATH = os.path.join(BASE_DIR, 'datath4', 'plth4.tif')
PLTH11_PATH = os.path.join(BASE_DIR, 'datath11', 'plth11.tif')

DIR_TH4 = os.path.join(WORKSPACE_DIR, 'results_thang_4')
DIR_TH11 = os.path.join(WORKSPACE_DIR, 'results_thang_11')

OUTPUT_DIR_COMBINED = os.path.join(WORKSPACE_DIR, 'results_ca_2_thang')
os.makedirs(OUTPUT_DIR_COMBINED, exist_ok=True)

# Diện tích 1 pixel Landsat 8 (30m x 30m = 900 m2 = 0.09 ha)
PIXEL_AREA_HA = 0.09

# Bảng mã lớp LULC
LULC_NAMES_TH4 = {
    1: "Thảm thực vật / Cây bụi",
    22: "Đất trống / Vết sạt trượt",
    26: "Mặt nước / Thủy vực",
    29: "Rừng tự nhiên / Thảm dày"
}

LULC_NAMES_TH11 = {
    1: "Thảm thực vật / Cây bụi",
    21: "Đất trống / Vết trượt mới (sau bão)",
    41: "Mặt nước / Thủy vực",
    46: "Rừng tự nhiên / Sinh khối cao"
}

# 5 cấp nguy cơ trượt lở
HAZARD_LEVEL_NAMES = {
    1: "Rất thấp (Very Low)",
    2: "Thấp (Low)",
    3: "Trung bình (Moderate)",
    4: "Cao (High)",
    5: "Rất cao (Very High)"
}

# Màu sắc biến động nguy cơ (Hazard Shift Map)
SHIFT_COLORS = {
    1: [43, 131, 186, 220],    # Giảm nguy cơ (Xanh dương)
    2: [240, 240, 240, 160],   # Không đổi (Xám nhạt mờ)
    3: [254, 204, 92, 230],    # Tăng 1 cấp (Vàng cam)
    4: [253, 141, 60, 240],    # Tăng 2 cấp (Da cam sẫm)
    5: [227, 26, 28, 255]      # Tăng mạnh >= 3 cấp / Sạt trượt mới (Đỏ cờ)
}


# ==============================================================================
# PHẦN 1: MA TRẬN BIẾN ĐỘNG LỚP PHỦ MẶT ĐẤT (LULC CHANGE DETECTION)
# ==============================================================================
def analyze_lulc_change():
    print("\n" + "=" * 80)
    print("  PHẦN 1: PHÂN TÍCH MA TRẬN BIẾN ĐỘNG LỚP PHỦ MẶT ĐẤT (THÁNG 4 -> THÁNG 11)")
    print("=" * 80)

    # Ưu tiên đọc từ plth4 & plth11, hoặc từ kết quả phân loại RF của 2 tháng
    r4_path = os.path.join(DIR_TH4, 'thang4_phan_loai_rf.tif') if os.path.exists(os.path.join(DIR_TH4, 'thang4_phan_loai_rf.tif')) else PLTH4_PATH
    r11_path = os.path.join(DIR_TH11, 'thang11_phan_loai_rf.tif') if os.path.exists(os.path.join(DIR_TH11, 'thang11_phan_loai_rf.tif')) else PLTH11_PATH

    print(f"[*] Raster Tháng 4:  {r4_path}")
    print(f"[*] Raster Tháng 11: {r11_path}")

    with rasterio.open(r4_path) as s4, rasterio.open(r11_path) as s11:
        arr4 = s4.read(1)
        arr11 = s11.read(1)

        # Lọc bỏ NoData (128 hoặc -9999 hoặc <=0)
        valid_mask = (arr4 > 0) & (arr4 != 128) & (arr4 != -9999) & \
                     (arr11 > 0) & (arr11 != 128) & (arr11 != -9999)

        v4 = arr4[valid_mask]
        v11 = arr11[valid_mask]

    # Lập ma trận chuyển đổi (Transition Matrix)
    df_trans = pd.DataFrame({'Thang4': v4, 'Thang11': v11})
    ct_pixel = pd.crosstab(df_trans['Thang4'], df_trans['Thang11'], margins=True, margins_name="Tổng cộng")
    ct_ha = (ct_pixel * PIXEL_AREA_HA).round(2)

    print("\n[+] MA TRẬN CHUYỂN ĐỔI DIỆN TÍCH (HÉC-TA - ha):")
    print(ct_ha)

    # Xuất ra file CSV
    csv_path = os.path.join(OUTPUT_DIR_COMBINED, 'ma_tran_chuyen_doi_lulc_thang4_thang11.csv')
    ct_ha.to_csv(csv_path, encoding='utf-8-sig')
    print(f"[+] Đã lưu ma trận biến động LULC: {csv_path}")

    # Vẽ biểu đồ chuyển đổi LULC
    plot_lulc_change_chart(ct_ha)

    return ct_ha


def plot_lulc_change_chart(ct_ha):
    try:
        # Lấy tổng diện tích Tháng 4 (theo hàng) và Tháng 11 (theo cột)
        cols_without_margin = [c for c in ct_ha.columns if c != "Tổng cộng"]
        rows_without_margin = [r for r in ct_ha.index if r != "Tổng cộng"]

        area_t4 = [ct_ha.loc[r, "Tổng cộng"] for r in rows_without_margin]
        area_t11 = [ct_ha.loc["Tổng cộng", c] for c in cols_without_margin]

        labels_t4 = [f"T4 Lớp {r}\n({LULC_NAMES_TH4.get(r, 'Khác')})" for r in rows_without_margin]
        labels_t11 = [f"T11 Lớp {c}\n({LULC_NAMES_TH11.get(c, 'Khác')})" for c in cols_without_margin]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Biểu đồ tròn Tháng 4
        axes[0].pie(area_t4, labels=labels_t4, autopct='%1.1f%%', startangle=140,
                    colors=['#74c476', '#fb6a4a', '#6baed6', '#31a354'])
        axes[0].set_title("Cơ cấu Lớp phủ Mặt đất - Tháng 4/2024\n(Trước bão Yagi)", fontsize=11, fontweight='bold')

        # Biểu đồ tròn Tháng 11
        axes[1].pie(area_t11, labels=labels_t11, autopct='%1.1f%%', startangle=140,
                    colors=['#74c476', '#de2d26', '#6baed6', '#006d2c'])
        axes[1].set_title("Cơ cấu Lớp phủ Mặt đất - Tháng 11/2024\n(Sau bão Yagi)", fontsize=11, fontweight='bold')

        plt.tight_layout()
        chart_file = os.path.join(OUTPUT_DIR_COMBINED, 'bieu_do_co_cau_lulc_thang4_thang11.png')
        plt.savefig(chart_file)
        plt.close()
        print(f"[+] Đã xuất biểu đồ cơ cấu biến động: {chart_file}")
    except Exception as e:
        print(f"[!] Bỏ qua biểu đồ LULC: {e}")


# ==============================================================================
# PHẦN 2: PHÂN TÍCH ĐỘNG THÁI DỊCH CHUYỂN NGUY CƠ TRƯỢT LỞ (HAZARD SHIFT)
# ==============================================================================
def analyze_hazard_shift(model_code="rf", model_name="Random Forest"):
    print("\n" + "=" * 80)
    print(f"  PHẦN 2: ĐÁNH GIÁ BIẾN ĐỘNG NGUY CƠ TRƯỢT LỞ ĐẤT ({model_name.upper()})")
    print("=" * 80)

    p4 = os.path.join(DIR_TH4, f'thang4_nguy_co_5cap_{model_code}.tif')
    p11 = os.path.join(DIR_TH11, f'thang11_nguy_co_5cap_{model_code}.tif')

    if not os.path.exists(p4) or not os.path.exists(p11):
        print(f"[!] Không tìm thấy kết quả 5 cấp nguy cơ cho {model_name}.")
        return None

    with rasterio.open(p4) as s4, rasterio.open(p11) as s11:
        crs, bounds = s4.crs, s4.bounds
        height, width = s4.height, s4.width
        r4 = s4.read(1)
        r11 = s11.read(1)

        valid = (r4 > 0) & (r4 <= 5) & (r11 > 0) & (r11 <= 5)
        v4 = r4[valid]
        v11 = r11[valid]

        # 1. Ma trận dịch chuyển 5 cấp nguy cơ (Hazard Shift Matrix)
        df_hz = pd.DataFrame({'T4_Hazard': v4, 'T11_Hazard': v11})
        ct_hz_pixel = pd.crosstab(df_hz['T4_Hazard'], df_hz['T11_Hazard'], margins=True, margins_name="Tổng cộng")
        ct_hz_ha = (ct_hz_pixel * PIXEL_AREA_HA).round(2)

        print(f"\n[+] MA TRẬN DỊCH CHUYỂN 5 CẤP NGUY CƠ TRƯỢT LỞ {model_name.upper()} (ha):")
        print(ct_hz_ha)

        csv_hz = os.path.join(OUTPUT_DIR_COMBINED, f'ma_tran_dich_chuyen_nguy_co_{model_code}.csv')
        ct_hz_ha.to_csv(csv_hz, encoding='utf-8-sig')

        # 2. Tạo bản đồ không gian biến động nguy cơ (Hazard Dynamics Raster)
        # Phân loại mức độ thay đổi:
        # 1: Giảm nguy cơ (Diff < 0)
        # 2: Không đổi (Diff == 0)
        # 3: Tăng nhẹ 1 cấp (Diff == 1)
        # 4: Tăng đáng kể 2 cấp (Diff == 2)
        # 5: Tăng rất mạnh >= 3 cấp (Vùng phát sinh sạt trượt nguy hiểm mới sau bão)
        diff_array = np.full((height, width), -9999, dtype=np.int16)
        shift_class = np.full((height, width), -9999, dtype=np.int16)
        rgba_shift = np.zeros((height, width, 4), dtype=np.uint8)

        diff_vals = r11.astype(np.int16) - r4.astype(np.int16)
        diff_array[valid] = diff_vals[valid]

        # Gán mã phân cấp biến động
        cond_giam = valid & (diff_vals < 0)
        cond_on_dinh = valid & (diff_vals == 0)
        cond_tang1 = valid & (diff_vals == 1)
        cond_tang2 = valid & (diff_vals == 2)
        cond_tang_manh = valid & (diff_vals >= 3)

        shift_class[cond_giam] = 1
        shift_class[cond_on_dinh] = 2
        shift_class[cond_tang1] = 3
        shift_class[cond_tang2] = 4
        shift_class[cond_tang_manh] = 5

        for cls_code, color in SHIFT_COLORS.items():
            rgba_shift[shift_class == cls_code] = color

        # Ghi file GeoTIFF bản đồ biến động nguy cơ
        out_shift_tif = os.path.join(OUTPUT_DIR_COMBINED, f'ban_do_bien_dong_nguy_co_{model_code}.tif')
        prof = s4.profile.copy()
        prof.update(dtype=rasterio.int16, count=1, nodata=-9999, compress='lzw')
        with rasterio.open(out_shift_tif, 'w', **prof) as dst:
            dst.write(shift_class.reshape((1, height, width)))

        # Ghi file màu .clr
        clr_shift = os.path.splitext(out_shift_tif)[0] + ".clr"
        with open(clr_shift, "w", encoding="utf-8") as f:
            for c_code, col in sorted(SHIFT_COLORS.items()):
                f.write(f"{c_code} {col[0]} {col[1]} {col[2]}\n")

        # Ghi file PNG RGBA cho WebGIS
        out_shift_png = os.path.join(OUTPUT_DIR_COMBINED, f'webgis_bien_dong_{model_code}_rgba.png')
        if Image is not None:
            Image.fromarray(rgba_shift, mode="RGBA").save(out_shift_png, format="PNG")
        else:
            plt.imsave(out_shift_png, rgba_shift)

        # Chuyển đổi BBox sang WGS84
        west, south, east, north = transform_bounds(crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top)
        bbox_info = {
            "type": "Hazard Dynamics Shift",
            "model": model_name,
            "crs": "EPSG:4326",
            "png_file": os.path.basename(out_shift_png),
            "bounds_geojson": [west, south, east, north],
            "leaflet_bounds": [[south, west], [north, east]],
            "center": [(south + north) / 2.0, (west + east) / 2.0]
        }
        out_shift_json = os.path.join(OUTPUT_DIR_COMBINED, f'webgis_bien_dong_{model_code}_bbox.json')
        with open(out_shift_json, "w", encoding="utf-8") as f:
            json.dump(bbox_info, f, indent=4, ensure_ascii=False)

        print(f"[+] [Cả 2 tháng] Xuất GeoTIFF biến động nguy cơ ({model_code}): {out_shift_tif}")
        print(f"[+] [Cả 2 tháng] Xuất WebGIS PNG biến động ({model_code}):    {out_shift_png}")

        # Thống kê tổng diện tích theo từng mức độ biến động
        stat_summary = {
            "Giảm nguy cơ (Hồi phục)": np.sum(cond_giam) * PIXEL_AREA_HA,
            "Ổn định (Không đổi)": np.sum(cond_on_dinh) * PIXEL_AREA_HA,
            "Tăng nhẹ (+1 cấp)": np.sum(cond_tang1) * PIXEL_AREA_HA,
            "Tăng đáng kể (+2 cấp)": np.sum(cond_tang2) * PIXEL_AREA_HA,
            "Tăng rất mạnh (>= 3 cấp / Sạt mới)": np.sum(cond_tang_manh) * PIXEL_AREA_HA,
        }
        print("\n[+] BẢNG TỔNG HỢP BIẾN ĐỘNG NGUY CƠ TRƯỢT LỞ (ha):")
        for k, v in stat_summary.items():
            print(f"    - {k:<38}: {v:8.2f} ha ({v / (len(v4)*PIXEL_AREA_HA)*100:5.2f}%)")

        # Vẽ biểu đồ so sánh biến động nguy cơ
        plot_hazard_shift_chart(ct_hz_ha, stat_summary, model_code, model_name)

        return stat_summary


def plot_hazard_shift_chart(ct_hz_ha, stat_summary, model_code, model_name):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

        # 1. So sánh diện tích 5 cấp giữa Tháng 4 và Tháng 11
        levels = [1, 2, 3, 4, 5]
        t4_areas = [ct_hz_ha.loc[lvl, "Tổng cộng"] for lvl in levels]
        t11_areas = [ct_hz_ha.loc["Tổng cộng", lvl] for lvl in levels]

        x = np.arange(len(levels))
        width = 0.35
        axes[0].bar(x - width/2, t4_areas, width, label='Tháng 4/2024 (Trước bão)', color='#4292c6')
        axes[0].bar(x + width/2, t11_areas, width, label='Tháng 11/2024 (Sau bão)', color='#ef3b2c')
        axes[0].set_title(f"So sánh Diện tích 5 Cấp Nguy cơ Trượt lở ({model_name})", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Cấp độ nguy cơ", fontsize=10)
        axes[0].set_ylabel("Diện tích (ha)", fontsize=10)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"Cấp {lvl}\n({HAZARD_LEVEL_NAMES[lvl].split()[0]})" for lvl in levels])
        axes[0].legend()
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)

        # 2. Biểu đồ thanh ngang mức độ biến động
        cats = list(stat_summary.keys())
        vals = list(stat_summary.values())
        colors = ['#2b83ba', '#bababa', '#fecc5c', '#fd8d3c', '#e31a1c']
        axes[1].barh(cats, vals, color=colors, edgecolor='#333333', height=0.6)
        for i, v in enumerate(vals):
            axes[1].text(v + 15, i, f"{v:.1f} ha", va='center', fontsize=9, fontweight='bold')
        axes[1].set_title("Phân bố Mức độ Dịch chuyển Nguy cơ sau Bão Yagi", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Diện tích (ha)", fontsize=10)
        axes[1].set_xlim(0, max(vals) * 1.15)
        axes[1].grid(axis='x', linestyle='--', alpha=0.5)

        plt.tight_layout()
        chart_p = os.path.join(OUTPUT_DIR_COMBINED, f'bieu_do_dich_chuyen_nguy_co_{model_code}.png')
        plt.savefig(chart_p)
        plt.close()
        print(f"[+] Đã lưu biểu đồ biến động nguy cơ: {chart_p}")
    except Exception as e:
        print(f"[!] Bỏ qua biểu đồ nguy cơ: {e}")


# ==============================================================================
# PHẦN 3: MÔ HÌNH HỌC MÁY ĐA THỜI KỲ KẾT HỢP (COMBINED BI-TEMPORAL ML MODEL)
# ==============================================================================
def train_combined_bi_temporal_model(random_state=42):
    print("\n" + "=" * 80)
    print("  PHẦN 3: HUẤN LUYỆN MÔ HÌNH HỌC MÁY ĐA THỜI KỲ KẾT HỢP (THÁNG 4 + THÁNG 11)")
    print("=" * 80)

    # 1. Lấy dữ liệu mẫu từ cả 2 tháng
    X4, y4 = extract_samples_th4()
    X11, y11 = extract_samples_th11()

    # Chuẩn hóa nhãn nhị phân cho bài toán chung:
    # 1: Vùng sạt trượt / trơ sỏi đá / đất trống có nguy cơ (Lớp 22 ở T4, Lớp 21 ở T11)
    # 0: Vùng an toàn / thảm phủ thực vật / mặt nước (các lớp còn lại)
    y4_bin = np.where(y4 == 22, 1, 0)
    y11_bin = np.where(y11 == 21, 1, 0)

    X_comb = np.vstack([X4, X11])
    y_comb = np.concatenate([y4_bin, y11_bin])

    print(f"[+] Tổng số mẫu gộp cả 2 mùa: {len(y_comb)} mẫu ({np.sum(y_comb==1)} Sạt trượt : {np.sum(y_comb==0)} An toàn)")

    # 2. Phân chia Train/Test 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X_comb, y_comb, test_size=0.30, random_state=random_state, stratify=y_comb
    )

    # 3. Huấn luyện Random Forest kết hợp
    print("[*] Huấn luyện Random Forest kết hợp đa thời vụ...")
    rf_comb = RandomForestClassifier(n_estimators=150, max_features='sqrt', random_state=random_state, n_jobs=-1)
    rf_comb.fit(X_train, y_train)

    # 4. Huấn luyện SVM kết hợp đa thời vụ
    print("[*] Huấn luyện SVM Pipeline kết hợp đa thời vụ...")
    svm_comb = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10.0, kernel='rbf', gamma='scale', probability=True, random_state=random_state))
    ])
    svm_comb.fit(X_train, y_train)

    # Đánh giá
    eval_dict = {}
    for name, model in [('Random Forest (Combined)', rf_comb), ('SVM (Combined)', svm_comb)]:
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probas)
        kappa = cohen_kappa_score(y_test, preds)
        eval_dict[name] = {'Acc': acc, 'F1': f1, 'AUC': auc, 'Kappa': kappa}

    print("\n" + "=" * 70)
    print("  HIỆU NĂNG MÔ HÌNH HỌC MÁY ĐA THỜI KỲ (COMBINED TEMPORAL MODEL)")
    print("=" * 70)
    print(f"{'Mô hình':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Kappa':<10}")
    print("-" * 70)
    for name, m in eval_dict.items():
        print(f"{name:<30} | {m['Acc']*100:6.2f}%   | {m['F1']:8.4f} | {m['AUC']:8.4f} | {m['Kappa']:8.4f}")
    print("=" * 70)

    # Lưu 2 mô hình kết hợp
    rf_pkl = os.path.join(OUTPUT_DIR_COMBINED, 'mo_hinh_rf_ket_hop_ca_2_thang.pkl')
    svm_pkl = os.path.join(OUTPUT_DIR_COMBINED, 'mo_hinh_svm_ket_hop_ca_2_thang.pkl')
    joblib.dump(rf_comb, rf_pkl)
    joblib.dump(svm_comb, svm_pkl)
    print(f"[+] Đã lưu mô hình RF kết hợp:  {rf_pkl}")
    print(f"[+] Đã lưu mô hình SVM kết hợp: {svm_pkl}")

    return rf_comb, svm_comb


# ==============================================================================
# PHẦN 4: TỰ ĐỘNG SINH ỨNG DỤNG WEBGIS ĐA THỜI GIAN (BI-TEMPORAL WEBGIS)
# ==============================================================================
def generate_bitemporal_webgis_html():
    print("\n[*] Tự động khởi tạo ứng dụng WebGIS Đa thời gian tương tác (Tháng 4 vs Tháng 11)...")

    # Đọc toạ độ bounding box từ Tháng 4
    bbox_path = os.path.join(DIR_TH4, 'thang4_webgis_rf_bbox.json')
    if not os.path.exists(bbox_path):
        return

    with open(bbox_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    center = meta['center']
    bounds = meta['leaflet_bounds']

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebGIS Đa Thời Gian - Đánh Giá Biến Động Nguy Cơ Trượt Lở Đất (Tháng 4 vs Tháng 11/2024)</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        #map {{ width: 100%; height: 100%; }}
        .header-panel {{
            position: absolute; top: 12px; left: 60px; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); padding: 12px 18px;
            border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);
            max-width: 520px; border-left: 5px solid #e31a1c;
        }}
        .header-panel h2 {{ margin: 0 0 4px 0; font-size: 15px; color: #800026; text-transform: uppercase; font-weight: bold; }}
        .header-panel p {{ margin: 0; font-size: 12px; color: #444; line-height: 1.4; }}
        .control-panel {{
            position: absolute; top: 12px; right: 12px; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); padding: 16px;
            border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);
            width: 280px; font-size: 12px;
        }}
        .control-panel h4 {{ margin: 0 0 10px 0; font-size: 13px; color: #111; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .section-title {{ font-weight: bold; margin-top: 10px; margin-bottom: 6px; color: #1e3d59; }}
        .layer-item {{ margin-bottom: 6px; }}
        .slider-box {{ margin-top: 12px; border-top: 1px dashed #ccc; padding-top: 8px; }}
        .slider-box label {{ display: block; margin-bottom: 3px; font-weight: bold; }}
        .legend-panel {{
            position: absolute; bottom: 25px; right: 12px; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); padding: 12px 15px;
            border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);
            font-size: 11px; width: 220px;
        }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 5px; }}
        .legend-color {{ width: 22px; height: 14px; margin-right: 8px; border-radius: 3px; border: 1px solid #777; }}
        .stat-badge {{
            display: inline-block; padding: 2px 6px; border-radius: 4px;
            font-size: 11px; font-weight: bold; background: #fee0d2; color: #99000d; margin-top: 4px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="header-panel">
        <h2>WebGIS Đa Thời Gian: Nguy Cơ Trượt Lở Đất</h2>
        <p><strong>Khu vực:</strong> Xã Mường Lai, Lục Yên, Yên Bái</p>
        <p><strong>So sánh đối chứng:</strong> Tháng 4/2024 (Trước bão) vs Tháng 11/2024 (Sau bão số 3 Yagi)</p>
        <span class="stat-badge">Mô hình: Random Forest & Support Vector Machine</span>
    </div>
    <div class="control-panel">
        <h4>ĐIỀU KHIỂN ĐA THỜI GIAN</h4>
        
        <div class="section-title">1. Chọn Thời Điểm Quan Sát:</div>
        <div class="layer-item">
            <input type="radio" id="time_t4" name="time_select" value="t4">
            <label for="time_t4"><strong>Tháng 4/2024</strong> (Mùa khô - Trước bão)</label>
        </div>
        <div class="layer-item">
            <input type="radio" id="time_t11" name="time_select" value="t11">
            <label for="time_t11"><strong>Tháng 11/2024</strong> (Sau bão Yagi)</label>
        </div>
        <div class="layer-item">
            <input type="radio" id="time_diff" name="time_select" value="diff" checked>
            <label for="time_diff"><strong style="color:#d95f02;">Bản đồ Biến động Nguy cơ</strong> (Diff)</label>
        </div>

        <div class="section-title">2. Chọn Thuật Toán Học Máy:</div>
        <div class="layer-item">
            <input type="radio" id="algo_rf" name="algo_select" value="rf" checked>
            <label for="algo_rf">Random Forest (RF)</label>
        </div>
        <div class="layer-item">
            <input type="radio" id="algo_svm" name="algo_select" value="svm">
            <label for="algo_svm">Support Vector Machine (SVM)</label>
        </div>

        <div class="slider-box">
            <label>Độ trong suốt lớp phủ: <span id="opacity_val">85%</span></label>
            <input type="range" id="opacity_slider" min="0" max="100" value="85" style="width: 100%;">
        </div>
    </div>

    <div class="legend-panel" id="legend_box">
        <!-- Chú giải động cập nhật theo lớp đang chọn -->
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('map').setView([{center[0]}, {center[1]}], 13);
        const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: 'OSM' }});
        const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{ maxZoom: 19, attribution: 'ESRI' }}).addTo(map);

        L.control.layers({{ 'Ảnh vệ tinh ESRI': esri, 'Bản đồ nền OSM': osm }}).addTo(map);

        const bounds = {bounds};

        // Đường dẫn tương đối tới các ảnh PNG RGBA
        const overlays = {{
            't4_rf': L.imageOverlay('../results_thang_4/thang4_webgis_rf_rgba.png', bounds, {{ opacity: 0.85 }}),
            't4_svm': L.imageOverlay('../results_thang_4/thang4_webgis_svm_rgba.png', bounds, {{ opacity: 0.85 }}),
            't11_rf': L.imageOverlay('../results_thang_11/thang11_webgis_rf_rgba.png', bounds, {{ opacity: 0.85 }}),
            't11_svm': L.imageOverlay('../results_thang_11/thang11_webgis_svm_rgba.png', bounds, {{ opacity: 0.85 }}),
            'diff_rf': L.imageOverlay('webgis_bien_dong_rf_rgba.png', bounds, {{ opacity: 0.85 }}),
            'diff_svm': L.imageOverlay('webgis_bien_dong_svm_rgba.png', bounds, {{ opacity: 0.85 }})
        }};

        let currentActiveLayer = null;

        function updateLayer() {{
            const timeVal = document.querySelector('input[name="time_select"]:checked').value;
            const algoVal = document.querySelector('input[name="algo_select"]:checked').value;
            const key = timeVal + '_' + algoVal;

            if (currentActiveLayer) {{
                map.removeLayer(currentActiveLayer);
            }}

            currentActiveLayer = overlays[key];
            if (currentActiveLayer) {{
                const op = document.getElementById('opacity_slider').value / 100.0;
                currentActiveLayer.setOpacity(op);
                map.addLayer(currentActiveLayer);
            }}

            updateLegend(timeVal);
        }}

        function updateLegend(timeVal) {{
            const leg = document.getElementById('legend_box');
            if (timeVal === 'diff') {{
                leg.innerHTML = `
                    <strong>CHÚ GIẢI BIẾN ĐỘNG NGUY CƠ</strong>
                    <div class="legend-item" style="margin-top:8px;"><div class="legend-color" style="background:#2b83ba;"></div> Giảm nguy cơ (Hồi phục)</div>
                    <div class="legend-item"><div class="legend-color" style="background:#e0e0e0;"></div> Không đổi (Ổn định)</div>
                    <div class="legend-item"><div class="legend-color" style="background:#fecc5c;"></div> Tăng 1 cấp nguy cơ</div>
                    <div class="legend-item"><div class="legend-color" style="background:#fd8d3c;"></div> Tăng 2 cấp nguy cơ</div>
                    <div class="legend-item"><div class="legend-color" style="background:#e31a1c;"></div> Tăng rất mạnh (>= 3 cấp / Sạt mới)</div>
                `;
            }} else {{
                leg.innerHTML = `
                    <strong>CHÚ GIẢI 5 CẤP NGUY CƠ</strong>
                    <div class="legend-item" style="margin-top:8px;"><div class="legend-color" style="background:#38a800;"></div> Cấp 1: Rất thấp (0 - 0.2)</div>
                    <div class="legend-item"><div class="legend-color" style="background:#b0e000;"></div> Cấp 2: Thấp (0.2 - 0.4)</div>
                    <div class="legend-item"><div class="legend-color" style="background:#ffff00;"></div> Cấp 3: Trung bình (0.4 - 0.6)</div>
                    <div class="legend-item"><div class="legend-color" style="background:#ff8000;"></div> Cấp 4: Cao (0.6 - 0.8)</div>
                    <div class="legend-item"><div class="legend-color" style="background:#ff0000;"></div> Cấp 5: Rất cao (0.8 - 1.0)</div>
                `;
            }}
        }}

        document.querySelectorAll('input[name="time_select"]').forEach(el => el.addEventListener('change', updateLayer));
        document.querySelectorAll('input[name="algo_select"]').forEach(el => el.addEventListener('change', updateLayer));

        document.getElementById('opacity_slider').addEventListener('input', (e) => {{
            const val = e.target.value / 100.0;
            document.getElementById('opacity_val').innerText = e.target.value + '%';
            if (currentActiveLayer) {{
                currentActiveLayer.setOpacity(val);
            }}
        }});

        updateLayer();
        map.fitBounds(bounds);
    </script>
</body>
</html>
"""
    webgis_path = os.path.join(OUTPUT_DIR_COMBINED, 'webgis_viewer_bitemporal.html')
    with open(webgis_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[+] [Cả 2 tháng] Đã tạo ứng dụng WebGIS Đa thời gian: {webgis_path}")


# ==============================================================================
# HÀM THỰC THI CHÍNH - CẢ 2 THÁNG
# ==============================================================================
def run_pipeline_ca_2_thang():
    print("=" * 80)
    print("  BẮT ĐẦU QUY TRÌNH PHÂN TÍCH VÀ MÔ HÌNH HÓA ĐA THỜI KỲ (CẢ 2 THÁNG)")
    print("  KHU VỰC: XÃ MƯỜNG LAI, HUYỆN LỤC YÊN, TỈNH YÊN BÁI")
    print("=" * 80)

    # 1. Phân tích ma trận chuyển đổi lớp phủ LULC
    analyze_lulc_change()

    # 2. Phân tích dịch chuyển nguy cơ trượt lở cho cả RF và SVM
    analyze_hazard_shift(model_code="rf", model_name="Random Forest")
    analyze_hazard_shift(model_code="svm", model_name="Support Vector Machine")

    # 3. Huấn luyện mô hình học máy đa thời vụ kết hợp
    train_combined_bi_temporal_model(random_state=42)

    # 4. Tự động sinh giao diện WebGIS Đa thời gian tương tác
    generate_bitemporal_webgis_html()

    print("\n" + "=" * 80)
    print("  HOÀN THÀNH QUY TRÌNH PHÂN TÍCH CẢ 2 THÁNG THÀNH CÔNG!")
    print(f"  Tất cả tệp kết quả lưu tại: {OUTPUT_DIR_COMBINED}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_ca_2_thang()
