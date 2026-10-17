# -*- coding: utf-8 -*-
"""
================================================================================
THUẬT TOÁN RANDOM FOREST (RF) - RIÊNG THÁNG 11/2024 (SAU BÃO YAGI)
ĐỀ TÀI: KHOANH VÙNG NGUY CƠ TRƯỢT LỞ ĐẤT XÃ MƯỜNG LAI, LỤC YÊN, YÊN BÁI
ẢNH LANDSAT 8 (30/11/2024) - BỘ DỮ LIỆU KIỂM CHỨNG DKT11 (200 MẪU)
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

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU ĐẦU VÀO VÀ ĐẦU RA - RANDOM FOREST THÁNG 11
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(BASE_DIR, 'datath11', 'Extrac11.tif')
ROI_PATH = os.path.join(BASE_DIR, 'DKTth11', 'DKT11.shp')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'results_thang_11')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File lưu mô hình huấn luyện Random Forest Tháng 11
MODEL_SAVE_PATH = os.path.join(WORKSPACE_DIR, 'rf_model_thang_11.pkl')

CLASS_COLUMN_NAME = "Nhan"
LANDSLIDE_CLASS = 21  # Lớp đất trống / vết sạt lở mới sau bão Yagi Tháng 11

CLASS_NAMES = {
    1: "Thảm thực vật / Cây bụi",
    21: "Đất trống / Vết sạt trượt đất mới (sau bão)",
    41: "Mặt nước / Thủy vực",
    46: "Rừng tự nhiên / Sinh khối cao"
}

FEATURE_NAMES = [
    'Kênh 1 (Coastal Aerosol)',
    'Kênh 2 (Blue)',
    'Kênh 3 (Green)',
    'Kênh 4 (Red)',
    'Kênh 5 (NIR)',
    'Kênh 6 (SWIR 1)',
    'Kênh 7 (SWIR 2)',
    'Chỉ số NDVI (Thực vật)',
    'Chỉ số NDBI (Đất khô/Xây dựng)',
    'Chỉ số MNDWI (Nước mặt tăng cường)',
    'Chỉ số NDWI (Độ ẩm/Mặt nước)',
    'Chỉ số BSI (Chỉ số đất trống/sạt lở)'
]

HAZARD_COLORS = {
    1: [56, 168, 0, 220],     # Cấp 1: Rất thấp (Xanh lá)
    2: [176, 224, 0, 220],    # Cấp 2: Thấp (Xanh lục nhạt)
    3: [255, 255, 0, 220],    # Cấp 3: Trung bình (Vàng)
    4: [255, 128, 0, 230],    # Cấp 4: Cao (Da cam)
    5: [255, 0, 0, 240],      # Cấp 5: Rất cao (Đỏ)
}


# ==============================================================================
# HÀM TÍNH TOÁN 5 CHỈ SỐ PHỔ VIỄN THÁM (12 KÊNH ĐẶC TRƯNG)
# ==============================================================================
def compute_spectral_indices(b1, b2, b3, b4, b5, b6, b7):
    with np.errstate(divide='ignore', invalid='ignore'):
        denom_ndvi = b5 + b4
        ndvi = np.where(denom_ndvi != 0, (b5 - b4) / denom_ndvi, 0.0)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

        denom_ndbi = b6 + b5
        ndbi = np.where(denom_ndbi != 0, (b6 - b5) / denom_ndbi, 0.0)
        ndbi = np.nan_to_num(ndbi, nan=0.0, posinf=0.0, neginf=0.0)

        denom_mndwi = b3 + b6
        mndwi = np.where(denom_mndwi != 0, (b3 - b6) / denom_mndwi, 0.0)
        mndwi = np.nan_to_num(mndwi, nan=0.0, posinf=0.0, neginf=0.0)

        denom_ndwi = b3 + b5
        ndwi = np.where(denom_ndwi != 0, (b3 - b5) / denom_ndwi, 0.0)
        ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)

        num_bsi = (b6 + b4) - (b5 + b2)
        denom_bsi = (b6 + b4) + (b5 + b2)
        bsi = np.where(denom_bsi != 0, num_bsi / denom_bsi, 0.0)
        bsi = np.nan_to_num(bsi, nan=0.0, posinf=0.0, neginf=0.0)

    return ndvi, ndbi, mndwi, ndwi, bsi


def add_spectral_indices_to_features(X_7bands):
    X_f = X_7bands.astype(np.float32)
    b1, b2, b3, b4, b5, b6, b7 = [X_f[:, i] for i in range(7)]
    ndvi, ndbi, mndwi, ndwi, bsi = compute_spectral_indices(b1, b2, b3, b4, b5, b6, b7)
    return np.column_stack([X_f, ndvi, ndbi, mndwi, ndwi, bsi])


# ==============================================================================
# TRÍCH XUẤT MẪU VÀ PHÂN CHIA TẬP DỮ LIỆU
# ==============================================================================
def extract_samples():
    print(f"[*] [RF Tháng 11] Đọc ảnh Landsat 8: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {IMAGE_PATH}")
    if not os.path.exists(ROI_PATH):
        raise FileNotFoundError(f"Không tìm thấy shapefile: {ROI_PATH}")

    with rasterio.open(IMAGE_PATH) as src:
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else 0
        roi_gdf = gpd.read_file(ROI_PATH)

        if roi_gdf.crs != crs:
            print(f"[*] Đồng bộ CRS {roi_gdf.crs} -> {crs}")
            roi_gdf = roi_gdf.to_crs(crs)

        pts, labels = [], []
        for _, row in roi_gdf.iterrows():
            geom = row.geometry
            lbl = row[CLASS_COLUMN_NAME] if CLASS_COLUMN_NAME in row else 1
            if geom is not None and not geom.is_empty:
                if geom.geom_type == 'Point':
                    pts.append((geom.x, geom.y))
                    labels.append(lbl)
                elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                    pts.append((geom.centroid.x, geom.centroid.y))
                    labels.append(lbl)

        samples = np.array(list(src.sample(pts)), dtype=np.float32)
        valid = (samples != nodata).all(axis=1) & np.isfinite(samples).all(axis=1)
        X_7 = samples[valid]
        y = np.array(labels)[valid]

    print(f"[+] [RF Tháng 11] Trích xuất thành công {len(y)} điểm mẫu từ Shapefile DKT11.")
    X_12 = add_spectral_indices_to_features(X_7)
    return X_12, y


def split_data(X, y, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    print(f"    - Tập Huấn luyện (Train): {len(X_train)} mẫu (70%)")
    print(f"    - Tập Đánh giá (Val):     {len(X_val)} mẫu (15%)")
    print(f"    - Tập Kiểm thử (Test):    {len(X_test)} mẫu (15%)")
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


# ==============================================================================
# HUẤN LUYỆN VÀ ĐÁNH GIÁ RANDOM FOREST
# ==============================================================================
def train_and_evaluate_rf(data_dict, random_state=42):
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

    print("\n[*] [RF Tháng 11] Đang huấn luyện Random Forest (n_estimators=150, max_features='sqrt')...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=150,
        max_features='sqrt',
        oob_score=True,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    fit_time = time.time() - t0

    preds = rf.predict(X_test)
    probas = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
    except Exception:
        auc = 0.0

    print("=" * 65)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST - THÁNG 11/2024")
    print("=" * 65)
    print(f"  - Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%")
    print(f"  - Precision:                        {prec * 100:.2f}%")
    print(f"  - Recall:                           {rec * 100:.2f}%")
    print(f"  - F1-Score:                         {f1:.4f}")
    print(f"  - Hệ số Cohen's Kappa:              {kappa:.4f}")
    print(f"  - ROC-AUC:                          {auc:.4f}")
    print(f"  - OOB Score (Out-of-Bag):           {rf.oob_score_ * 100:.2f}%")
    print(f"  - Thời gian huấn luyện:             {fit_time:.3f} giây")
    print("\nBáo cáo chi tiết (Classification Report):")
    print(classification_report(y_test, preds, zero_division=0))

    # Lưu mô hình .pkl
    joblib.dump(rf, MODEL_SAVE_PATH)
    print(f"[+] Đã lưu mô hình RF Tháng 11 tại: {MODEL_SAVE_PATH}")

    # Xuất đồ thị Feature Importance & Confusion Matrix
    plot_evaluation(rf, X_test, y_test, preds)

    return rf


def plot_evaluation(rf, X_test, y_test, preds):
    # 1. Feature Importance
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    imp = rf.feature_importances_ * 100
    idx = np.argsort(imp)
    ax.barh([FEATURE_NAMES[i] for i in idx], imp[idx], color='#2b8cbe', edgecolor='#084081')
    ax.set_title("Random Forest - Độ quan trọng đặc trưng (Tháng 11/2024 - Sau bão Yagi)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Đóng góp (%)", fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'rf_thang11_feature_importance.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"[+] Đã xuất biểu đồ Feature Importance: {chart_path}")

    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cm = confusion_matrix(y_test, preds)
    labels = [f"Lớp {c}" for c in rf.classes_]
    im = ax.matshow(cm, cmap='Blues', alpha=0.85)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, str(val), ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white' if val > cm.max()/2 else 'black')
    ax.set_title('Ma trận nhầm lẫn Random Forest (Tháng 11)', fontsize=11, fontweight='bold', pad=15)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Dự đoán', fontsize=9)
    ax.set_ylabel('Thực tế', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'rf_thang11_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"[+] Đã xuất biểu đồ Ma trận nhầm lẫn: {cm_path}")


# ==============================================================================
# DỰ ĐOÁN TOÀN VÙNG VÀ XUẤT RASTER / WEBGIS
# ==============================================================================
def classify_and_export(rf):
    print(f"\n[*] [RF Tháng 11] Bắt đầu suy luận không gian toàn ảnh...")
    with rasterio.open(IMAGE_PATH) as src:
        height, width = src.height, src.width
        crs, bounds = src.crs, src.bounds
        nodata = src.nodata

        profile_discrete = src.profile.copy()
        profile_discrete.update(dtype=rasterio.int32, count=1, nodata=-9999, compress='lzw')

        profile_hazard = src.profile.copy()
        profile_hazard.update(dtype=rasterio.int16, count=1, nodata=-9999, compress='lzw')

        profile_proba = src.profile.copy()
        profile_proba.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='lzw')

        classes = list(rf.classes_)
        target_idx = classes.index(LANDSLIDE_CLASS) if LANDSLIDE_CLASS in classes else len(classes) - 1
        print(f"    - Lớp chỉ điểm nguy cơ trượt lở: '{classes[target_idx]}' (index={target_idx})")

        out_discrete = os.path.join(OUTPUT_DIR, 'thang11_phan_loai_rf.tif')
        out_hazard = os.path.join(OUTPUT_DIR, 'thang11_nguy_co_5cap_rf.tif')
        out_proba = os.path.join(OUTPUT_DIR, 'thang11_xac_suat_rf.tif')
        out_png = os.path.join(OUTPUT_DIR, 'thang11_webgis_rf_rgba.png')
        out_bbox = os.path.join(OUTPUT_DIR, 'thang11_webgis_rf_bbox.json')
        out_clr = os.path.join(OUTPUT_DIR, 'thang11_nguy_co_5cap_rf.clr')

        full_data = src.read()
        flat_7 = full_data.reshape(src.count, -1).T

        if nodata is not None:
            valid_mask = (flat_7 != nodata).all(axis=1) & np.isfinite(flat_7).all(axis=1)
        else:
            valid_mask = (flat_7 != 0).any(axis=1) & np.isfinite(flat_7).all(axis=1)

        flat_discrete = np.full(flat_7.shape[0], -9999, dtype=np.int32)
        flat_hazard = np.full(flat_7.shape[0], -9999, dtype=np.int16)
        flat_proba = np.full(flat_7.shape[0], -9999.0, dtype=np.float32)

        rgba_img = np.zeros((height, width, 4), dtype=np.uint8)

        if np.any(valid_mask):
            v7 = flat_7[valid_mask]
            v12 = add_spectral_indices_to_features(v7)

            flat_discrete[valid_mask] = rf.predict(v12)
            probas = rf.predict_proba(v12)
            p_ls = probas[:, target_idx]
            flat_proba[valid_mask] = p_ls

            h_levels = np.digitize(p_ls, bins=[0.2, 0.4, 0.6, 0.8]) + 1
            h_levels = np.clip(h_levels, 1, 5).astype(np.int16)
            flat_hazard[valid_mask] = h_levels

            h_2d = flat_hazard.reshape((height, width))
            for lvl, col in HAZARD_COLORS.items():
                mask_lvl = (h_2d == lvl)
                rgba_img[mask_lvl] = col

        with rasterio.open(out_discrete, 'w', **profile_discrete) as dst:
            dst.write(flat_discrete.reshape((height, width)), 1)
        with rasterio.open(out_hazard, 'w', **profile_hazard) as dst:
            dst.write(flat_hazard.reshape((height, width)), 1)
        with rasterio.open(out_proba, 'w', **profile_proba) as dst:
            dst.write(flat_proba.reshape((height, width)), 1)

        if Image:
            Image.fromarray(rgba_img).save(out_png)

        bbox_4326 = transform_bounds(crs, 'EPSG:4326', *bounds)
        bbox_info = {
            'south': bbox_4326[1], 'west': bbox_4326[0],
            'north': bbox_4326[3], 'east': bbox_4326[2],
            'bounds': [[bbox_4326[1], bbox_4326[0]], [bbox_4326[3], bbox_4326[2]]],
            'center': [(bbox_4326[1] + bbox_4326[3])/2, (bbox_4326[0] + bbox_4326[2])/2]
        }
        with open(out_bbox, 'w', encoding='utf-8') as f:
            json.dump(bbox_info, f, indent=2)

        with open(out_clr, 'w') as f:
            f.write("1 56 168 0\n2 176 224 0\n3 255 255 0\n4 255 128 0\n5 255 0 0\n")

        print(f"[+] [RF Tháng 11] GeoTIFF phân loại:    {out_discrete}")
        print(f"[+] [RF Tháng 11] GeoTIFF 5 cấp nguy cơ: {out_hazard}")
        print(f"[+] [RF Tháng 11] GeoTIFF xác suất:     {out_proba}")
        print(f"[+] [RF Tháng 11] WebGIS PNG RGBA:       {out_png}")


def main():
    print("""
================================================================================
  MÔ HÌNH RANDOM FOREST - RIÊNG THÁNG 11/2024 (SAU BÃO YAGI)
  KHU VỰC: XÃ MƯỜNG LAI, HUYỆN LỤC YÊN, TỈNH YÊN BÁI
================================================================================
    """)
    X, y = extract_samples()
    data_dict = split_data(X, y)
    rf = train_and_evaluate_rf(data_dict)
    classify_and_export(rf)
    print("\n[+] HOÀN THÀNH TOÀN BỘ QUY TRÌNH RANDOM FOREST THÁNG 11/2024!")


if __name__ == '__main__':
    main()
