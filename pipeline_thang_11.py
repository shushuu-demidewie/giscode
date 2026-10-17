# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH PHÂN LOẠI VÀ KHOANH VÙNG NGUY CƠ TRƯỢT LỞ ĐẤT - RIÊNG THÁNG 11/2024
ỨNG DỤNG MÔ HÌNH RANDOM FOREST (RF) VÀ SUPPORT VECTOR MACHINE (SVM)
KHU VỰC: XÃ MƯỜNG LAI, HUYỆN LỤC YÊN, TỈNH YÊN BÁI (ẢNH LANDSAT 8: 30/11/2024 - SAU BÃO YAGI)
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
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve
)
import joblib

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU ĐẦU VÀO VÀ ĐẦU RA - THÁNG 11
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_TH11_PATH = os.path.join(BASE_DIR, 'datath11', 'Extrac11.tif')
ROI_TH11_PATH = os.path.join(BASE_DIR, 'DKTth11', 'DKT11.shp')
REF_PLTH11_PATH = os.path.join(BASE_DIR, 'datath11', 'plth11.tif')

OUTPUT_DIR_TH11 = os.path.join(WORKSPACE_DIR, 'results_thang_11')
os.makedirs(OUTPUT_DIR_TH11, exist_ok=True)

# Tên các lớp phân loại cho Tháng 11
CLASS_NAMES_TH11 = {
    1: "Thảm thực vật / Cây bụi",
    21: "Đất trống / Vết sạt trượt đất mới (sau bão)",
    41: "Mặt nước / Thủy vực",
    46: "Rừng tự nhiên / Sinh khối cao"
}
LANDSLIDE_CLASS_TH11 = 21  # Lớp sạt trượt / đất trống đặc trưng sau bão

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
    """Tính 5 chỉ số phổ từ 7 kênh quang học Landsat 8."""
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
    """Ghép 5 chỉ số phổ vào ma trận 7 kênh ban đầu, trả về 12 kênh đặc trưng."""
    X_f = X_7bands.astype(np.float32)
    b1, b2, b3, b4, b5, b6, b7 = [X_f[:, i] for i in range(7)]
    ndvi, ndbi, mndwi, ndwi, bsi = compute_spectral_indices(b1, b2, b3, b4, b5, b6, b7)
    return np.column_stack([X_f, ndvi, ndbi, mndwi, ndwi, bsi])


# ==============================================================================
# HÀM TRÍCH XUẤT MẪU VÀ CHIA TẬP DỮ LIỆU
# ==============================================================================
def extract_samples(image_path=IMAGE_TH11_PATH, shp_path=ROI_TH11_PATH, class_col="Nhan"):
    print(f"[*] [Tháng 11] Đọc ảnh Landsat 8: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Không tìm thấy shapefile: {shp_path}")

    with rasterio.open(image_path) as src:
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else 0
        roi_gdf = gpd.read_file(shp_path)

        if roi_gdf.crs != crs:
            print(f"[*] Đồng bộ CRS {roi_gdf.crs} -> {crs}")
            roi_gdf = roi_gdf.to_crs(crs)

        pts = []
        labels = []
        for _, row in roi_gdf.iterrows():
            geom = row.geometry
            lbl = row[class_col] if class_col in row else 1
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

    print(f"[+] [Tháng 11] Trích xuất {len(y)} điểm mẫu từ Shapefile DKT11.")
    X_12 = add_spectral_indices_to_features(X_7)
    return X_12, y


def split_dataset(X, y, random_state=42):
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
# HUẤN LUYỆN VÀ ĐÁNH GIÁ RANDOM FOREST & SVM
# ==============================================================================
def train_and_evaluate_models(data_dict, random_state=42):
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

    # 1. Huấn luyện Random Forest
    print("\n[*] [Tháng 11] Huấn luyện mô hình Random Forest (n_estimators=150)...")
    t0_rf = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_features='sqrt',
        oob_score=True,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - t0_rf

    # 2. Huấn luyện SVM Pipeline
    print(f"[*] [Tháng 11] Huấn luyện SVM Pipeline (StandardScaler + SVC RBF, C=10.0)...")
    t0_svm = time.time()
    svm_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10.0, kernel='rbf', gamma='scale', probability=True, random_state=random_state))
    ])
    svm_model.fit(X_train, y_train)
    svm_time = time.time() - t0_svm

    # Đánh giá 2 mô hình
    metrics = {}
    for name, model, fit_t in [('Random Forest', rf_model, rf_time), ('Support Vector Machine', svm_model, svm_time)]:
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
        except Exception:
            auc = np.nan

        metrics[name] = {
            'Accuracy': acc, 'Precision': prec, 'Recall': rec,
            'F1-Score': f1, 'Kappa': kappa, 'ROC-AUC': auc,
            'Fit_Time_s': fit_t, 'Model': model, 'preds': preds, 'probas': probas
        }

    # In bảng so sánh
    print("\n" + "=" * 78)
    print("  BẢNG SO SÁNH HIỆU NĂNG RANDOM FOREST & SVM - THÁNG 11/2024")
    print("=" * 78)
    print(f"{'Chỉ số đánh giá':<30} | {'Random Forest':<20} | {'Support Vector Machine':<20}")
    print("-" * 78)
    print(f"{'Độ chính xác tổng thể (Acc)':<30} | {metrics['Random Forest']['Accuracy']*100:6.2f}%              | {metrics['Support Vector Machine']['Accuracy']*100:6.2f}%")
    print(f"{'Chỉ số F1-Score (Weighted)':<30} | {metrics['Random Forest']['F1-Score']:6.4f}               | {metrics['Support Vector Machine']['F1-Score']:6.4f}")
    print(f"{'Hệ số Cohen Kappa':<30} | {metrics['Random Forest']['Kappa']:6.4f}               | {metrics['Support Vector Machine']['Kappa']:6.4f}")
    print(f"{'Chỉ số ROC-AUC (Multi-class)':<30} | {metrics['Random Forest']['ROC-AUC']:6.4f}               | {metrics['Support Vector Machine']['ROC-AUC']:6.4f}")
    print(f"{'Thời gian huấn luyện':<30} | {metrics['Random Forest']['Fit_Time_s']:6.3f} giây            | {metrics['Support Vector Machine']['Fit_Time_s']:6.3f} giây")
    print("=" * 78)

    # Xuất đồ thị Feature Importance và Confusion Matrix
    plot_evaluation_th11(rf_model, svm_model, data_dict, metrics)

    return rf_model, svm_model, metrics


def plot_evaluation_th11(rf_model, svm_model, data_dict, metrics):
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

    # 1. Feature Importance Chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # RF Importance
    rf_imp = rf_model.feature_importances_ * 100
    idx_rf = np.argsort(rf_imp)
    axes[0].barh([FEATURE_NAMES[i] for i in idx_rf], rf_imp[idx_rf], color='#238b45', edgecolor='#00441b')
    axes[0].set_title("Random Forest - Feature Importance (Tháng 11/2024 - Sau bão)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Đóng góp (%)", fontsize=10)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    # SVM Permutation Importance
    perm = permutation_importance(svm_model, X_test, y_test, n_repeats=10, random_state=42)
    svm_imp = np.maximum(0, perm.importances_mean)
    if svm_imp.sum() > 0:
        svm_imp = (svm_imp / svm_imp.sum()) * 100
    idx_svm = np.argsort(svm_imp)
    axes[1].barh([FEATURE_NAMES[i] for i in idx_svm], svm_imp[idx_svm], color='#88419d', edgecolor='#4d004b')
    axes[1].set_title("SVM - Permutation Importance (Tháng 11/2024 - Sau bão)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Đóng góp (%)", fontsize=10)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR_TH11, 'thang11_feature_importance_rf_vs_svm.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"[+] [Tháng 11] Đã xuất biểu đồ Feature Importance: {chart_path}")

    # 2. Confusion Matrix Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    cm_rf = confusion_matrix(y_test, metrics['Random Forest']['preds'])
    cm_svm = confusion_matrix(y_test, metrics['Support Vector Machine']['preds'])
    labels = [f"Lớp {c}" for c in rf_model.classes_]

    for ax, cm, title, cmap in zip(axes, [cm_rf, cm_svm], ['Random Forest (Tháng 11)', 'SVM (Tháng 11)'], ['Greens', 'Purples']):
        im = ax.matshow(cm, cmap=cmap, alpha=0.85)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(j, i, str(val), ha='center', va='center', fontsize=12, fontweight='bold',
                        color='white' if val > cm.max()/2 else 'black')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Dự đoán', fontsize=9)
        ax.set_ylabel('Thực tế', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR_TH11, 'thang11_confusion_matrices.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"[+] [Tháng 11] Đã xuất biểu đồ Ma trận nhầm lẫn: {cm_path}")


# ==============================================================================
# DỰ ĐOÁN TOÀN ẢNH VÀ XUẤT GEOTIFF / WEBGIS
# ==============================================================================
def classify_and_export_th11(model, model_code="rf", model_name="Random Forest"):
    print(f"\n[*] [Tháng 11] Bắt đầu suy luận không gian toàn ảnh ({model_name})...")
    with rasterio.open(IMAGE_TH11_PATH) as src:
        height, width = src.height, src.width
        crs, bounds = src.crs, src.bounds
        nodata = src.nodata

        profile_discrete = src.profile.copy()
        profile_discrete.update(dtype=rasterio.int32, count=1, nodata=-9999, compress='lzw')

        profile_hazard = src.profile.copy()
        profile_hazard.update(dtype=rasterio.int16, count=1, nodata=-9999, compress='lzw')

        profile_proba = src.profile.copy()
        profile_proba.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='lzw')

        classes = list(model.classes_)
        if LANDSLIDE_CLASS_TH11 in classes:
            target_idx = classes.index(LANDSLIDE_CLASS_TH11)
        else:
            target_idx = len(classes) - 1

        print(f"    - Lớp chỉ điểm nguy cơ trượt lở: '{classes[target_idx]}' (index={target_idx})")

        out_discrete = os.path.join(OUTPUT_DIR_TH11, f'thang11_phan_loai_{model_code}.tif')
        out_hazard = os.path.join(OUTPUT_DIR_TH11, f'thang11_nguy_co_5cap_{model_code}.tif')
        out_proba = os.path.join(OUTPUT_DIR_TH11, f'thang11_xac_suat_{model_code}.tif')
        out_png = os.path.join(OUTPUT_DIR_TH11, f'thang11_webgis_{model_code}_rgba.png')
        out_bbox = os.path.join(OUTPUT_DIR_TH11, f'thang11_webgis_{model_code}_bbox.json')

        rgba_img = np.zeros((height, width, 4), dtype=np.uint8)

        # Đọc toàn ảnh (277x290 pixels)
        full_data = src.read()
        flat_7 = full_data.reshape(src.count, -1).T

        if nodata is not None:
            valid_mask = (flat_7 != nodata).all(axis=1) & np.isfinite(flat_7).all(axis=1)
        else:
            valid_mask = (flat_7 != 0).any(axis=1) & np.isfinite(flat_7).all(axis=1)

        flat_discrete = np.full(flat_7.shape[0], -9999, dtype=np.int32)
        flat_hazard = np.full(flat_7.shape[0], -9999, dtype=np.int16)
        flat_proba = np.full(flat_7.shape[0], -9999.0, dtype=np.float32)

        if np.any(valid_mask):
            v7 = flat_7[valid_mask]
            v12 = add_spectral_indices_to_features(v7)

            flat_discrete[valid_mask] = model.predict(v12)
            probas = model.predict_proba(v12)
            p_ls = probas[:, target_idx]
            flat_proba[valid_mask] = p_ls

            h_levels = np.digitize(p_ls, bins=[0.2, 0.4, 0.6, 0.8]) + 1
            h_levels = np.clip(h_levels, 1, 5).astype(np.int16)
            flat_hazard[valid_mask] = h_levels

            h_2d = flat_hazard.reshape((height, width))
            for lvl, col in HAZARD_COLORS.items():
                rgba_img[h_2d == lvl] = col

        # Ghi file GeoTIFF
        with rasterio.open(out_discrete, 'w', **profile_discrete) as dst:
            dst.write(flat_discrete.reshape((1, height, width)))
        with rasterio.open(out_hazard, 'w', **profile_hazard) as dst:
            dst.write(flat_hazard.reshape((1, height, width)))
        with rasterio.open(out_proba, 'w', **profile_proba) as dst:
            dst.write(flat_proba.reshape((1, height, width)))

        # Ghi file ArcMap .clr
        clr_path = os.path.splitext(out_hazard)[0] + ".clr"
        with open(clr_path, "w", encoding="utf-8") as f:
            for lvl, col in sorted(HAZARD_COLORS.items()):
                f.write(f"{lvl} {col[0]} {col[1]} {col[2]}\n")

        # Ghi ảnh PNG RGBA
        if Image is not None:
            Image.fromarray(rgba_img, mode="RGBA").save(out_png, format="PNG")
        else:
            plt.imsave(out_png, rgba_img)

        # Chuyển đổi bounds sang WGS84 cho WebGIS
        west, south, east, north = transform_bounds(crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top)
        bbox_info = {
            "month": "Tháng 11/2024",
            "model": model_name,
            "crs": "EPSG:4326",
            "png_file": os.path.basename(out_png),
            "bounds_geojson": [west, south, east, north],
            "leaflet_bounds": [[south, west], [north, east]],
            "center": [(south + north) / 2.0, (west + east) / 2.0]
        }
        with open(out_bbox, "w", encoding="utf-8") as f:
            json.dump(bbox_info, f, indent=4, ensure_ascii=False)

        print(f"[+] [Tháng 11] Xuất kết quả GeoTIFF phân loại ({model_code}): {out_discrete}")
        print(f"[+] [Tháng 11] Xuất bản đồ 5 cấp nguy cơ ({model_code}):    {out_hazard}")
        print(f"[+] [Tháng 11] Xuất WebGIS RGBA PNG:                     {out_png}")


# ==============================================================================
# TẠO GIAO DIỆN WEBGIS THÁNG 11 (LEAFLET HTML)
# ==============================================================================
def generate_webgis_html_th11():
    bbox_rf = os.path.join(OUTPUT_DIR_TH11, 'thang11_webgis_rf_bbox.json')
    if not os.path.exists(bbox_rf):
        return

    with open(bbox_rf, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    center = meta['center']
    bounds = meta['leaflet_bounds']

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebGIS Khoanh Vùng Nguy Cơ Trượt Lở Đất - Tháng 11/2024 (Sau Bão Yagi)</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        #map {{ width: 100%; height: 100%; }}
        .header-panel {{
            position: absolute; top: 12px; left: 60px; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); padding: 12px 18px;
            border-radius: 8px; box-shadow: 0 3px 14px rgba(0,0,0,0.25);
            max-width: 480px; border-left: 5px solid #238b45;
        }}
        .header-panel h2 {{ margin: 0 0 4px 0; font-size: 15px; color: #00441b; text-transform: uppercase; font-weight: bold; }}
        .header-panel p {{ margin: 0; font-size: 12px; color: #555; line-height: 1.4; }}
        .control-panel {{
            position: absolute; top: 12px; right: 12px; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); padding: 14px;
            border-radius: 8px; box-shadow: 0 3px 14px rgba(0,0,0,0.25);
            width: 250px; font-size: 12px;
        }}
        .control-panel h4 {{ margin: 0 0 10px 0; font-size: 13px; color: #222; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .layer-item {{ margin-bottom: 8px; }}
        .slider-box {{ margin-top: 10px; }}
        .slider-box label {{ display: block; margin-bottom: 3px; font-weight: bold; }}
        .legend-panel {{
            position: absolute; bottom: 25px; right: 12px; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); padding: 12px;
            border-radius: 8px; box-shadow: 0 3px 14px rgba(0,0,0,0.25);
            font-size: 11px; width: 200px;
        }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 5px; }}
        .legend-color {{ width: 22px; height: 14px; margin-right: 8px; border-radius: 3px; border: 1px solid #777; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="header-panel">
        <h2>Bản Đồ Nguy Cơ Trượt Lở Đất - Tháng 11/2024</h2>
        <p><strong>Vùng nghiên cứu:</strong> Xã Mường Lai, Lục Yên, Yên Bái | <strong>Viễn thám:</strong> Landsat 8 (30/11/2024 - Sau bão Yagi)</p>
    </div>
    <div class="control-panel">
        <h4>LỚP NGUY CƠ TRƯỢT LỞ</h4>
        <div class="layer-item">
            <input type="radio" id="layer_rf" name="hazard_layer" value="rf" checked>
            <label for="layer_rf"><strong>Random Forest (RF)</strong></label>
        </div>
        <div class="layer-item">
            <input type="radio" id="layer_svm" name="hazard_layer" value="svm">
            <label for="layer_svm"><strong>Support Vector Machine (SVM)</strong></label>
        </div>
        <div class="slider-box">
            <label>Độ trong suốt (Opacity): <span id="opacity_val">85%</span></label>
            <input type="range" id="opacity_slider" min="0" max="100" value="85" style="width: 100%;">
        </div>
    </div>
    <div class="legend-panel">
        <strong>CHÚ GIẢI 5 CẤP NGUY CƠ</strong>
        <div class="legend-item" style="margin-top: 8px;"><div class="legend-color" style="background:#38a800;"></div> Cấp 1: Rất thấp (0 - 0.2)</div>
        <div class="legend-item"><div class="legend-color" style="background:#b0e000;"></div> Cấp 2: Thấp (0.2 - 0.4)</div>
        <div class="legend-item"><div class="legend-color" style="background:#ffff00;"></div> Cấp 3: Trung bình (0.4 - 0.6)</div>
        <div class="legend-item"><div class="legend-color" style="background:#ff8000;"></div> Cấp 4: Cao (0.6 - 0.8)</div>
        <div class="legend-item"><div class="legend-color" style="background:#ff0000;"></div> Cấp 5: Rất cao (0.8 - 1.0)</div>
    </div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map('map').setView([{center[0]}, {center[1]}], 13);
        const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: 'OSM' }});
        const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{ maxZoom: 19, attribution: 'ESRI' }}).addTo(map);

        L.control.layers({{ 'Ảnh vệ tinh ESRI': esri, 'Bản đồ nền OSM': osm }}).addTo(map);

        const bounds = {bounds};
        const layerRF = L.imageOverlay('thang11_webgis_rf_rgba.png', bounds, {{ opacity: 0.85 }}).addTo(map);
        const layerSVM = L.imageOverlay('thang11_webgis_svm_rgba.png', bounds, {{ opacity: 0.85 }});

        document.getElementById('layer_rf').addEventListener('change', () => {{
            map.addLayer(layerRF);
            map.removeLayer(layerSVM);
        }});
        document.getElementById('layer_svm').addEventListener('change', () => {{
            map.addLayer(layerSVM);
            map.removeLayer(layerRF);
        }});
        document.getElementById('opacity_slider').addEventListener('input', (e) => {{
            const val = e.target.value / 100.0;
            document.getElementById('opacity_val').innerText = e.target.value + '%';
            layerRF.setOpacity(val);
            layerSVM.setOpacity(val);
        }});
        map.fitBounds(bounds);
    </script>
</body>
</html>
"""
    webgis_path = os.path.join(OUTPUT_DIR_TH11, 'webgis_thang_11.html')
    with open(webgis_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[+] [Tháng 11] Đã tạo ứng dụng WebGIS Tháng 11: {webgis_path}")


# ==============================================================================
# HÀM THỰC THI CHÍNH - THÁNG 11
# ==============================================================================
def run_pipeline_thang_11():
    print("=" * 80)
    print("  BẮT ĐẦU QUY TRÌNH XỬ LÝ DỮ LIỆU RIÊNG CHO THÁNG 11/2024 (SAU BÃO YAGI)")
    print("  KHU VỰC: XÃ MƯỜNG LAI, HUYỆN LỤC YÊN, TỈNH YÊN BÁI")
    print("=" * 80)

    # 1. Trích xuất mẫu
    X, y = extract_samples()

    # 2. Phân chia tập dữ liệu 70/15/15
    data_dict = split_dataset(X, y, random_state=42)

    # 3. Huấn luyện và đánh giá RF & SVM
    rf_model, svm_model, metrics = train_and_evaluate_models(data_dict, random_state=42)

    # 4. Lưu mô hình huấn luyện
    rf_pkl = os.path.join(OUTPUT_DIR_TH11, 'thang11_rf_model.pkl')
    svm_pkl = os.path.join(OUTPUT_DIR_TH11, 'thang11_svm_model.pkl')
    joblib.dump(rf_model, rf_pkl)
    joblib.dump(svm_model, svm_pkl)
    print(f"[+] [Tháng 11] Đã lưu mô hình RF:  {rf_pkl}")
    print(f"[+] [Tháng 11] Đã lưu mô hình SVM: {svm_pkl}")

    # 5. Dự đoán không gian toàn ảnh cho cả RF và SVM
    classify_and_export_th11(rf_model, model_code="rf", model_name="Random Forest")
    classify_and_export_th11(svm_model, model_code="svm", model_name="Support Vector Machine")

    # 6. Tự động sinh WebGIS HTML Tháng 11
    generate_webgis_html_th11()

    print("\n" + "=" * 80)
    print("  HOÀN THÀNH QUY TRÌNH THÁNG 11/2024 THÀNH CÔNG!")
    print(f"  Tất cả tệp kết quả lưu tại: {OUTPUT_DIR_TH11}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_thang_11()
