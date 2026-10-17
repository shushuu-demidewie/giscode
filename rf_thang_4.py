# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH PHÂN LOẠI VÀ KHOANH VÙNG NGUY CƠ TRƯỢT LỞ BẰNG RANDOM FOREST (RF)
RIÊNG THÁNG 4/2024 (TRƯỚC BÃO YAGI) - ĐỀ TÀI NGHIÊN CỨU ỨNG DỤNG HỌC MÁY & WEBGIS
KHU VỰC: XÃ MƯỜNG LAI, HUYỆN LỤC YÊN, TỈNH YÊN BÁI
ẢNH LANDSAT 8 (27/04/2024) - BỘ DỮ LIỆU MẪU KIỂM CHỨNG DKT4 (200 MẪU)
================================================================================

Các tính năng chuyên sâu tương đương rf_classification.py:
1. Đọc 7 kênh quang học Landsat 8 (B1 - B7) và trích xuất 5 chỉ số phổ viễn thám:
   - NDVI (Thực vật): (B5 - B4) / (B5 + B4)
   - NDBI (Xây dựng / Đất khô): (B6 - B5) / (B6 + B5)
   - MNDWI (Nước mặt tăng cường): (B3 - B6) / (B3 + B6)
   - NDWI (Độ ẩm bề mặt): (B3 - B5) / (B3 + B5)
   - BSI (Đất trống / Vết sạt trơ sỏi đá): ((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2))
   -> Tổng cộng 12 kênh đặc trưng không gian - phổ đa chiều.
2. Trích xuất mẫu linh hoạt (Point, MultiPoint, Polygon centroid), tự nhận diện cột nhãn,
   tự sinh mẫu đối chứng (nhãn 0) nếu Shapefile chỉ có điểm sạt lở.
3. Cân bằng mẫu thông minh (balance_and_split_data) khống chế trần mẫu tránh thiên lệch.
4. Huấn luyện Random Forest Classifier chuẩn:
   - n_estimators=150, max_features='sqrt', criterion='gini', bootstrap=True, oob_score=True.
5. Phân chia tập dữ liệu 70% Train - 15% Validation - 15% Test (Stratified).
6. Đánh giá toàn diện trên cả 2 tập độc lập (Validation & Test):
   - Accuracy, Precision, Recall, F1-Score, Cohen's Kappa, ROC-AUC, Classification Report, Confusion Matrix.
7. Phân tích độ quan trọng đặc trưng (Gini Feature Importance) và vẽ biểu đồ ROC Curve + Ma trận nhầm lẫn.
8. Dự đoán phân khối (Block-wise / Windowed Prediction 512x512) chống tràn RAM, xuất các tệp:
   - GeoTIFF phân loại nhãn rời rạc
   - GeoTIFF bản đồ 5 cấp độ nguy cơ trượt lở (Rất thấp -> Rất cao)
   - GeoTIFF xác suất nguy cơ trượt lở liên tục (0.0 -> 1.0)
   - File bảng màu .clr cho ArcMap/QGIS
   - Ảnh PNG RGBA nền trong suốt kèm file JSON Bounding Box (EPSG:4326) cho WebGIS.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

# Khắc phục triệt để xung đột PROJ database trên Windows
try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    if os.path.exists(proj_dir):
        os.environ['PROJ_LIB'] = proj_dir
        os.environ['PROJ_DATA'] = proj_dir
except Exception:
    pass

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi font/mã hóa UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Tự động tạo lại file .shx nếu bị thiếu khi đọc Shapefile
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
import geopandas as gpd
from shapely.geometry import Point

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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
    roc_auc_score,
    roc_curve
)
import joblib

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ TOÀN CỤC - RF THÁNG 4/2024
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'results_thang_4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Danh sách ứng viên ảnh Landsat 8 Tháng 4 (27/04/2024)
CANDIDATE_IMAGES = [
    os.path.join(BASE_DIR, 'datath4', 'Extracth4.tif'),
    os.path.join(BASE_DIR, 'data', 'Extracth4.tif'),
    os.path.join(BASE_DIR, 'datath4', 'landsat8_thang4.tif')
]

# Danh sách ứng viên Shapefile mẫu sạt lở Tháng 4
CANDIDATE_ROIS = [
    os.path.join(BASE_DIR, 'DKTth4', 'DKT4.shp'),
    os.path.join(BASE_DIR, 'data', 'DKT4.shp'),
    os.path.join(BASE_DIR, 'DKTth4', 'dkt4.shp')
]

CLASS_COLUMN_NAME = "Nhan"
LANDSLIDE_CLASS_LABEL = 22  # Lớp sạt lở / đất trống Tháng 4 (trước bão)

# Tên các lớp phân loại Tháng 4
CLASS_NAMES = {
    1: "Thảm thực vật / Cây bụi",
    22: "Đất trống / Vết sạt trượt lở đất",
    26: "Mặt nước / Thủy vực",
    29: "Rừng tự nhiên / Thảm phủ dày"
}

# Tên 12 kênh đặc trưng
FEATURE_NAMES = [
    'Kênh 1 (Coastal Aerosol)',
    'Kênh 2 (Blue)',
    'Kênh 3 (Green)',
    'Kênh 4 (Red)',
    'Kênh 5 (NIR)',
    'Kênh 6 (SWIR 1)',
    'Kênh 7 (SWIR 2)',
    'Chỉ số NDVI (Thực vật)',
    'Chỉ số NDBI (Xây dựng/Đất khô)',
    'Chỉ số MNDWI (Nước tăng cường)',
    'Chỉ số NDWI (Nước hở/Độ ẩm)',
    'Chỉ số BSI (Đất trống/Bare Soil)'
]

# Màu sắc chuẩn 5 cấp độ nguy cơ trượt lở
HAZARD_COLORS = {
    1: [56, 168, 0, 220],     # Cấp 1: Rất thấp (Xanh lá)
    2: [176, 224, 0, 220],    # Cấp 2: Thấp (Xanh lục nhạt)
    3: [255, 255, 0, 220],    # Cấp 3: Trung bình (Vàng)
    4: [255, 128, 0, 230],    # Cấp 4: Cao (Da cam)
    5: [255, 0, 0, 240],      # Cấp 5: Rất cao (Đỏ)
}

# Đường dẫn lưu trữ kết quả đầu ra Tháng 4
OUTPUT_RASTER_PATH = os.path.join(OUTPUT_DIR, 'thang4_phan_loai_rf.tif')
OUTPUT_HAZARD_RASTER_PATH = os.path.join(OUTPUT_DIR, 'thang4_nguy_co_5cap_rf.tif')
OUTPUT_PROBA_RASTER_PATH = os.path.join(OUTPUT_DIR, 'thang4_xac_suat_rf.tif')
OUTPUT_WEBGIS_PNG_PATH = os.path.join(OUTPUT_DIR, 'thang4_webgis_rf_rgba.png')
OUTPUT_WEBGIS_BBOX_JSON = os.path.join(OUTPUT_DIR, 'thang4_webgis_rf_bbox.json')
OUTPUT_IMPORTANCE_PNG = os.path.join(OUTPUT_DIR, 'rf_thang4_feature_importance.png')
OUTPUT_CONFUSION_PNG = os.path.join(OUTPUT_DIR, 'rf_thang4_confusion_matrix.png')
OUTPUT_ROC_PNG = os.path.join(OUTPUT_DIR, 'rf_thang4_roc_confusion_matrix.png')

# File lưu mô hình RF
MODEL_SAVE_PATH = os.path.join(WORKSPACE_DIR, 'rf_model_thang_4.pkl')
MODEL_COPY_PATH = os.path.join(OUTPUT_DIR, 'thang4_rf_model.pkl')


# ==============================================================================
# BƯỚC 1: TÍNH TOÁN 5 CHỈ SỐ PHỔ VIỄN THÁM (12 KÊNH ĐẶC TRƯNG)
# ==============================================================================
def compute_spectral_indices(b1, b2, b3, b4, b5, b6, b7):
    """Tính 5 chỉ số phổ viễn thám từ 7 kênh quang học Landsat 8."""
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
# BƯỚC 2: TÌM KIẾM DỮ LIỆU VÀ TRÍCH XUẤT MẪU HUẤN LUYỆN
# ==============================================================================
def resolve_data_paths():
    """Tự động xác định đường dẫn ảnh và Shapefile Tháng 4 có sẵn."""
    image_path = next((p for p in CANDIDATE_IMAGES if os.path.exists(p)), None)
    if image_path is None:
        raise FileNotFoundError(f"Không tìm thấy file ảnh Landsat 8 Tháng 4 nào trong: {CANDIDATE_IMAGES}")

    roi_path = next((p for p in CANDIDATE_ROIS if os.path.exists(p)), None)
    return image_path, roi_path


def extract_training_data(image_path, roi_path=None, class_col=CLASS_COLUMN_NAME, random_state=42):
    """Trích xuất 12 kênh đặc trưng từ ảnh và shapefile Tháng 4."""
    print(f"[*] [RF Tháng 4] Ảnh Landsat 8 đầu vào: {image_path}")

    with rasterio.open(image_path) as src:
        crs, transform = src.crs, src.transform
        nodata = src.nodata if src.nodata is not None else 0

        if roi_path and os.path.exists(roi_path):
            print(f"[*] [RF Tháng 4] Đang đọc Shapefile mẫu huấn luyện: {roi_path}")
            roi_gdf = gpd.read_file(roi_path)

            if roi_gdf.crs != crs:
                print(f"[*] Đồng bộ hệ quy chiếu Shapefile ({roi_gdf.crs}) -> Ảnh ({crs})...")
                roi_gdf = roi_gdf.to_crs(crs)

            actual_col = None
            if class_col in roi_gdf.columns:
                actual_col = class_col
            else:
                for cand in ["Nhan", "nhan", "NHAN", "label", "Classvalue", "Classname", "Id", "id"]:
                    if cand in roi_gdf.columns:
                        actual_col = cand
                        break

            labels_list = []
            points_list = []

            for idx, row in roi_gdf.iterrows():
                geom = row.geometry
                lbl = row[actual_col] if actual_col else 1
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type in ['Point', 'MultiPoint']:
                    if geom.geom_type == 'MultiPoint':
                        for pt in geom.geoms:
                            points_list.append((pt.x, pt.y))
                            labels_list.append(lbl)
                    else:
                        points_list.append((geom.x, geom.y))
                        labels_list.append(lbl)
                elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                    points_list.append((geom.centroid.x, geom.centroid.y))
                    labels_list.append(lbl)

            sampled_values = np.array(list(src.sample(points_list)), dtype=np.float32)
            valid_mask = (sampled_values != nodata).all(axis=1) & np.isfinite(sampled_values).all(axis=1)

            X_7 = sampled_values[valid_mask]
            y = np.array(labels_list)[valid_mask]

            unique_classes = np.unique(y)
            if len(unique_classes) == 1:
                print(f"[*] Shapefile chỉ chứa 1 lớp nhãn {unique_classes}. Tự động sinh mẫu Không sạt lở (0)...")
                b4 = src.read(4).astype(np.float32)
                b5 = src.read(5).astype(np.float32)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi_map = np.where((b5 + b4) != 0, (b5 - b4) / (b5 + b4), 0.0)

                high_veg_mask = (ndvi_map > 0.4) & (b4 != nodata) & np.isfinite(ndvi_map)
                candidate_indices = np.argwhere(high_veg_mask)

                np.random.seed(random_state)
                np.random.shuffle(candidate_indices)
                target_count = len(X_7)
                selected_non_ls = candidate_indices[:target_count]

                non_ls_coords = [rasterio.transform.xy(transform, r, c) for r, c in selected_non_ls]
                non_ls_samples = np.array(list(src.sample(non_ls_coords)), dtype=np.float32)
                valid_non_ls = (non_ls_samples != nodata).all(axis=1) & np.isfinite(non_ls_samples).all(axis=1)

                X_7 = np.vstack([X_7, non_ls_samples[valid_non_ls]])
                y = np.concatenate([y, np.zeros(np.sum(valid_non_ls), dtype=int)])
                print(f"[+] Đã bổ sung thành công {np.sum(valid_non_ls)} điểm mẫu đối chứng (Nhãn 0).")

        else:
            print("[!] Không tìm thấy Shapefile bên ngoài. Đang tự động trích xuất mẫu thích ứng...")
            data_all = src.read().astype(np.float32)
            b2, b3, b4, b5, b6 = data_all[1], data_all[2], data_all[3], data_all[4], data_all[5]

            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = np.where((b5 + b4) != 0, (b5 - b4) / (b5 + b4), 0.0)
                bsi = np.where(((b6 + b4) + (b5 + b2)) != 0, ((b6 + b4) - (b5 + b2)) / ((b6 + b4) + (b5 + b2)), 0.0)

            valid_raster = (data_all != nodata).all(axis=0) & np.isfinite(data_all).all(axis=0)
            ls_candidates = np.argwhere(valid_raster & (bsi > 0.15) & (ndvi < 0.25))
            non_ls_candidates = np.argwhere(valid_raster & (ndvi > 0.45))

            np.random.seed(random_state)
            np.random.shuffle(ls_candidates)
            np.random.shuffle(non_ls_candidates)

            n_samples = min(200, len(ls_candidates), len(non_ls_candidates))
            idx_1 = ls_candidates[:n_samples]
            idx_0 = non_ls_candidates[:n_samples]

            pts_1 = [rasterio.transform.xy(transform, r, c) for r, c in idx_1]
            pts_0 = [rasterio.transform.xy(transform, r, c) for r, c in idx_0]

            s_1 = np.array(list(src.sample(pts_1)), dtype=np.float32)
            s_0 = np.array(list(src.sample(pts_0)), dtype=np.float32)

            X_7 = np.vstack([s_1, s_0])
            y = np.concatenate([np.ones(len(s_1), dtype=int), np.zeros(len(s_0), dtype=int)])
            print(f"[+] Đã tự động khởi tạo {len(y)} mẫu cân bằng.")

    print(f"[+] [RF Tháng 4] Trích xuất thành công {X_7.shape[0]} điểm mẫu với {X_7.shape[1]} kênh gốc Landsat 8.")
    print("[*] Đang tính toán 5 chỉ số phổ viễn thám: NDVI, NDBI, MNDWI, NDWI, BSI...")
    X_12 = add_spectral_indices_to_features(X_7)
    print(f"[+] Tổng số kênh đặc trưng đầu vào sau khi bổ sung chỉ số phổ: {X_12.shape[1]} kênh.")

    return X_12, y


# ==============================================================================
# BƯỚC 3: CÂN BẰNG MẪU VÀ PHÂN CHIA TẬP DỮ LIỆU (70/15/15)
# ==============================================================================
def balance_and_split_data(X, y, max_samples_per_class=1500, random_state=42):
    """Cân bằng mẫu và chia 70% Train - 15% Validation - 15% Test (Stratified)."""
    print(f"\n[*] [RF Tháng 4] Cân bằng mẫu (Khống chế tối đa {max_samples_per_class} mẫu/lớp)...")
    df = pd.DataFrame(X)
    df['__label__'] = y

    counts = df['__label__'].value_counts()
    print("    - Số lượng mẫu ban đầu theo từng lớp:")
    for cls_val, cnt in counts.items():
        c_label = CLASS_NAMES.get(cls_val, f"Lớp {cls_val}")
        print(f"      + {c_label} (Giá trị {cls_val}): {cnt} mẫu")

    min_samples = counts.min()
    target_samples = min(min_samples, max_samples_per_class)

    if target_samples >= 30:
        df_balanced = df.groupby('__label__', group_keys=False).sample(
            n=target_samples, random_state=random_state
        ).reset_index(drop=True)
    else:
        df_balanced = df.groupby('__label__', group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max_samples_per_class), random_state=random_state)
        ).reset_index(drop=True)

    X_bal = df_balanced.drop('__label__', axis=1).values
    y_bal = df_balanced['__label__'].values

    print("    - Số lượng mẫu sau khi cân bằng:")
    for cls_val, cnt in pd.Series(y_bal).value_counts().items():
        c_label = CLASS_NAMES.get(cls_val, f"Lớp {cls_val}")
        print(f"      + {c_label} (Giá trị {cls_val}): {cnt} mẫu")

    print("\n[*] Chia tập dữ liệu (70% Train - 15% Val - 15% Test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_bal, y_bal, test_size=0.30, random_state=random_state, stratify=y_bal
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    print(f"    - Tập Huấn luyện (Train):   {len(X_train)} mẫu ({len(X_train)/len(X_bal)*100:.1f}%)")
    print(f"    - Tập Đánh giá (Val):       {len(X_val)} mẫu ({len(X_val)/len(X_bal)*100:.1f}%)")
    print(f"    - Tập Kiểm thử độc lập:     {len(X_test)} mẫu ({len(X_test)/len(X_bal)*100:.1f}%)")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


# ==============================================================================
# BƯỚC 4: HUẤN LUYỆN VÀ ĐÁNH GIÁ RANDOM FOREST
# ==============================================================================
def train_and_evaluate_rf(data_dict, n_estimators=150, random_state=42):
    """
    Huấn luyện Random Forest Classifier chuyên sâu cho Tháng 4/2024:
    1. n_estimators=150, max_features='sqrt', bootstrap=True, oob_score=True.
    2. Đánh giá toàn diện trên cả 2 tập: Validation VÀ Test.
    3. Tính toán Gini Feature Importance và vẽ đồ thị ROC Curve + Confusion Matrix.
    """
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

    print(f"\n[*] [RF Tháng 4] Huấn luyện Random Forest ({n_estimators} cây, max_features='sqrt', oob_score=True)...")
    t0 = time.time()
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state
    )

    rf_clf.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"[+] Huấn luyện xong trong {fit_time:.3f} giây.")

    if hasattr(rf_clf, "oob_score_"):
        print(f"[+] Out-Of-Bag (OOB) Score: {rf_clf.oob_score_ * 100:.2f}%")

    eval_results = {}
    for set_name, X_set, y_set in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        preds = rf_clf.predict(X_set)
        acc = accuracy_score(y_set, preds)
        prec = precision_score(y_set, preds, average='weighted', zero_division=0)
        rec = recall_score(y_set, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_set, preds, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_set, preds)

        auc_val = None
        try:
            probas = rf_clf.predict_proba(X_set)
            classes = rf_clf.classes_
            if len(classes) == 2:
                auc_val = roc_auc_score(y_set, probas[:, 1])
            else:
                auc_val = roc_auc_score(y_set, probas, multi_class='ovr', average='weighted')
        except Exception:
            pass

        eval_results[set_name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'kappa': kappa, 'auc': auc_val
        }

        print("\n" + "=" * 65)
        print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RF THÁNG 4 TRÊN TẬP {set_name.upper()}")
        print("=" * 65)
        print(f"  - Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%")
        print(f"  - Độ chính xác Precision:           {prec * 100:.2f}%")
        print(f"  - Độ nhạy Recall:                   {rec * 100:.2f}%")
        print(f"  - F1-Score:                         {f1:.4f}")
        print(f"  - Hệ số Cohen's Kappa:              {kappa:.4f}")
        print(f"  - Chỉ số ROC-AUC:                   {auc_val:.4f}" if auc_val is not None else "  - ROC-AUC: N/A")
        print(f"\nBáo cáo chi tiết (Classification Report) tập {set_name}:")
        print(classification_report(y_set, preds, zero_division=0))
        print(f"Ma trận nhầm lẫn (Confusion Matrix) tập {set_name}:")
        print(confusion_matrix(y_set, preds))

    # In và vẽ Feature Importance
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n" + "=" * 65)
    print("BẢNG ĐỘ QUAN TRỌNG ĐẶC TRƯNG CỦA RF THÁNG 4 (FEATURE IMPORTANCE)")
    print("=" * 65)
    for rank, idx in enumerate(indices, 1):
        cname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"Kênh {idx+1}"
        bar = "█" * int(round(importances[idx] * 40))
        print(f"  {rank:2d}. {cname:<34} : {importances[idx]*100:6.2f}%  {bar}")
    print("=" * 65)

    plot_rf_charts(rf_clf, data_dict, importances, indices)

    return rf_clf, eval_results


def plot_rf_charts(rf_clf, data_dict, importances, indices):
    """Vẽ và xuất biểu đồ Feature Importance, Confusion Matrix và ROC Curve."""
    if plt is None:
        return

    try:
        # 1. Feature Importance
        plt.figure(figsize=(9, 5.5), dpi=300)
        sorted_names = [FEATURE_NAMES[i] for i in indices][::-1]
        sorted_vals = [importances[i] * 100 for i in indices][::-1]
        bars = plt.barh(sorted_names, sorted_vals, color='#2c7bb6', edgecolor='#1a4c73', height=0.6)
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                     ha='left', va='center', fontsize=9, fontweight='bold', color='#111111')
        plt.title("Độ quan trọng đặc trưng Landsat 8 - RF Tháng 4/2024", fontsize=11, fontweight='bold')
        plt.xlabel("Tỷ lệ đóng góp (%)", fontsize=10)
        plt.xlim(0, max(sorted_vals) + 5)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUTPUT_IMPORTANCE_PNG)
        plt.close()
        print(f"[+] Đã xuất biểu đồ Feature Importance: {OUTPUT_IMPORTANCE_PNG}")

        # 2. Confusion Matrix đơn
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        preds = rf_clf.predict(X_test)
        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        labels = [f"Lớp {c}" for c in rf_clf.classes_]
        im = ax.matshow(cm, cmap='Blues', alpha=0.85)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(j, i, str(val), ha='center', va='center', fontsize=12, fontweight='bold',
                        color='white' if val > cm.max()/2 else 'black')
        ax.set_title('Ma trận nhầm lẫn RF (Tháng 4/2024)', fontsize=11, fontweight='bold', pad=15)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Dự đoán', fontsize=9)
        ax.set_ylabel('Thực tế', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(OUTPUT_CONFUSION_PNG)
        plt.close()
        print(f"[+] Đã xuất biểu đồ Ma trận nhầm lẫn: {OUTPUT_CONFUSION_PNG}")

        # 3. Kép: ROC Curve + Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

        if len(rf_clf.classes_) == 2:
            probas = rf_clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probas)
            auc_val = roc_auc_score(y_test, probas)
            axes[0].plot(fpr, tpr, color='#d7191c', lw=2.5, label=f'Random Forest (AUC = {auc_val:.4f})')
            axes[0].plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label='Ngẫu nhiên (0.50)')
            axes[0].set_xlim([0.0, 1.0])
            axes[0].set_ylim([0.0, 1.05])
            axes[0].set_xlabel('False Positive Rate', fontsize=9)
            axes[0].set_ylabel('True Positive Rate', fontsize=9)
            axes[0].set_title('Đường cong ROC (RF Tháng 4)', fontsize=10, fontweight='bold')
            axes[0].legend(loc="lower right", fontsize=9)
            axes[0].grid(True, linestyle=':', alpha=0.6)
        else:
            axes[0].text(0.5, 0.5, "Phân loại đa lớp\n(Multi-class ROC)", ha='center', va='center', fontsize=10)
            axes[0].set_title('Đường cong ROC RF Tháng 4', fontsize=10, fontweight='bold')

        cax = axes[1].matshow(cm, cmap='Blues', alpha=0.85)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                axes[1].text(j, i, str(val), ha='center', va='center', fontsize=12, fontweight='bold',
                             color='white' if val > cm.max()/2 else 'black')
        axes[1].set_title('Ma trận nhầm lẫn RF (Test Set)', fontsize=10, fontweight='bold', pad=15)
        axes[1].set_xlabel('Dự đoán (Predicted)', fontsize=9, labelpad=10)
        axes[1].set_ylabel('Thực tế (Actual)', fontsize=9)
        fig.colorbar(cax, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(OUTPUT_ROC_PNG)
        plt.close()
        print(f"[+] Đã xuất biểu đồ tổng hợp ROC & Confusion Matrix: {OUTPUT_ROC_PNG}")
    except Exception as e:
        print(f"[!] Lỗi khi xuất đồ thị đánh giá RF: {e}")


# ==============================================================================
# BƯỚC 5: PHÂN LOẠI THEO KHỐI (BLOCK-WISE 512x512) VÀ XUẤT TÀI NGUYÊN WEBGIS
# ==============================================================================
def classify_and_export_raster(
    image_path,
    model,
    output_discrete_path=OUTPUT_RASTER_PATH,
    output_hazard_path=OUTPUT_HAZARD_RASTER_PATH,
    output_proba_path=OUTPUT_PROBA_RASTER_PATH,
    output_png_path=OUTPUT_WEBGIS_PNG_PATH,
    output_bbox_json=OUTPUT_WEBGIS_BBOX_JSON,
    landslide_class=LANDSLIDE_CLASS_LABEL,
    block_size=512
):
    """
    Phân loại toàn bộ ảnh GeoTIFF bằng Random Forest theo từng khối (block_size=512):
    - Chống tràn RAM (OOM) triệt để.
    - Dự đoán xác suất sạt lở và phân cấp 5 mức độ nguy cơ:
      1: Rất thấp (0.0 - 0.2)
      2: Thấp (0.2 - 0.4)
      3: Trung bình (0.4 - 0.6)
      4: Cao (0.6 - 0.8)
      5: Rất cao (0.8 - 1.0)
    - Xuất GeoTIFF kết quả, file bảng màu ArcMap (.clr), ảnh PNG RGBA và file JSON Bounding Box.
    """
    print(f"\n[*] [RF Tháng 4] Bắt đầu suy luận không gian toàn vùng: {image_path}")
    print(f"[*] Kích thước khối xử lý: {block_size}x{block_size} pixels")

    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        crs, bounds = src.crs, src.bounds
        nodata_val = src.nodata

        profile_discrete = src.profile.copy()
        profile_discrete.update(dtype=rasterio.int32, count=1, nodata=-9999, compress='lzw')

        profile_hazard = src.profile.copy()
        profile_hazard.update(dtype=rasterio.int16, count=1, nodata=-9999, compress='lzw')

        profile_proba = src.profile.copy()
        profile_proba.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress='lzw')

        classes = list(model.classes_)
        if landslide_class is not None and landslide_class in classes:
            landslide_idx = classes.index(landslide_class)
        elif 22 in classes:
            landslide_idx = classes.index(22)  # Nhãn đất trống / sạt trượt đất Tháng 4
        elif 1 in classes and len(classes) == 2:
            landslide_idx = classes.index(1)
        else:
            landslide_idx = len(classes) - 1

        print(f"[*] Lớp đại diện sạt lở: '{classes[landslide_idx]}' (index={landslide_idx})")

        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

        total_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)
        block_count = 0

        with rasterio.open(output_discrete_path, 'w', **profile_discrete) as dst_discrete, \
             rasterio.open(output_hazard_path, 'w', **profile_hazard) as dst_hazard, \
             rasterio.open(output_proba_path, 'w', **profile_proba) as dst_proba:

            for row in range(0, height, block_size):
                h_chunk = min(block_size, height - row)
                for col in range(0, width, block_size):
                    w_chunk = min(block_size, width - col)
                    window = rasterio.windows.Window(col, row, w_chunk, h_chunk)
                    block_count += 1

                    block_data = src.read(window=window)
                    reshaped_7 = block_data.reshape(src.count, -1).T

                    if nodata_val is not None:
                        valid_mask = (reshaped_7 != nodata_val).all(axis=1) & np.isfinite(reshaped_7).all(axis=1)
                    else:
                        valid_mask = (reshaped_7 != 0).any(axis=1) & np.isfinite(reshaped_7).all(axis=1)

                    block_discrete = np.full(reshaped_7.shape[0], -9999, dtype=np.int32)
                    block_hazard = np.full(reshaped_7.shape[0], -9999, dtype=np.int16)
                    block_proba = np.full(reshaped_7.shape[0], -9999.0, dtype=np.float32)

                    if np.any(valid_mask):
                        valid_7 = reshaped_7[valid_mask]
                        valid_12 = add_spectral_indices_to_features(valid_7)

                        # Dự đoán nhãn rời rạc
                        block_discrete[valid_mask] = model.predict(valid_12)

                        # Dự đoán xác suất
                        probas = model.predict_proba(valid_12)
                        p_landslide = probas[:, landslide_idx]
                        block_proba[valid_mask] = p_landslide

                        # Phân 5 cấp nguy cơ
                        hazard_levels = np.digitize(p_landslide, bins=[0.2, 0.4, 0.6, 0.8]) + 1
                        hazard_levels = np.clip(hazard_levels, 1, 5).astype(np.int16)
                        block_hazard[valid_mask] = hazard_levels

                        # Gán màu RGBA
                        block_hazard_2d = block_hazard.reshape((h_chunk, w_chunk))
                        for level, color in HAZARD_COLORS.items():
                            rgba_image[row:row+h_chunk, col:col+w_chunk][block_hazard_2d == level] = color

                    dst_discrete.write(block_discrete.reshape((1, h_chunk, w_chunk)), window=window)
                    dst_hazard.write(block_hazard.reshape((1, h_chunk, w_chunk)), window=window)
                    dst_proba.write(block_proba.reshape((1, h_chunk, w_chunk)), window=window)

                    if block_count % 10 == 0 or block_count == total_blocks:
                        print(f"    - Tiến độ phân loại RF Tháng 4: {block_count}/{total_blocks} khối ({block_count/total_blocks*100:.1f}%)")

        print(f"[+] Đã xuất GeoTIFF phân loại nhãn RF Tháng 4:           {output_discrete_path}")
        print(f"[+] Đã xuất GeoTIFF bản đồ 5 cấp nguy cơ RF Tháng 4:      {output_hazard_path}")
        print(f"[+] Đã xuất GeoTIFF xác suất nguy cơ trượt lở RF Tháng 4: {output_proba_path}")

        # Xuất file màu .clr
        export_hazard_clr(output_hazard_path, HAZARD_COLORS)

        # Xuất WebGIS Assets (PNG RGBA + JSON BBOX)
        export_webgis_assets(
            rgba_image=rgba_image,
            src_crs=crs,
            src_bounds=bounds,
            png_path=output_png_path,
            json_path=output_bbox_json,
            model_name="Random Forest (Tháng 4/2024)"
        )


def export_hazard_clr(raster_path, hazard_colors):
    """Xuất file bảng màu .clr để ArcMap/QGIS tự nhận màu phân loại 5 cấp."""
    clr_path = os.path.splitext(raster_path)[0] + ".clr"
    try:
        with open(clr_path, "w", encoding="utf-8") as f:
            for level, rgba in sorted(hazard_colors.items()):
                f.write(f"{level} {rgba[0]} {rgba[1]} {rgba[2]}\n")
        print(f"[+] Đã tạo file bảng màu ArcMap (.clr): {clr_path}")
    except Exception as e:
        print(f"[!] Lỗi ghi file .clr: {e}")


def export_webgis_assets(rgba_image, src_crs, src_bounds, png_path, json_path, model_name="Random Forest"):
    """Xuất ảnh PNG trong suốt và file JSON Bounding Box EPSG:4326 phục vụ WebGIS."""
    print(f"\n[*] Đang xuất ảnh RGBA nền trong suốt cho WebGIS ({model_name}): {png_path}")
    if Image is not None:
        img = Image.fromarray(rgba_image, mode="RGBA")
        img.save(png_path, format="PNG")
    elif plt is not None:
        plt.imsave(png_path, rgba_image)
    else:
        with rasterio.open(
            png_path, 'w', driver='PNG',
            width=rgba_image.shape[1], height=rgba_image.shape[0],
            count=4, dtype=rasterio.uint8
        ) as dst_png:
            for i in range(4):
                dst_png.write(rgba_image[:, :, i], i + 1)
    print(f"[+] Đã xuất file PNG trong suốt: {png_path} ({rgba_image.shape[1]}x{rgba_image.shape[0]} px)")

    try:
        min_x, min_y, max_x, max_y = src_bounds
        west, south, east, north = transform_bounds(src_crs, 'EPSG:4326', min_x, min_y, max_x, max_y)
    except Exception as e:
        print(f"[!] Chuyển đổi CRS sang EPSG:4326 gặp lỗi: {e}. Dùng bounds gốc.")
        west, south, east, north = src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top

    bbox_info = {
        "month": "Tháng 4/2024",
        "model": model_name,
        "crs": "EPSG:4326",
        "description": f"Tọa độ Bounding Box bản đồ nguy cơ trượt lở {model_name} trên WebGIS",
        "png_file": os.path.basename(png_path),
        "bounds_geojson": [west, south, east, north],
        "leaflet_bounds": [[south, west], [north, east]],
        "center": [(south + north) / 2.0, (west + east) / 2.0],
        "west_lon": west, "south_lat": south,
        "east_lon": east, "north_lat": north
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bbox_info, f, indent=4, ensure_ascii=False)
    print(f"[+] Đã lưu thông số tọa độ Bounding Box tại: {json_path}")


# ==============================================================================
# HÀM THỰC THI CHÍNH
# ==============================================================================
def main():
    print("=" * 75)
    print("  ỨNG DỤNG RANDOM FOREST & LANDSAT 8 KHOANH VÙNG NGUY CƠ TRƯỢT LỞ")
    print("  CHUYÊN BIỆT THÁNG 4/2024 (TRƯỚC BÃO YAGI) - XÃ MƯỜNG LAI, LỤC YÊN, YÊN BÁI")
    print("=" * 75)

    # 1. Xác định dữ liệu ảnh và Shapefile Tháng 4
    image_path, roi_path = resolve_data_paths()

    # 2. Trích xuất mẫu và tính toán 12 kênh đặc trưng
    X, y = extract_training_data(
        image_path=image_path,
        roi_path=roi_path,
        class_col=CLASS_COLUMN_NAME
    )

    # 3. Cân bằng mẫu và chia tập dữ liệu 70/15/15
    data_dict = balance_and_split_data(
        X=X, y=y,
        max_samples_per_class=1500,
        random_state=42
    )

    # 4. Huấn luyện Random Forest và đánh giá mô hình
    rf_clf, eval_results = train_and_evaluate_rf(
        data_dict=data_dict,
        n_estimators=150,
        random_state=42
    )

    # 5. Lưu mô hình RF Tháng 4
    joblib.dump(rf_clf, MODEL_SAVE_PATH)
    print(f"[*] Đã lưu mô hình Random Forest Tháng 4 tại: {MODEL_SAVE_PATH}")
    try:
        joblib.dump(rf_clf, MODEL_COPY_PATH)
        print(f"[*] Đã sao lưu mô hình RF tại thư mục kết quả: {MODEL_COPY_PATH}")
    except Exception:
        pass

    # 6. Phân loại toàn ảnh theo khối (512x512) và xuất tài nguyên WebGIS
    classify_and_export_raster(
        image_path=image_path,
        model=rf_clf,
        output_discrete_path=OUTPUT_RASTER_PATH,
        output_hazard_path=OUTPUT_HAZARD_RASTER_PATH,
        output_proba_path=OUTPUT_PROBA_RASTER_PATH,
        output_png_path=OUTPUT_WEBGIS_PNG_PATH,
        output_bbox_json=OUTPUT_WEBGIS_BBOX_JSON,
        landslide_class=LANDSLIDE_CLASS_LABEL,
        block_size=512
    )

    print("\n" + "=" * 75)
    print("[+] HOÀN THÀNH TOÀN BỘ QUY TRÌNH RANDOM FOREST THÁNG 4/2024!")
    print(f"Các file kết quả được lưu tại: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
