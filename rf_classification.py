# -*- coding: utf-8 -*-
"""
Quy trình phân loại ảnh viễn thám Landsat 8 bằng Random Forest (12 kênh đặc trưng)
phục vụ đề tài: "Ứng dụng học máy và ảnh viễn thám Landsat 8 thành lập bản đồ / WebGIS
khoanh vùng nguy cơ trượt lở".

Các tính năng chính:
1. Đọc 7 kênh quang học Landsat 8 (B1 - B7) và tự động tính thêm 5 chỉ số phổ viễn thám:
   NDVI, NDBI, MNDWI, NDWI, BSI (Bare Soil Index) -> Tổng cộng 12 kênh đầu vào.
2. Cân bằng mẫu giữa các lớp (Undersampling khống chế tối đa 1500 mẫu/lớp).
3. Chia tập dữ liệu 70% Train - 15% Val - 15% Test.
4. Đánh giá toàn diện mô hình: Overall Accuracy, Cohen's Kappa, ROC-AUC, ma trận nhầm lẫn
   và trích xuất độ quan trọng của từng kênh ảnh (feature_importances_).
5. Phân loại theo từng khối (window/block) 12 kênh để chống tràn RAM.
6. Tùy chọn xuất bản đồ phân vùng 5 cấp nguy cơ trượt lở dựa trên xác suất predict_proba().
7. Tự động xuất ảnh .png nền trong suốt (RGBA) kèm tọa độ Bounding Box (EPSG:4326)
   sẵn sàng tích hợp trực tiếp lên nền bản đồ WebGIS (Leaflet / OpenLayers / Mapbox).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
import geopandas as gpd
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
    cohen_kappa_score,
    roc_auc_score
)
import joblib

# Bỏ qua cảnh báo chia cho 0 trong raster
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi font/mã hóa
if sys.stdout and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Tự động tạo lại file .shx nếu bị thiếu khi đọc Shapefile
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Tự động khắc phục xung đột PROJ_LIB / PROJ_DATA giữa PostgreSQL và Python
try:
    import pyproj.datadir
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ['PROJ_LIB'] = proj_dir
    os.environ['PROJ_DATA'] = proj_dir
except Exception:
    pass


# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU VÀ TÙY CHỌN ĐẦU RA
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 1. Đường dẫn ảnh GeoTIFF đầu vào (7 kênh Landsat 8: B1 đến B7)
# Hệ thống tự kiểm tra các đường dẫn có sẵn trong thư mục dự án
CANDIDATE_IMAGES = [
    os.path.join(BASE_DIR, 'data', 'dp2026Extrac.tif'),
    os.path.join(BASE_DIR, 'datahanoi', 'anhcat.tif'),
    os.path.join(BASE_DIR, 'data', 'anhcat.tif'),
    r"D:\DLGIS\hanoi2026\datahanoi\anhcat01.tif"
]
IMAGE_PATH = next((p for p in CANDIDATE_IMAGES if os.path.exists(p)), CANDIDATE_IMAGES[0])

# 2. Đường dẫn Shapefile (.shp) mẫu huấn luyện (số hóa từ ArcMap/QGIS)
CANDIDATE_ROIS = [
    os.path.join(BASE_DIR, 'DKT2026', 'DKT2026.shp'),
    os.path.join(BASE_DIR, 'Dkt2015', 'DKT2015.shp'),
    os.path.join(BASE_DIR, 'Dkt', 'DKT30.shp'),
    os.path.join(BASE_DIR, 'data', 'ketquatach.shp')
]
ROI_SHAPEFILE_PATH = next((p for p in CANDIDATE_ROIS if os.path.exists(p)), CANDIDATE_ROIS[0])

# 3. Tên cột chứa mã loại / lớp nhãn phân loại trong Shapefile
CLASS_COLUMN_NAME = "Nhan"

# 4. Tùy chọn lớp đại diện cho trượt lở (nếu bài toán là phát hiện trượt lở nhị phân hoặc chọn 1 lớp)
# Đặt None để mô hình tự động chọn lớp trượt lở (hoặc lớp có nhãn lớn nhất trong bài toán nhị phân)
LANDSLIDE_CLASS_LABEL = None

# 5. Đường dẫn lưu kết quả đầu ra
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File GeoTIFF phân loại nhãn rời rạc
OUTPUT_RASTER_PATH = os.path.join(OUTPUT_DIR, 'anh_phan_loai_rf.tif')

# File GeoTIFF phân vùng 5 cấp nguy cơ trượt lở
OUTPUT_HAZARD_RASTER_PATH = os.path.join(OUTPUT_DIR, 'ban_do_5cap_nguy_co_truot_lo.tif')

# File GeoTIFF xác suất liên tục (0.0 - 1.0)
OUTPUT_PROBA_RASTER_PATH = os.path.join(OUTPUT_DIR, 'xac_suat_nguy_co_truot_lo.tif')

# File ảnh PNG trong suốt phục vụ WebGIS kèm metadata Bounding Box
OUTPUT_WEBGIS_PNG_PATH = os.path.join(OUTPUT_DIR, 'webgis_nguy_co_truot_lo_rgba.png')
OUTPUT_WEBGIS_BBOX_JSON = os.path.join(OUTPUT_DIR, 'webgis_nguy_co_truot_lo_bbox.json')

# File lưu mô hình Random Forest
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'giscode', 'rf_model.pkl')

# Tên 12 kênh đặc trưng phục vụ giải thích mô hình
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


# ==============================================================================
# HÀM TÍNH TOÁN 5 CHỈ SỐ PHỔ VIỄN THÁM TỪ LANDSAT 8 (B1 - B7)
# ==============================================================================
def compute_spectral_indices(b1, b2, b3, b4, b5, b6, b7):
    """
    Tính toán 5 chỉ số phổ viễn thám chuyên sâu từ 7 kênh quang học Landsat 8:
    1. NDVI  = (B5 - B4) / (B5 + B4) : Độ che phủ thực vật / sinh khối
    2. NDBI  = (B6 - B5) / (B6 + B5) : Đất xây dựng, vật liệu nhân tạo, đất khô
    3. MNDWI = (B3 - B6) / (B3 + B6) : Mặt nước sạch, tách biệt công trình
    4. NDWI  = (B3 - B5) / (B3 + B5) : Độ ẩm thực vật / nước mặt hở
    5. BSI   = ((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2)) : Chỉ số đất trống (Bare Soil Index)
               Đặc biệt hữu ích nhận diện vùng mất thảm phủ, sạt lở lộ tầng đất đá.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # 1. NDVI (NIR: B5, Red: B4)
        ndvi = (b5 - b4) / (b5 + b4)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. NDBI (SWIR1: B6, NIR: B5)
        ndbi = (b6 - b5) / (b6 + b5)
        ndbi = np.nan_to_num(ndbi, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. MNDWI (Green: B3, SWIR1: B6)
        mndwi = (b3 - b6) / (b3 + b6)
        mndwi = np.nan_to_num(mndwi, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. NDWI (Green: B3, NIR: B5)
        ndwi = (b3 - b5) / (b3 + b5)
        ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. BSI (Bare Soil Index - Landsat 8)
        bsi = ((b6 + b4) - (b5 + b2)) / ((b6 + b4) + (b5 + b2))
        bsi = np.nan_to_num(bsi, nan=0.0, posinf=0.0, neginf=0.0)

    return ndvi, ndbi, mndwi, ndwi, bsi


def add_spectral_indices_to_features(X_7bands):
    """
    Ghép 5 chỉ số phổ vào ma trận 7 kênh ban đầu, trả về ma trận 12 kênh đặc trưng.
    X_7bands: np.ndarray shape (N, 7)
    Trả về: np.ndarray shape (N, 12)
    """
    X_f = X_7bands.astype(np.float32)
    b1 = X_f[:, 0]
    b2 = X_f[:, 1]
    b3 = X_f[:, 2]
    b4 = X_f[:, 3]
    b5 = X_f[:, 4]
    b6 = X_f[:, 5]
    b7 = X_f[:, 6]

    ndvi, ndbi, mndwi, ndwi, bsi = compute_spectral_indices(b1, b2, b3, b4, b5, b6, b7)
    return np.column_stack([X_f, ndvi, ndbi, mndwi, ndwi, bsi])


# ==============================================================================
# BƯỚC 1: TRÍCH XUẤT MẪU VÀ TÍNH 12 KÊNH ĐẶC TRƯNG
# ==============================================================================
def extract_training_data(image_path, roi_path, class_col):
    """
    Trích xuất giá trị 7 kênh gốc từ Shapefile mẫu (Point hoặc Polygon),
    sau đó tự động tính toán 5 chỉ số phổ để tạo tập dữ liệu 12 kênh hoàn chỉnh.
    """
    print(f"[*] Đang đọc ảnh viễn thám đầu vào: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"Không tìm thấy file Shapefile: {roi_path}")

    print(f"[*] Đang đọc Shapefile mẫu huấn luyện: {roi_path}")
    roi_gdf = gpd.read_file(roi_path)

    # Tự động tìm cột nhãn phân loại nếu tên cột chưa khớp
    if class_col not in roi_gdf.columns:
        for candidate in ["Nhan", "nhan", "NHAN", "label", "Classvalue", "Classname", "Id", "id"]:
            if candidate in roi_gdf.columns and roi_gdf[candidate].nunique() > 1:
                print(f"[*] Cột '{class_col}' không tồn tại, tự động chuyển sang cột: '{candidate}'")
                class_col = candidate
                break
        else:
            raise ValueError(f"Không tìm thấy cột phân loại trong Shapefile. Các cột hiện có: {list(roi_gdf.columns)}")

    X_raw_list = []
    y_raw_list = []

    with rasterio.open(image_path) as src:
        # Đồng bộ hệ tọa độ (CRS) nếu khác nhau
        if roi_gdf.crs != src.crs:
            print(f"[*] Đồng bộ hệ quy chiếu Shapefile ({roi_gdf.crs}) -> Ảnh ({src.crs})...")
            roi_gdf = roi_gdf.to_crs(src.crs)

        geom_types = roi_gdf.geometry.geom_type.unique()
        print(f"[*] Kiểu đối tượng mẫu (Geometry): {geom_types}")

        # Trường hợp 1: Dữ liệu dạng POINT
        if any("Point" in gt for gt in geom_types):
            points = [(geom.x, geom.y) for geom in roi_gdf.geometry if geom.geom_type in ["Point", "MultiPoint"]]
            labels = [row[class_col] for idx, row in roi_gdf.iterrows() if row["geometry"].geom_type in ["Point", "MultiPoint"]]

            sampled_values = list(src.sample(points))
            sampled_values = np.array(sampled_values)

            if src.nodata is not None:
                valid_mask = (sampled_values != src.nodata).all(axis=1)
            else:
                valid_mask = (sampled_values != 0).any(axis=1)

            X_raw_list.append(sampled_values[valid_mask])
            y_raw_list.append(np.array(labels)[valid_mask])

        # Trường hợp 2: Dữ liệu dạng POLYGON
        if any("Polygon" in gt for gt in geom_types):
            for idx, row in roi_gdf.iterrows():
                if "Polygon" not in row["geometry"].geom_type:
                    continue
                geom = [row["geometry"]]
                class_label = row[class_col]

                try:
                    masked_data, _ = mask(src, geom, crop=True, nodata=src.nodata if src.nodata is not None else -9999)
                    bands, h, w = masked_data.shape
                    pixels = masked_data.reshape(bands, -1).T

                    if src.nodata is not None:
                        valid_mask = (pixels != src.nodata).all(axis=1)
                    else:
                        valid_mask = (pixels != 0).any(axis=1)

                    valid_pixels = pixels[valid_mask]
                    if len(valid_pixels) > 0:
                        X_raw_list.append(valid_pixels)
                        y_raw_list.append(np.full(len(valid_pixels), class_label))
                except Exception as e:
                    print(f"[!] Bỏ qua đa giác mẫu số {idx}: {e}")

    if not X_raw_list:
        raise ValueError("Không trích xuất được pixel mẫu nào! Vui lòng kiểm tra lại tọa độ giữa ảnh và Shapefile.")

    X_7 = np.vstack(X_raw_list)
    y = np.concatenate(y_raw_list)

    print(f"[+] Trích xuất thành công {X_7.shape[0]} điểm mẫu với {X_7.shape[1]} kênh gốc.")

    # Tính toán 5 chỉ số phổ -> nâng lên 12 kênh
    print("[*] Đang tính toán 5 chỉ số phổ viễn thám: NDVI, NDBI, MNDWI, NDWI, BSI...")
    X_12 = add_spectral_indices_to_features(X_7)
    print(f"[+] Tổng số kênh đặc trưng đầu vào sau khi bổ sung chỉ số phổ: {X_12.shape[1]} kênh.")

    return X_12, y


# ==============================================================================
# BƯỚC 2: CÂN BẰNG MẪU (UNDERSAMPLING TỐI ĐA 1500 MẪU/LỚP)
# ==============================================================================
def balance_classes_undersampling(X, y, max_samples_per_class=1500, random_state=42):
    """
    Cân bằng mẫu giữa các lớp nhằm loại trừ hiện tượng mất cân bằng dữ liệu (class imbalance):
    - Đặt cận trên tối đa 1500 mẫu cho mỗi lớp.
    - Đảm bảo mô hình học đều các lớp, tránh thiên lệch về các lớp chiếm số lượng lớn.
    """
    print(f"\n[*] Đang thực hiện cân bằng mẫu (Undersampling <= {max_samples_per_class} mẫu/lớp)...")
    df = pd.DataFrame(X)
    df['__label__'] = y

    counts = df['__label__'].value_counts()
    print("    - Số lượng mẫu ban đầu theo từng lớp:")
    for cls_val, cnt in counts.items():
        print(f"      + Lớp {cls_val}: {cnt} mẫu")

    min_samples = counts.min()
    target_samples = min(min_samples, max_samples_per_class)

    # Nếu số lượng mẫu tối thiểu của lớp nhỏ nhất >= 50, cân bằng chính xác số lượng
    if target_samples >= 50:
        df_balanced = df.groupby('__label__', group_keys=False).sample(
            n=target_samples, random_state=random_state
        ).reset_index(drop=True)
    else:
        # Nếu có lớp ít hơn 50 mẫu, chỉ cắt giảm các lớp lớn vượt quá 1500 mẫu
        df_balanced = df.groupby('__label__', group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max_samples_per_class), random_state=random_state)
        ).reset_index(drop=True)

    X_bal = df_balanced.drop('__label__', axis=1).values
    y_bal = df_balanced['__label__'].values

    print("    - Số lượng mẫu sau khi cân bằng:")
    for cls_val, cnt in pd.Series(y_bal).value_counts().items():
        print(f"      + Lớp {cls_val}: {cnt} mẫu")

    return X_bal, y_bal


# ==============================================================================
# BƯỚC 3: HUẤN LUYỆN RANDOM FOREST VÀ ĐÁNH GIÁ (KAPPA, ROC-AUC, IMPORTANCE)
# ==============================================================================
def train_random_forest(X, y, n_estimators=150, max_depth=None, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Chia 70% Train - 15% Val - 15% Test, huấn luyện Random Forest và đánh giá mô hình:
    - Overall Accuracy
    - Cohen's Kappa score
    - ROC-AUC score (hỗ trợ cả nhị phân và đa lớp OVR)
    - Feature Importances của toàn bộ 12 kênh
    """
    print(f"\n[*] Chia tập dữ liệu ({int(train_ratio*100)}% Train - {int(val_ratio*100)}% Val - {int(test_ratio*100)}% Test)...")

    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_ratio, random_state=random_state, stratify=y
    )

    relative_test_ratio = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"    - Tập Train:      {len(X_train)} mẫu ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    - Tập Validation: {len(X_val)} mẫu ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    - Tập Test:       {len(X_test)} mẫu ({len(X_test)/len(X)*100:.1f}%)")

    print(f"\n[*] Đang huấn luyện Random Forest ({n_estimators} cây quyết định, 12 kênh đặc trưng)...")
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        oob_score=True
    )
    rf_clf.fit(X_train, y_train)

    def evaluate_set(name, X_set, y_set):
        preds = rf_clf.predict(X_set)
        acc = accuracy_score(y_set, preds)
        kappa = cohen_kappa_score(y_set, preds)

        # Tính ROC-AUC
        auc_str = "N/A"
        try:
            probas = rf_clf.predict_proba(X_set)
            if len(rf_clf.classes_) == 2:
                auc_val = roc_auc_score(y_set, probas[:, 1])
                auc_str = f"{auc_val:.4f}"
            elif len(rf_clf.classes_) > 2:
                auc_val = roc_auc_score(y_set, probas, multi_class='ovr', average='weighted')
                auc_str = f"{auc_val:.4f} (OVR Weighted)"
        except Exception as e:
            auc_str = f"Không tính được ({e})"

        print("\n" + "=" * 60)
        print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP {name.upper()}")
        print("=" * 60)
        print(f"- Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%")
        print(f"- Hệ số Cohen's Kappa:             {kappa:.4f}")
        print(f"- Chỉ số ROC-AUC:                  {auc_str}")
        print(f"\nBáo cáo chi tiết (Classification Report) tập {name}:")
        print(classification_report(y_set, preds))
        print(f"Ma trận nhầm lẫn (Confusion Matrix) tập {name}:")
        print(confusion_matrix(y_set, preds))
        return acc, kappa

    # Đánh giá trên Validation và Test độc lập
    evaluate_set("Validation", X_val, y_val)
    evaluate_set("Test (Kiểm tra độc lập)", X_test, y_test)

    if hasattr(rf_clf, "oob_score_"):
        print(f"- Điểm kiểm định Out-Of-Bag (OOB Score): {rf_clf.oob_score_ * 100:.2f}%")

    # In ra độ quan trọng của từng kênh đặc trưng (Feature Importances)
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n" + "=" * 60)
    print("BẢNG ĐỘ QUAN TRỌNG CỦA 12 KÊNH ĐẶC TRƯNG (FEATURE IMPORTANCES)")
    print("=" * 60)
    for rank, idx in enumerate(indices, 1):
        col_name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"Kênh {idx+1}"
        bar = "█" * int(round(importances[idx] * 40))
        print(f"  {rank:2d}. {col_name:<34} : {importances[idx]*100:6.2f}%  {bar}")
    print("=" * 60)

    return rf_clf


# ==============================================================================
# BƯỚC 4: PHÂN LOẠI TOÀN BỘ ẢNH, KHOANH VÙNG 5 CẤP NGUY CƠ VÀ XUẤT WEBGIS
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
    block_size=1024
):
    """
    Phân loại toàn bộ ảnh theo từng khối (window/block) 12 kênh:
    1. Đọc khối 7 kênh gốc từ ảnh GeoTIFF và tính ngay 5 chỉ số phổ.
    2. Dự đoán lớp nhãn (Discrete LULC / Hazard Class).
    3. Dự đoán xác suất liên tục (predict_proba) và phân vùng 5 cấp nguy cơ trượt lở:
       - Cấp 1 (Rất thấp): 0.00 <= P < 0.20  (Màu xanh lá đậm)
       - Cấp 2 (Thấp)    : 0.20 <= P < 0.40  (Màu xanh lá nhạt)
       - Cấp 3 (Trung bình): 0.40 <= P < 0.60 (Màu vàng)
       - Cấp 4 (Cao)     : 0.60 <= P < 0.80  (Màu da cam)
       - Cấp 5 (Rất cao) : 0.80 <= P <= 1.00 (Màu đỏ)
    4. Xuất ảnh màu PNG nền trong suốt (RGBA) kèm tọa độ Bounding Box (EPSG:4326)
       cho phép đưa trực tiếp lên WebGIS (Leaflet / OpenLayers / Mapbox).
    """
    print(f"\n[*] Bắt đầu xử lý và phân loại toàn bộ ảnh viễn thám: {image_path}")

    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width
        crs = src.crs
        bounds = src.bounds
        nodata_val = src.nodata

        # Cấu hình lưu trữ file GeoTIFF kết quả 5 cấp nguy cơ
        hazard_profile = src.profile.copy()
        hazard_profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=-9999,
            compress='lzw'
        )

        # Cấu hình lưu trữ file GeoTIFF xác suất liên tục
        proba_profile = src.profile.copy()
        proba_profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=-9999.0,
            compress='lzw'
        )

        # Xác định vị trí cột xác suất đại diện cho nguy cơ trượt lở
        classes = list(model.classes_)
        if landslide_class is not None and landslide_class in classes:
            landslide_idx = classes.index(landslide_class)
        else:
            # Nếu không chỉ định: Trong bài toán nhị phân, chọn lớp thứ 2 (dương);
            # Trong bài toán đa lớp, chọn lớp có giá trị nhãn lớn nhất (hoặc lớp cuối cùng)
            landslide_idx = len(classes) - 1

        print(f"[*] Sử dụng xác suất của lớp '{classes[landslide_idx]}' làm thước đo nguy cơ trượt lở.")

        # Mảng RGBA kích thước đầy đủ để xuất file WebGIS PNG trong suốt
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

        # Bảng màu 5 cấp độ nguy cơ trượt lở [R, G, B, Alpha]:
        # Cấp 1: Rất thấp (Xanh lá)
        # Cấp 2: Thấp (Xanh lục nhạt)
        # Cấp 3: Trung bình (Vàng)
        # Cấp 4: Cao (Da cam)
        # Cấp 5: Rất cao (Đỏ)
        hazard_colors = {
            1: [56, 168, 0, 220],     # Cấp 1: Rất thấp
            2: [176, 224, 0, 220],    # Cấp 2: Thấp
            3: [255, 255, 0, 220],    # Cấp 3: Trung bình
            4: [255, 128, 0, 230],    # Cấp 4: Cao
            5: [255, 0, 0, 240],      # Cấp 5: Rất cao
        }

        with rasterio.open(output_hazard_path, 'w', **hazard_profile) as dst_hazard, \
             rasterio.open(output_proba_path, 'w', **proba_profile) as dst_proba:

            # Duyệt từng khối block để tiết kiệm bộ nhớ RAM
            for row in range(0, height, block_size):
                h_chunk = min(block_size, height - row)

                for col in range(0, width, block_size):
                    w_chunk = min(block_size, width - col)
                    window = rasterio.windows.Window(col, row, w_chunk, h_chunk)

                    # Đọc dữ liệu 7 kênh của block: shape (7, h_chunk, w_chunk)
                    block_data = src.read(window=window)

                    # Chuyển thành dạng ma trận pixel: shape (N_pixels, 7)
                    reshaped_7 = block_data.reshape(src.count, -1).T

                    # Lọc pixel hợp lệ (loại bỏ NoData)
                    if nodata_val is not None:
                        valid_mask = (reshaped_7 != nodata_val).all(axis=1)
                    else:
                        valid_mask = (reshaped_7 != 0).any(axis=1)

                    # Mảng kết quả cho block
                    block_hazard = np.full(reshaped_7.shape[0], -9999, dtype=np.int16)
                    block_proba = np.full(reshaped_7.shape[0], -9999.0, dtype=np.float32)

                    if np.any(valid_mask):
                        valid_pixels_7 = reshaped_7[valid_mask]

                        # TÍNH TOÁN 12 KÊNH ĐẶC TRƯNG CHO TỪNG BLOCK
                        valid_pixels_12 = add_spectral_indices_to_features(valid_pixels_7)

                        # Dự đoán xác suất trượt lở
                        probas = model.predict_proba(valid_pixels_12)
                        p_landslide = probas[:, landslide_idx]
                        block_proba[valid_mask] = p_landslide

                        # Phân vùng 5 cấp độ nguy cơ trượt lở dựa trên xác suất:
                        # 0.0 - 0.2: Cấp 1 (Rất thấp)
                        # 0.2 - 0.4: Cấp 2 (Thấp)
                        # 0.4 - 0.6: Cấp 3 (Trung bình)
                        # 0.6 - 0.8: Cấp 4 (Cao)
                        # 0.8 - 1.0: Cấp 5 (Rất cao)
                        hazard_levels = np.digitize(p_landslide, bins=[0.2, 0.4, 0.6, 0.8]) + 1
                        hazard_levels = np.clip(hazard_levels, 1, 5).astype(np.int16)
                        block_hazard[valid_mask] = hazard_levels

                        # Gán màu RGBA cho ảnh WebGIS
                        block_hazard_2d = block_hazard.reshape((h_chunk, w_chunk))
                        for level, color in hazard_colors.items():
                            match_mask = (block_hazard_2d == level)
                            rgba_image[row:row+h_chunk, col:col+w_chunk][match_mask] = color

                    # Ghi khối dữ liệu vào GeoTIFF
                    dst_hazard.write(block_hazard.reshape((1, h_chunk, w_chunk)), window=window)
                    dst_proba.write(block_proba.reshape((1, h_chunk, w_chunk)), window=window)

        print(f"[+] Đã xuất GeoTIFF bản đồ 5 cấp nguy cơ trượt lở: {output_hazard_path}")
        print(f"[+] Đã xuất GeoTIFF xác suất nguy cơ trượt lở:       {output_proba_path}")

        # Tạo file bảng màu .clr tương thích mở trên ArcMap / QGIS
        export_hazard_clr(output_hazard_path, hazard_colors)

        # Xuất ảnh PNG trong suốt và Bounding Box (EPSG:4326) phục vụ WebGIS
        export_webgis_assets(
            rgba_image=rgba_image,
            src_crs=crs,
            src_bounds=bounds,
            png_path=output_png_path,
            json_path=output_bbox_json
        )


def export_hazard_clr(raster_path, hazard_colors):
    """
    Tạo file .clr đi kèm file GeoTIFF để ArcMap/QGIS tự động nạp bảng màu phân loại 5 cấp.
    """
    clr_path = os.path.splitext(raster_path)[0] + ".clr"
    try:
        with open(clr_path, "w", encoding="utf-8") as f:
            for level, rgba in sorted(hazard_colors.items()):
                f.write(f"{level} {rgba[0]} {rgba[1]} {rgba[2]}\n")
        print(f"[+] Đã tạo file bảng màu ArcMap (.clr): {clr_path}")
    except Exception as e:
        print(f"[!] Không thể ghi file .clr: {e}")


def export_webgis_assets(rgba_image, src_crs, src_bounds, png_path, json_path):
    """
    1. Xuất ảnh màu PNG nền trong suốt (RGBA) chất lượng cao.
    2. Chuyển đổi tọa độ Bounding Box từ CRS gốc sang hệ tọa độ quốc tế WGS 84 (EPSG:4326).
    3. Lưu file JSON cấu hình và in mã nguồn JavaScript để nhúng ngay vào Leaflet / OpenLayers.
    """
    print(f"\n[*] Đang xuất ảnh RGBA nền trong suốt cho WebGIS: {png_path}")
    if Image is not None:
        img = Image.fromarray(rgba_image, mode="RGBA")
        img.save(png_path, format="PNG")
    elif plt is not None:
        plt.imsave(png_path, rgba_image)
    else:
        with rasterio.open(
            png_path, 'w',
            driver='PNG',
            width=rgba_image.shape[1],
            height=rgba_image.shape[0],
            count=4,
            dtype=rasterio.uint8
        ) as dst_png:
            for i in range(4):
                dst_png.write(rgba_image[:, :, i], i + 1)
    print(f"[+] Đã xuất file PNG trong suốt: {png_path} ({rgba_image.shape[1]}x{rgba_image.shape[0]} px)")

    # Chuyển đổi Bounding Box sang EPSG:4326 (WGS 84 kinh độ - vĩ độ)
    try:
        min_x, min_y, max_x, max_y = src_bounds
        west, south, east, north = transform_bounds(src_crs, 'EPSG:4326', min_x, min_y, max_x, max_y)
    except Exception as e:
        print(f"[!] Chuyển đổi CRS sang EPSG:4326 gặp lỗi: {e}. Sử dụng bounds gốc.")
        west, south, east, north = src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top

    bbox_info = {
        "crs": "EPSG:4326",
        "description": "Tọa độ Bounding Box hiển thị ảnh bản đồ nguy cơ trượt lở trên WebGIS",
        "png_file": os.path.basename(png_path),
        "bounds_geojson": [west, south, east, north],
        "leaflet_bounds": [[south, west], [north, east]],
        "center": [(south + north) / 2.0, (west + east) / 2.0],
        "west_lon": west,
        "south_lat": south,
        "east_lon": east,
        "north_lat": north
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bbox_info, f, indent=4, ensure_ascii=False)
    print(f"[+] Đã lưu thông số tọa độ Bounding Box tại: {json_path}")

    print("\n" + "=" * 65)
    print("THÔNG SỐ BOUNDING BOX DÀNH CHO WEBGIS (EPSG:4326 WGS 84)")
    print("=" * 65)
    print(f"• Góc Tây - Nam (South-West) : Vĩ độ {south:.6f}°N, Kinh độ {west:.6f}°E")
    print(f"• Góc Đông - Bắc (North-East): Vĩ độ {north:.6f}°N, Kinh độ {east:.6f}°E")
    print(f"• Tâm bản đồ (Center)        : [{bbox_info['center'][0]:.6f}, {bbox_info['center'][1]:.6f}]")
    print("\nĐoạn mã mẫu nhúng ảnh trực tiếp vào Leaflet JS:")
    print("```javascript")
    print(f"const imageUrl = '{os.path.basename(png_path)}';")
    print(f"const imageBounds = [[{south:.6f}, {west:.6f}], [{north:.6f}, {east:.6f}]];")
    print("const landslideOverlay = L.imageOverlay(imageUrl, imageBounds, {")
    print("    opacity: 0.85,")
    print("    interactive: true")
    print("}).addTo(map);")
    print("map.fitBounds(imageBounds);")
    print("```")
    print("=" * 65)


# ==============================================================================
# HÀM THỰC THI CHÍNH
# ==============================================================================
def main():
    print("=" * 65)
    print("  ỨNG DỤNG RANDOM FOREST & ẢNH LANDSAT 8 KHOANH VÙNG NGUY CƠ TRƯỢT LỞ")
    print("=" * 65)

    # 1. Trích xuất mẫu và tính toán 12 kênh đặc trưng
    X, y = extract_training_data(
        image_path=IMAGE_PATH,
        roi_path=ROI_SHAPEFILE_PATH,
        class_col=CLASS_COLUMN_NAME
    )

    # 2. Cân bằng mẫu giữa các lớp (Undersampling <= 1500 mẫu/lớp)
    X_bal, y_bal = balance_classes_undersampling(
        X=X,
        y=y,
        max_samples_per_class=1500,
        random_state=42
    )

    # 3. Huấn luyện Random Forest và đánh giá mô hình (70% Train - 15% Val - 15% Test)
    rf_model = train_random_forest(
        X=X_bal,
        y=y_bal,
        n_estimators=150,
        max_depth=None,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    # 4. Lưu mô hình Random Forest
    joblib.dump(rf_model, MODEL_SAVE_PATH)
    print(f"[*] Đã lưu mô hình huấn luyện tại: {MODEL_SAVE_PATH}")

    # 5. Phân loại toàn cảnh 12 kênh, phân vùng 5 cấp nguy cơ và xuất WebGIS PNG
    classify_and_export_raster(
        image_path=IMAGE_PATH,
        model=rf_model,
        output_hazard_path=OUTPUT_HAZARD_RASTER_PATH,
        output_proba_path=OUTPUT_PROBA_RASTER_PATH,
        output_png_path=OUTPUT_WEBGIS_PNG_PATH,
        output_bbox_json=OUTPUT_WEBGIS_BBOX_JSON,
        landslide_class=LANDSLIDE_CLASS_LABEL,
        block_size=1024
    )


if __name__ == "__main__":
    main()
