# -*- coding: utf-8 -*-
"""
Quy trình phân loại và khoanh vùng nguy cơ trượt lở bằng mô hình Support Vector Machine (SVM)
với 12 kênh đặc trưng từ ảnh viễn thám Landsat 8 (7 kênh quang học + 5 chỉ số phổ viễn thám).

Các tính năng chính:
1. Đọc và tính toán 12 kênh đặc trưng:
   - 7 kênh quang học gốc Landsat 8 (B1 - B7).
   - 5 chỉ số phổ chuyên sâu: NDVI, NDBI, MNDWI, NDWI và BSI (Bare Soil Index).
2. Cân bằng mẫu giữa các lớp (Undersampling <= 1500 mẫu/lớp).
3. Tiêu chuẩn hóa dữ liệu bằng StandardScaler và huấn luyện SVM RBF với probability=True.
4. Đánh giá toàn diện mô hình: Accuracy, Cohen's Kappa, ROC-AUC (OVR), Confusion Matrix.
5. Lưu trọn bộ StandardScaler và SVM Model vào file svm_model.pkl bằng joblib (dưới dạng Pipeline).
6. Hàm classify_and_export_raster phân loại theo từng khối (block_size=512) có chuẩn hóa dữ liệu,
   xuất GeoTIFF phân loại và phân vùng 5 cấp nguy cơ trượt lở.
7. Xuất file bảng màu .clr cho ArcMap, ảnh .png nền trong suốt (RGBA) kèm Bounding Box EPSG:4326
   cho WebGIS (Leaflet / OpenLayers / Mapbox).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.warp import transform_bounds
import geopandas as gpd

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    plt = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    roc_auc_score
)
import joblib

# Bỏ qua các cảnh báo chia 0 trong raster và cảnh báo tham số thư viện
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi font/mã hóa
if sys.stdout and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Tự động tạo lại file .shx nếu bị thiếu khi đọc Shapefile
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Tự động khắc phục xung đột PROJ_LIB / PROJ_DATA giữa PostgreSQL và Python pyproj
try:
    import pyproj.datadir
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ['PROJ_LIB'] = proj_dir
    os.environ['PROJ_DATA'] = proj_dir
except Exception:
    pass


# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU VÀ ĐẦU RA
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Danh sách ứng viên tìm kiếm ảnh viễn thám
CANDIDATE_IMAGES = [
    os.path.join(BASE_DIR, 'data', 'dp2026Extrac.tif'),
    os.path.join(BASE_DIR, 'data', 'longchoose1.tif'),
    os.path.join(BASE_DIR, 'datahanoi', 'anhcat.tif'),
    os.path.join(BASE_DIR, 'data', 'anhcat.tif'),
    r"D:\DLGIS\hanoi2026\datahanoi\anhcat01.tif"
]
IMAGE_PATH = next((p for p in CANDIDATE_IMAGES if os.path.exists(p)), CANDIDATE_IMAGES[0])

# Danh sách ứng viên tìm kiếm Shapefile mẫu
CANDIDATE_ROIS = [
    os.path.join(BASE_DIR, 'DKT2026', 'DKT2026.shp'),
    os.path.join(BASE_DIR, 'Dkt2015', 'DKT2015.shp'),
    os.path.join(BASE_DIR, 'Dkt', 'DKT30.shp'),
    os.path.join(BASE_DIR, 'data', 'ketquatach.shp')
]
ROI_SHAPEFILE_PATH = next((p for p in CANDIDATE_ROIS if os.path.exists(p)), CANDIDATE_ROIS[0])

# Shapefile ranh giới xã (tùy chọn)
XA_SHP_PATH = os.path.join(BASE_DIR, 'data', 'KQ_Tach01.shp')

# Thư mục lưu kết quả
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tên cột nhãn phân loại trong Shapefile
CLASS_COLUMN_NAME = "Nhan"

# File lưu mô hình SVM (kèm StandardScaler)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'giscode', 'svm_model.pkl')

# Các file bản đồ đầu ra
OUTPUT_RASTER_PATH = os.path.join(OUTPUT_DIR, 'anh_phan_loai_svm.tif')
OUTPUT_HAZARD_RASTER_PATH = os.path.join(OUTPUT_DIR, 'ban_do_5cap_nguy_co_truot_lo_svm.tif')
OUTPUT_PROBA_RASTER_PATH = os.path.join(OUTPUT_DIR, 'xac_suat_nguy_co_truot_lo_svm.tif')
OUTPUT_WEBGIS_PNG_PATH = os.path.join(OUTPUT_DIR, 'webgis_nguy_co_truot_lo_svm_rgba.png')
OUTPUT_WEBGIS_BBOX_JSON = os.path.join(OUTPUT_DIR, 'webgis_nguy_co_truot_lo_svm_bbox.json')
OUTPUT_SPECTRAL_PLOT = os.path.join(OUTPUT_DIR, 'chi_so_pho_svm.png')

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


# ==============================================================================
# BƯỚC 1: TÍNH TOÁN 5 CHỈ SỐ PHỔ VIỄN THÁM (12 KÊNH ĐẶC TRƯNG)
# ==============================================================================
def compute_spectral_indices(b1, b2, b3, b4, b5, b6, b7):
    """
    Tính toán 5 chỉ số phổ từ 7 kênh quang học Landsat 8:
    1. NDVI  = (B5 - B4) / (B5 + B4) : Độ che phủ thực vật / sinh khối
    2. NDBI  = (B6 - B5) / (B6 + B5) : Vật liệu xây dựng, đất khô
    3. MNDWI = (B3 - B6) / (B3 + B6) : Mặt nước sạch, tách biệt công trình
    4. NDWI  = (B3 - B5) / (B3 + B5) : Độ ẩm thực vật / nước mặt hở
    5. BSI   = ((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2)) : Chỉ số đất trống (Bare Soil Index)
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

        # 5. BSI (Bare Soil Index)
        bsi = ((b6 + b4) - (b5 + b2)) / ((b6 + b4) + (b5 + b2))
        bsi = np.nan_to_num(bsi, nan=0.0, posinf=0.0, neginf=0.0)

    return ndvi, ndbi, mndwi, ndwi, bsi


def add_spectral_indices_to_features(X_7bands):
    """
    Ghép 5 chỉ số phổ vào ma trận 7 kênh ban đầu, trả về ma trận 12 kênh đặc trưng.
    X_7bands: shape (N, 7) -> Trả về: shape (N, 12)
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
# BƯỚC 2: TRÍCH XUẤT MẪU ĐẶC TRƯNG TỪ SHAPEFILE VÀ ẢNH VIỄN THÁM
# ==============================================================================
def extract_training_data(image_path, roi_path, class_col):
    """
    Đọc ảnh viễn thám Landsat 8, trích xuất pixel mẫu từ Shapefile (Point hoặc Polygon),
    và tính toán 12 kênh đặc trưng.
    """
    print(f"[*] Đang đọc ảnh viễn thám đầu vào: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"Không tìm thấy file Shapefile: {roi_path}")

    print(f"[*] Đang đọc Shapefile mẫu huấn luyện: {roi_path}")
    roi_gdf = gpd.read_file(roi_path)

    # Tự động xác định cột nhãn
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
        if roi_gdf.crs != src.crs:
            print(f"[*] Đồng bộ hệ quy chiếu Shapefile ({roi_gdf.crs}) -> Ảnh ({src.crs})...")
            roi_gdf = roi_gdf.to_crs(src.crs)

        geom_types = roi_gdf.geometry.geom_type.unique()
        print(f"[*] Kiểu đối tượng mẫu (Geometry): {geom_types}")

        # Trường hợp 1: Dữ liệu POINT
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

        # Trường hợp 2: Dữ liệu POLYGON
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
# BƯỚC 3: CÂN BẰNG MẪU (UNDERSAMPLING TỐI ĐA 1500 MẪU/LỚP)
# ==============================================================================
def balance_classes_undersampling(X, y, max_samples_per_class=1500, random_state=42):
    """
    Cân bằng mẫu giữa các lớp bằng Undersampling: khống chế tối đa 1500 mẫu/lớp
    giúp mô hình SVM phân tách biên công bằng, tránh overfit vào lớp đa số.
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

    if target_samples >= 50:
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
        print(f"      + Lớp {cls_val}: {cnt} mẫu")

    return X_bal, y_bal


# ==============================================================================
# BƯỚC 4: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH SVM (VỚI PIPELINE & PROBABILITY=TRUE)
# ==============================================================================
def train_svm_pipeline(X, y, C=1.0, kernel='rbf', gamma='scale', train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Huấn luyện Support Vector Machine (SVM) kết hợp chuẩn hóa dữ liệu:
    1. Tách tập 70% Train - 15% Val - 15% Test.
    2. Đóng gói StandardScaler và SVC(probability=True) thành một Pipeline hoàn chỉnh.
    3. Đánh giá Accuracy, Cohen's Kappa và ROC-AUC trên cả Validation và Test.
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

    print(f"\n[*] Đang huấn luyện mô hình SVM ({kernel.upper()} kernel, C={C}, probability=True)...")

    # Đóng gói Pipeline: Chuẩn hóa dữ liệu + Mô hình SVM
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=random_state))
    ])

    # Huấn luyện Pipeline trên tập Train
    svm_pipeline.fit(X_train, y_train)

    def evaluate_set(name, X_set, y_set):
        preds = svm_pipeline.predict(X_set)
        acc = accuracy_score(y_set, preds)
        kappa = cohen_kappa_score(y_set, preds)

        auc_str = "N/A"
        try:
            probas = svm_pipeline.predict_proba(X_set)
            classes = svm_pipeline.classes_
            if len(classes) == 2:
                auc_val = roc_auc_score(y_set, probas[:, 1])
                auc_str = f"{auc_val:.4f}"
            elif len(classes) > 2:
                auc_val = roc_auc_score(y_set, probas, multi_class='ovr', average='weighted')
                auc_str = f"{auc_val:.4f} (OVR Weighted)"
        except Exception as e:
            auc_str = f"Không tính được ({e})"

        print("\n" + "=" * 60)
        print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH SVM TRÊN TẬP {name.upper()}")
        print("=" * 60)
        print(f"- Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%")
        print(f"- Hệ số Cohen's Kappa:             {kappa:.4f}")
        print(f"- Chỉ số ROC-AUC:                  {auc_str}")
        print(f"\nBáo cáo chi tiết (Classification Report) tập {name}:")
        print(classification_report(y_set, preds))
        print(f"Ma trận nhầm lẫn (Confusion Matrix) tập {name}:")
        print(confusion_matrix(y_set, preds))

    evaluate_set("Validation", X_val, y_val)
    evaluate_set("Test (Kiểm tra độc lập)", X_test, y_test)

    return svm_pipeline


# ==============================================================================
# BƯỚC 5: PHÂN LOẠI TOÀN BỘ ẢNH THEO KHỐI (BLOCK 512), PHÂN VÙNG VÀ XUẤT WEBGIS
# ==============================================================================
def classify_and_export_raster(
    image_path,
    pipeline,
    output_discrete_path=OUTPUT_RASTER_PATH,
    output_hazard_path=OUTPUT_HAZARD_RASTER_PATH,
    output_proba_path=OUTPUT_PROBA_RASTER_PATH,
    output_png_path=OUTPUT_WEBGIS_PNG_PATH,
    output_bbox_json=OUTPUT_WEBGIS_BBOX_JSON,
    landslide_class=None,
    block_size=512
):
    """
    Phân loại toàn bộ ảnh GeoTIFF theo từng khối (block_size=512):
    1. Đọc khối dữ liệu 7 kênh gốc.
    2. Tính toán 5 chỉ số phổ (tổng 12 kênh).
    3. Áp dụng pipeline (StandardScaler + SVC) để phân loại và dự đoán xác suất trượt lở.
    4. Phân vùng 5 cấp nguy cơ trượt lở:
       - Cấp 1 (Rất thấp): 0.00 <= P < 0.20  (Xanh lá)
       - Cấp 2 (Thấp)    : 0.20 <= P < 0.40  (Xanh lục nhạt)
       - Cấp 3 (Trung bình): 0.40 <= P < 0.60 (Vàng)
       - Cấp 4 (Cao)     : 0.60 <= P < 0.80  (Da cam)
       - Cấp 5 (Rất cao) : 0.80 <= P <= 1.00 (Đỏ)
    5. Xuất file GeoTIFF kết quả, file bảng màu ArcMap (.clr), ảnh PNG nền trong suốt (RGBA)
       và tọa độ Bounding Box (EPSG:4326) phục vụ WebGIS.
    """
    print(f"\n[*] Bắt đầu xử lý và phân loại toàn bộ ảnh viễn thám bằng SVM: {image_path}")
    print(f"[*] Kích thước khối xử lý (block_size): {block_size}x{block_size} pixels")

    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width
        crs = src.crs
        bounds = src.bounds
        nodata_val = src.nodata

        # Metadata GeoTIFF phân loại nhãn rời rạc
        discrete_profile = src.profile.copy()
        discrete_profile.update(
            dtype=rasterio.int32,
            count=1,
            nodata=-9999,
            compress='lzw'
        )

        # Metadata GeoTIFF bản đồ 5 cấp nguy cơ trượt lở
        hazard_profile = src.profile.copy()
        hazard_profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=-9999,
            compress='lzw'
        )

        # Metadata GeoTIFF xác suất nguy cơ trượt lở liên tục (Float32)
        proba_profile = src.profile.copy()
        proba_profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=-9999.0,
            compress='lzw'
        )

        # Xác định cột xác suất nguy cơ trượt lở
        classes = list(pipeline.classes_)
        if landslide_class is not None and landslide_class in classes:
            landslide_idx = classes.index(landslide_class)
        else:
            landslide_idx = len(classes) - 1

        print(f"[*] Xác suất của lớp '{classes[landslide_idx]}' được sử dụng làm thước đo nguy cơ trượt lở.")

        # Mảng RGBA kích thước đầy đủ để xuất PNG nền trong suốt cho WebGIS
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

        # Bảng màu 5 cấp độ nguy cơ trượt lở [R, G, B, Alpha]:
        hazard_colors = {
            1: [56, 168, 0, 220],     # Cấp 1: Rất thấp (Xanh lá)
            2: [176, 224, 0, 220],    # Cấp 2: Thấp (Xanh lục nhạt)
            3: [255, 255, 0, 220],    # Cấp 3: Trung bình (Vàng)
            4: [255, 128, 0, 230],    # Cấp 4: Cao (Da cam)
            5: [255, 0, 0, 240],      # Cấp 5: Rất cao (Đỏ)
        }

        with rasterio.open(output_discrete_path, 'w', **discrete_profile) as dst_discrete, \
             rasterio.open(output_hazard_path, 'w', **hazard_profile) as dst_hazard, \
             rasterio.open(output_proba_path, 'w', **proba_profile) as dst_proba:

            for row in range(0, height, block_size):
                h_chunk = min(block_size, height - row)

                for col in range(0, width, block_size):
                    w_chunk = min(block_size, width - col)
                    window = rasterio.windows.Window(col, row, w_chunk, h_chunk)

                    block_data = src.read(window=window)
                    reshaped_7 = block_data.reshape(src.count, -1).T

                    if nodata_val is not None:
                        valid_mask = (reshaped_7 != nodata_val).all(axis=1)
                    else:
                        valid_mask = (reshaped_7 != 0).any(axis=1)

                    block_discrete = np.full(reshaped_7.shape[0], -9999, dtype=np.int32)
                    block_hazard = np.full(reshaped_7.shape[0], -9999, dtype=np.int16)
                    block_proba = np.full(reshaped_7.shape[0], -9999.0, dtype=np.float32)

                    if np.any(valid_mask):
                        valid_pixels_7 = reshaped_7[valid_mask]

                        # 1. Bổ sung 5 chỉ số phổ -> 12 kênh đặc trưng
                        valid_pixels_12 = add_spectral_indices_to_features(valid_pixels_7)

                        # 2. Dự đoán nhãn phân loại (Pipeline tự động áp dụng StandardScaler)
                        block_discrete[valid_mask] = pipeline.predict(valid_pixels_12)

                        # 3. Dự đoán xác suất trượt lở bằng SVM predict_proba
                        probas = pipeline.predict_proba(valid_pixels_12)
                        p_landslide = probas[:, landslide_idx]
                        block_proba[valid_mask] = p_landslide

                        # 4. Phân vùng 5 cấp nguy cơ trượt lở
                        hazard_levels = np.digitize(p_landslide, bins=[0.2, 0.4, 0.6, 0.8]) + 1
                        hazard_levels = np.clip(hazard_levels, 1, 5).astype(np.int16)
                        block_hazard[valid_mask] = hazard_levels

                        # 5. Gán màu RGBA cho ảnh WebGIS
                        block_hazard_2d = block_hazard.reshape((h_chunk, w_chunk))
                        for level, color in hazard_colors.items():
                            match_mask = (block_hazard_2d == level)
                            rgba_image[row:row+h_chunk, col:col+w_chunk][match_mask] = color

                    # Ghi khối vào các file GeoTIFF
                    dst_discrete.write(block_discrete.reshape((1, h_chunk, w_chunk)), window=window)
                    dst_hazard.write(block_hazard.reshape((1, h_chunk, w_chunk)), window=window)
                    dst_proba.write(block_proba.reshape((1, h_chunk, w_chunk)), window=window)

        print(f"[+] Đã xuất GeoTIFF phân loại SVM:               {output_discrete_path}")
        print(f"[+] Đã xuất GeoTIFF bản đồ 5 cấp nguy cơ trượt lở: {output_hazard_path}")
        print(f"[+] Đã xuất GeoTIFF xác suất nguy cơ trượt lở:       {output_proba_path}")

        # Xuất file bảng màu .clr cho ArcMap
        export_hazard_clr(output_hazard_path, hazard_colors)

        # Xuất ảnh PNG nền trong suốt và metadata Bounding Box EPSG:4326 cho WebGIS
        export_webgis_assets(
            rgba_image=rgba_image,
            src_crs=crs,
            src_bounds=bounds,
            png_path=output_png_path,
            json_path=output_bbox_json
        )


def export_hazard_clr(raster_path, hazard_colors):
    """
    Tạo file .clr đi kèm file GeoTIFF để ArcMap/QGIS tự nhận bảng màu phân loại 5 cấp.
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
    Xuất ảnh màu PNG nền trong suốt (RGBA) và chuyển đổi Bounding Box sang EPSG:4326 (WGS 84).
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

    try:
        min_x, min_y, max_x, max_y = src_bounds
        west, south, east, north = transform_bounds(src_crs, 'EPSG:4326', min_x, min_y, max_x, max_y)
    except Exception as e:
        print(f"[!] Chuyển đổi CRS sang EPSG:4326 gặp lỗi: {e}. Sử dụng bounds gốc.")
        west, south, east, north = src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top

    bbox_info = {
        "crs": "EPSG:4326",
        "description": "Tọa độ Bounding Box hiển thị ảnh bản đồ nguy cơ trượt lở SVM trên WebGIS",
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
    print("  ỨNG DỤNG SVM & ẢNH LANDSAT 8 KHOANH VÙNG NGUY CƠ TRƯỢT LỞ")
    print("=" * 65)

    # 1. Trích xuất mẫu và tính toán 12 kênh đặc trưng
    X, y = extract_training_data(
        image_path=IMAGE_PATH,
        roi_path=ROI_SHAPEFILE_PATH,
        class_col=CLASS_COLUMN_NAME
    )

    # 2. Cân bằng mẫu (Undersampling <= 1500 mẫu/lớp)
    X_bal, y_bal = balance_classes_undersampling(
        X=X,
        y=y,
        max_samples_per_class=1500,
        random_state=42
    )

    # 3. Huấn luyện SVM Pipeline (StandardScaler + SVC probability=True)
    svm_pipeline = train_svm_pipeline(
        X=X_bal,
        y=y_bal,
        C=1.0,
        kernel='rbf',
        gamma='scale',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    # 4. Lưu cả StandardScaler và SVM model vào file svm_model.pkl
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_SAVE_PATH)), exist_ok=True)
    joblib.dump(svm_pipeline, MODEL_SAVE_PATH)
    print(f"[*] Đã lưu thành công StandardScaler và mô hình SVM tại: {MODEL_SAVE_PATH}")

    # 5. Phân loại toàn cảnh 12 kênh bằng block 512, phân vùng 5 cấp nguy cơ và xuất WebGIS
    classify_and_export_raster(
        image_path=IMAGE_PATH,
        pipeline=svm_pipeline,
        output_discrete_path=OUTPUT_RASTER_PATH,
        output_hazard_path=OUTPUT_HAZARD_RASTER_PATH,
        output_proba_path=OUTPUT_PROBA_RASTER_PATH,
        output_png_path=OUTPUT_WEBGIS_PNG_PATH,
        output_bbox_json=OUTPUT_WEBGIS_BBOX_JSON,
        block_size=512
    )


if __name__ == "__main__":
    main()