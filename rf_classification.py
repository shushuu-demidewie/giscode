# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH PHÂN LOẠI VÀ KHOANH VÙNG NGUY CƠ TRƯỢT LỞ BẰNG RANDOM FOREST (RF)
KẾT HỢP ẢNH VIỄN THÁM LANDSAT 8 VÀ CÔNG NGHỆ WEBGIS
================================================================================

Đề tài: Nghiên cứu ứng dụng học máy/học sâu và ảnh viễn thám Landsat 8 
thành lập bản đồ / WebGIS khoanh vùng nguy cơ trượt lở.

Các tính năng chính:
1. Đọc ảnh 7 kênh quang học Landsat 8 (B1 - B7) và tự động tính 5 chỉ số phổ viễn thám:
   - NDVI (Normalized Difference Vegetation Index): Thảm phủ thực vật / sinh khối
   - NDBI (Normalized Difference Built-up Index): Đất xây dựng, đất khô cằn
   - MNDWI (Modified Normalized Difference Water Index): Nước mặt hở sạch
   - NDWI (Normalized Difference Water Index): Độ ẩm thực vật và mặt nước
   - BSI (Bare Soil Index): Đất trống, sườn dốc trơ sỏi đá / sạt lở
   -> Tổng cộng 12 kênh đặc trưng không gian - phổ đa chiều.
2. Tự động nhận diện dữ liệu thông minh (Smart Data Discovery) và cơ chế tự phục hồi (Self-Healing Fallback):
   - Quét tìm ảnh Landsat 8 thực tế trong workspace (landsat8_study_area.tif, Extracth8.tif).
   - Tự động đồng bộ CRS, tự động bổ sung mẫu Không sạt lở (0) từ vùng an toàn nếu Shapefile chỉ có điểm sạt lở (1).
3. Cân bằng mẫu (Undersampling khống chế lớp đa số, bảo đảm tỷ lệ học cân đối).
4. Phân chia tập dữ liệu 70% Train - 15% Validation - 15% Test (Stratified).
5. Huấn luyện Random Forest Classifier (n_estimators=150, max_features='sqrt', oob_score=True).
6. Đánh giá toàn diện: Accuracy, Precision, Recall, F1-Score, Cohen's Kappa, ROC-AUC, Confusion Matrix,
   và xuất biểu đồ trực quan hóa độ quan trọng của 12 kênh đặc trưng (Feature Importance).
7. Dự đoán theo khối (Block-wise / Windowed Prediction) chống tràn RAM, xuất 3 tệp GeoTIFF chuẩn:
   - Ảnh phân loại nhãn rời rạc
   - Bản đồ 5 cấp độ nguy cơ trượt lở (Rất thấp -> Rất cao)
   - Bản đồ xác suất nguy cơ trượt lở liên tục (0.0 -> 1.0)
8. Xuất file bảng màu (.clr) cho ArcGIS/QGIS, ảnh PNG nền trong suốt (RGBA) kèm tọa độ Bounding Box (EPSG:4326)
   chuẩn hóa cho nền tảng WebGIS (Leaflet / OpenLayers / Mapbox).
"""

import os
import sys

# Khắc phục triệt để lỗi PROJ database trên hệ điều hành Windows
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

import json
import warnings
import numpy as np
import pandas as pd
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

# Bỏ qua cảnh báo phép chia ma trận cho 0
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ TOÀN CỤC
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Danh sách ứng viên tìm kiếm ảnh viễn thám Landsat 8 (7 kênh B1 - B7)
CANDIDATE_IMAGES = [
    os.path.join(WORKSPACE_DIR, 'data', 'landsat8_study_area.tif'),
    os.path.join(BASE_DIR, 'data', 'Extracth8.tif'),
    os.path.join(BASE_DIR, 'data', 'dp2026Extrac.tif'),
    os.path.join(BASE_DIR, 'datahanoi', 'anhcat.tif'),
    os.path.join(BASE_DIR, 'data', 'anhcat.tif'),
    r"D:\DLGIS\hanoi2026\datahanoi\anhcat01.tif"
]

# Danh sách ứng viên tìm kiếm Shapefile mẫu sạt lở / kiểm chứng
CANDIDATE_ROIS = [
    os.path.join(WORKSPACE_DIR, 'lsm_results', 'demo_workspace', 'demo_inventory', 'landslides_demo.shp'),
    os.path.join(BASE_DIR, 'DKT2026', 'DKT2026.shp'),
    os.path.join(BASE_DIR, 'DKT', 'DKT.shp'),
    os.path.join(BASE_DIR, 'Dkt2015', 'DKT2015.shp'),
    os.path.join(BASE_DIR, 'Dkt', 'DKT30.shp'),
    os.path.join(BASE_DIR, 'data', 'ketquatach.shp')
]

# Tên cột chứa mã loại / lớp nhãn phân loại trong Shapefile
CLASS_COLUMN_NAME = "Nhan"

# Tùy chọn nhãn sạt lở (None = tự động phát hiện lớp 1 trong bài toán nhị phân, hoặc nhãn lớn nhất)
LANDSLIDE_CLASS_LABEL = None

# Đường dẫn lưu trữ kết quả đầu ra của thuật toán Random Forest
OUTPUT_RASTER_PATH = os.path.join(OUTPUT_DIR, 'anh_phan_loai_rf.tif')
OUTPUT_HAZARD_RASTER_PATH = os.path.join(OUTPUT_DIR, 'ban_do_5cap_nguy_co_truot_lo_rf.tif')
OUTPUT_PROBA_RASTER_PATH = os.path.join(OUTPUT_DIR, 'xac_suat_nguy_co_truot_lo_rf.tif')
OUTPUT_WEBGIS_PNG_PATH = os.path.join(OUTPUT_DIR, 'webgis_nguy_co_truot_lo_rf_rgba.png')
OUTPUT_WEBGIS_BBOX_JSON = os.path.join(OUTPUT_DIR, 'webgis_nguy_co_truot_lo_rf_bbox.json')
OUTPUT_IMPORTANCE_PNG = os.path.join(OUTPUT_DIR, 'rf_feature_importance.png')
OUTPUT_ROC_PNG = os.path.join(OUTPUT_DIR, 'rf_roc_confusion_matrix.png')

# File lưu mô hình huấn luyện Random Forest
MODEL_SAVE_PATH = os.path.join(WORKSPACE_DIR, 'rf_model.pkl')

# Tên 12 kênh đặc trưng phục vụ phân tích giải thích mô hình
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
# BƯỚC 1: TÍNH TOÁN 5 CHỈ SỐ PHỔ VIỄN THÁM TỪ 7 KÊNH LANDSAT 8
# ==============================================================================
def compute_spectral_indices(b1, b2, b3, b4, b5, b6, b7):
    """
    Tính toán 5 chỉ số phổ viễn thám từ 7 kênh quang học Landsat 8:
    1. NDVI  = (B5 - B4) / (B5 + B4) : Độ che phủ thực vật / sinh khối
    2. NDBI  = (B6 - B5) / (B6 + B5) : Đất xây dựng, vật liệu nhân tạo, đất khô
    3. MNDWI = (B3 - B6) / (B3 + B6) : Mặt nước sạch, tách biệt công trình
    4. NDWI  = (B3 - B5) / (B3 + B5) : Độ ẩm thực vật / nước mặt hở
    5. BSI   = ((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2)) : Chỉ số đất trống
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # 1. NDVI (NIR: B5, Red: B4)
        denom_ndvi = b5 + b4
        ndvi = np.where(denom_ndvi != 0, (b5 - b4) / denom_ndvi, 0.0)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. NDBI (SWIR1: B6, NIR: B5)
        denom_ndbi = b6 + b5
        ndbi = np.where(denom_ndbi != 0, (b6 - b5) / denom_ndbi, 0.0)
        ndbi = np.nan_to_num(ndbi, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. MNDWI (Green: B3, SWIR1: B6)
        denom_mndwi = b3 + b6
        mndwi = np.where(denom_mndwi != 0, (b3 - b6) / denom_mndwi, 0.0)
        mndwi = np.nan_to_num(mndwi, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. NDWI (Green: B3, NIR: B5)
        denom_ndwi = b3 + b5
        ndwi = np.where(denom_ndwi != 0, (b3 - b5) / denom_ndwi, 0.0)
        ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. BSI (Bare Soil Index)
        num_bsi = (b6 + b4) - (b5 + b2)
        denom_bsi = (b6 + b4) + (b5 + b2)
        bsi = np.where(denom_bsi != 0, num_bsi / denom_bsi, 0.0)
        bsi = np.nan_to_num(bsi, nan=0.0, posinf=0.0, neginf=0.0)

    return ndvi, ndbi, mndwi, ndwi, bsi


def add_spectral_indices_to_features(X_7bands):
    """
    Ghép 5 chỉ số phổ vào ma trận 7 kênh ban đầu, trả về ma trận 12 kênh đặc trưng.
    X_7bands: np.ndarray shape (N, 7) -> Trả về: shape (N, 12)
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
# BƯỚC 2: TÌM KIẾM DỮ LIỆU THÔNG MINH VÀ TRÍCH XUẤT MẪU HUẤN LUYỆN
# ==============================================================================
def resolve_data_paths():
    """
    Tự động tìm kiếm file ảnh Landsat 8 và Shapefile mẫu sẵn có trong thư mục làm việc.
    Nếu chưa có, tự động chuyển về file mẫu có sẵn nhằm đảm bảo chương trình chạy thông suốt.
    """
    image_path = None
    for p in CANDIDATE_IMAGES:
        if os.path.exists(p):
            image_path = p
            break

    if image_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy file ảnh Landsat 8 hợp lệ nào trong các đường dẫn cấu hình!\n"
            f"Vui lòng đặt file ảnh vào một trong các vị trí: {CANDIDATE_IMAGES}"
        )

    roi_path = None
    for p in CANDIDATE_ROIS:
        if os.path.exists(p):
            roi_path = p
            break

    return image_path, roi_path


def extract_training_data(image_path, roi_path=None, class_col=CLASS_COLUMN_NAME, random_state=42):
    """
    Trích xuất giá trị mẫu từ ảnh Landsat 8 theo Shapefile.
    Nếu Shapefile chỉ chứa các điểm sạt lở (1), tự động sinh thêm các điểm không sạt lở (0)
    ở các vùng có thực vật che phủ tốt (NDVI cao) và cách xa điểm sạt lở tối thiểu 200m.
    Nếu không có Shapefile, tự động trích xuất mẫu kiểm chứng thích ứng theo chỉ số phổ.
    """
    print(f"[*] Ảnh Landsat 8 đầu vào: {image_path}")

    with rasterio.open(image_path) as src:
        crs = src.crs
        transform = src.transform
        width, height = src.width, src.height
        nodata = src.nodata if src.nodata is not None else 0

        # Nếu có file Shapefile
        if roi_path and os.path.exists(roi_path):
            print(f"[*] Đang đọc Shapefile mẫu huấn luyện: {roi_path}")
            roi_gdf = gpd.read_file(roi_path)

            if roi_gdf.crs != crs:
                print(f"[*] Đồng bộ hệ quy chiếu Shapefile ({roi_gdf.crs}) -> Ảnh ({crs})...")
                roi_gdf = roi_gdf.to_crs(crs)

            # Tìm cột nhãn phù hợp
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

            # Lấy mẫu 7 kênh quang học
            sampled_values = np.array(list(src.sample(points_list)), dtype=np.float32)
            valid_mask = (sampled_values != nodata).all(axis=1) & np.isfinite(sampled_values).all(axis=1)
            
            X_7 = sampled_values[valid_mask]
            y = np.array(labels_list)[valid_mask]

            # Kiểm tra nếu chỉ có 1 lớp (ví dụ toàn bộ là điểm sạt lở 1)
            unique_classes = np.unique(y)
            if len(unique_classes) == 1:
                print(f"[*] Shapefile chỉ chứa 1 lớp nhãn {unique_classes}. Tự động sinh mẫu Không sạt lở (0)...")
                b4 = src.read(4).astype(np.float32)
                b5 = src.read(5).astype(np.float32)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi_map = np.where((b5 + b4) != 0, (b5 - b4) / (b5 + b4), 0.0)

                # Chọn các pixel có NDVI cao (rừng rậm, thảm phủ bền vững) làm vùng không sạt lở an toàn
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
            print("[!] Không tìm thấy Shapefile bên ngoài. Đang tự động trích xuất mẫu thích ứng theo chỉ số phổ...")
            # Đọc 7 kênh
            data_all = src.read().astype(np.float32)
            b2 = data_all[1]
            b3 = data_all[2]
            b4 = data_all[3]
            b5 = data_all[4]
            b6 = data_all[5]

            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = np.where((b5 + b4) != 0, (b5 - b4) / (b5 + b4), 0.0)
                bsi = np.where(((b6 + b4) + (b5 + b2)) != 0, ((b6 + b4) - (b5 + b2)) / ((b6 + b4) + (b5 + b2)), 0.0)

            valid_raster = (data_all != nodata).all(axis=0) & np.isfinite(data_all).all(axis=0)

            # Sạt lở / Đất trống (BSI cao, NDVI thấp) -> Nhãn 1
            ls_candidates = np.argwhere(valid_raster & (bsi > 0.15) & (ndvi < 0.25))
            # Không sạt lở / Thảm phủ dày (NDVI cao) -> Nhãn 0
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
            print(f"[+] Đã tự động khởi tạo {len(y)} mẫu cân bằng (50% Nguy cơ sạt lở : 50% An toàn).")

    print(f"[+] Trích xuất thành công {X_7.shape[0]} điểm mẫu với {X_7.shape[1]} kênh gốc Landsat 8.")

    # Tính toán 5 chỉ số phổ -> tạo 12 kênh đặc trưng
    print("[*] Đang tính toán 5 chỉ số phổ viễn thám: NDVI, NDBI, MNDWI, NDWI, BSI...")
    X_12 = add_spectral_indices_to_features(X_7)
    print(f"[+] Tổng số kênh đặc trưng đầu vào sau khi bổ sung chỉ số phổ: {X_12.shape[1]} kênh.")

    return X_12, y


# ==============================================================================
# BƯỚC 3: CÂN BẰNG MẪU VÀ PHÂN CHIA TẬP DỮ LIỆU (70/15/15)
# ==============================================================================
def balance_and_split_data(X, y, max_samples_per_class=1500, random_state=42):
    """
    Cân bằng mẫu (Undersampling) và chia 70% Train - 15% Validation - 15% Test (Stratified).
    """
    print(f"\n[*] Cân bằng mẫu (Khống chế tối đa {max_samples_per_class} mẫu/lớp)...")
    df = pd.DataFrame(X)
    df['__label__'] = y

    counts = df['__label__'].value_counts()
    print("    - Số lượng mẫu ban đầu theo từng lớp:")
    for cls_val, cnt in counts.items():
        print(f"      + Lớp {cls_val}: {cnt} mẫu")

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
        print(f"      + Lớp {cls_val}: {cnt} mẫu")

    # Phân chia tập dữ liệu 70% Train - 15% Val - 15% Test
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
# BƯỚC 4: HUẤN LUYỆN RANDOM FOREST VÀ ĐÁNH GIÁ MÔ HÌNH TOÀN DIỆN
# ==============================================================================
def train_and_evaluate_rf(data_dict, n_estimators=150, random_state=42):
    """
    Huấn luyện Random Forest Classifier, đánh giá chi tiết trên Validation và Test:
    - Overall Accuracy, Precision, Recall, F1-Score
    - Cohen's Kappa score
    - ROC-AUC score
    - Xuất biểu đồ Feature Importances và ROC Curve / Confusion Matrix.
    """
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

    print(f"\n[*] Đang huấn luyện Random Forest ({n_estimators} cây, tiêu chí Gini, max_features='sqrt')...")
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

    if hasattr(rf_clf, "oob_score_"):
        print(f"[+] Out-Of-Bag (OOB) Score: {rf_clf.oob_score_ * 100:.2f}%")

    # Đánh giá trên Validation và Test
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
            if len(rf_clf.classes_) == 2:
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
        print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST TRÊN TẬP {set_name.upper()}")
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
    print("BẢNG ĐỘ QUAN TRỌNG CỦA 12 KÊNH ĐẶC TRƯNG LANDSAT 8 (FEATURE IMPORTANCE)")
    print("=" * 65)
    for rank, idx in enumerate(indices, 1):
        cname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"Kênh {idx+1}"
        bar = "█" * int(round(importances[idx] * 40))
        print(f"  {rank:2d}. {cname:<34} : {importances[idx]*100:6.2f}%  {bar}")
    print("=" * 65)

    plot_rf_charts(rf_clf, data_dict, importances, indices)

    return rf_clf, eval_results


def plot_rf_charts(rf_clf, data_dict, importances, indices):
    """Vẽ và lưu biểu đồ Feature Importance và ROC Curve / Confusion Matrix."""
    if plt is None:
        return

    try:
        # 1. Biểu đồ Feature Importance
        plt.figure(figsize=(9, 5), dpi=300)
        sorted_names = [FEATURE_NAMES[i] for i in indices][::-1]
        sorted_vals = [importances[i] * 100 for i in indices][::-1]
        bars = plt.barh(sorted_names, sorted_vals, color='#2c7bb6', edgecolor='#1a4c73', height=0.6)
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                     ha='left', va='center', fontsize=9, fontweight='bold', color='#111111')
        plt.title("Độ quan trọng của 12 kênh đặc trưng Landsat 8 (Random Forest)", fontsize=11, fontweight='bold')
        plt.xlabel("Tỷ lệ đóng góp (%)", fontsize=10)
        plt.xlim(0, max(sorted_vals) + 5)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUTPUT_IMPORTANCE_PNG)
        plt.close()
        print(f"[+] Đã xuất biểu đồ Feature Importance: {OUTPUT_IMPORTANCE_PNG}")

        # 2. Biểu đồ ROC Curve và Confusion Matrix
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        preds = rf_clf.predict(X_test)
        cm = confusion_matrix(y_test, preds)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

        # ROC Curve (nếu nhị phân)
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
            axes[0].set_title('Đường cong ROC (Random Forest)', fontsize=10, fontweight='bold')
            axes[0].legend(loc="lower right", fontsize=9)
            axes[0].grid(True, linestyle=':', alpha=0.6)
        else:
            axes[0].text(0.5, 0.5, "Bài toán phân loại đa lớp\n(Multi-class ROC xem trong báo cáo)",
                         ha='center', va='center', fontsize=10)
            axes[0].set_title('Đường cong ROC', fontsize=10, fontweight='bold')

        # Confusion Matrix
        cax = axes[1].matshow(cm, cmap='Blues', alpha=0.85)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                axes[1].text(j, i, str(val), ha='center', va='center', fontsize=12, fontweight='bold',
                             color='white' if val > cm.max()/2 else 'black')
        axes[1].set_title('Ma trận nhầm lẫn (Test Set)', fontsize=10, fontweight='bold', pad=15)
        axes[1].set_xlabel('Dự đoán (Predicted)', fontsize=9, labelpad=10)
        axes[1].set_ylabel('Thực tế (Actual)', fontsize=9)
        fig.colorbar(cax, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(OUTPUT_ROC_PNG)
        plt.close()
        print(f"[+] Đã xuất biểu đồ đánh giá mô hình: {OUTPUT_ROC_PNG}")
    except Exception as e:
        print(f"[!] Lỗi khi xuất đồ thị đánh giá: {e}")


# ==============================================================================
# BƯỚC 5: DỰ ĐOÁN THEO KHỐI (BLOCK-WISE) VÀ XUẤT CÁC TỆP BẢN ĐỒ / WEBGIS
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
    Phân loại toàn bộ ảnh GeoTIFF theo từng khối (window/block):
    - Tính 5 chỉ số phổ cho từng khối -> 12 kênh.
    - Dự đoán nhãn phân loại (Discrete LULC / Hazard class).
    - Dự đoán xác suất liên tục (P(landslide)) và phân cấp 5 mức độ nguy cơ:
      1: Rất thấp (0.0 - 0.2) -> Xanh lá
      2: Thấp (0.2 - 0.4) -> Xanh lục nhạt
      3: Trung bình (0.4 - 0.6) -> Vàng
      4: Cao (0.6 - 0.8) -> Da cam
      5: Rất cao (0.8 - 1.0) -> Đỏ
    - Xuất GeoTIFF, file bảng màu ArcMap (.clr), ảnh PNG nền trong suốt RGBA và file JSON Bounding Box.
    """
    print(f"\n[*] Bắt đầu suy luận không gian toàn vùng bằng Random Forest: {image_path}")
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
        else:
            landslide_idx = len(classes) - 1

        print(f"[*] Lớp đại diện sạt lở: '{classes[landslide_idx]}' (index={landslide_idx})")

        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

        hazard_colors = {
            1: [56, 168, 0, 220],     # Cấp 1: Rất thấp (Xanh lá)
            2: [176, 224, 0, 220],    # Cấp 2: Thấp (Xanh lục nhạt)
            3: [255, 255, 0, 220],    # Cấp 3: Trung bình (Vàng)
            4: [255, 128, 0, 230],    # Cấp 4: Cao (Da cam)
            5: [255, 0, 0, 240],      # Cấp 5: Rất cao (Đỏ)
        }

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

                        # Dự đoán nhãn
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
                        for level, color in hazard_colors.items():
                            rgba_image[row:row+h_chunk, col:col+w_chunk][block_hazard_2d == level] = color

                    dst_discrete.write(block_discrete.reshape((1, h_chunk, w_chunk)), window=window)
                    dst_hazard.write(block_hazard.reshape((1, h_chunk, w_chunk)), window=window)
                    dst_proba.write(block_proba.reshape((1, h_chunk, w_chunk)), window=window)

                    if block_count % 10 == 0 or block_count == total_blocks:
                        print(f"    - Tiến độ phân loại RF: {block_count}/{total_blocks} khối ({block_count/total_blocks*100:.1f}%)")

        print(f"[+] Đã xuất GeoTIFF phân loại nhãn RF:           {output_discrete_path}")
        print(f"[+] Đã xuất GeoTIFF bản đồ 5 cấp nguy cơ RF:      {output_hazard_path}")
        print(f"[+] Đã xuất GeoTIFF xác suất nguy cơ trượt lở RF: {output_proba_path}")

        # Xuất file màu .clr
        export_hazard_clr(output_hazard_path, hazard_colors)

        # Xuất WebGIS Assets (PNG RGBA + JSON BBOX)
        export_webgis_assets(
            rgba_image=rgba_image,
            src_crs=crs,
            src_bounds=bounds,
            png_path=output_png_path,
            json_path=output_bbox_json,
            model_name="Random Forest"
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
    print("=" * 70)
    print("  ỨNG DỤNG RANDOM FOREST & ẢNH LANDSAT 8 KHOANH VÙNG NGUY CƠ TRƯỢT LỞ")
    print("  Đề tài: Nghiên cứu ứng dụng học máy và viễn thám Landsat 8 lập WebGIS")
    print("=" * 70)

    # 1. Tự động tìm kiếm đường dẫn dữ liệu ảnh và shapefile
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
    rf_model, eval_results = train_and_evaluate_rf(
        data_dict=data_dict,
        n_estimators=150,
        random_state=42
    )

    # 5. Lưu mô hình Random Forest
    joblib.dump(rf_model, MODEL_SAVE_PATH)
    print(f"[*] Đã lưu mô hình Random Forest tại: {MODEL_SAVE_PATH}")

    # 6. Phân loại toàn ảnh, phân vùng 5 cấp nguy cơ và xuất tài nguyên WebGIS
    classify_and_export_raster(
        image_path=image_path,
        model=rf_model,
        output_discrete_path=OUTPUT_RASTER_PATH,
        output_hazard_path=OUTPUT_HAZARD_RASTER_PATH,
        output_proba_path=OUTPUT_PROBA_RASTER_PATH,
        output_png_path=OUTPUT_WEBGIS_PNG_PATH,
        output_bbox_json=OUTPUT_WEBGIS_BBOX_JSON,
        landslide_class=LANDSLIDE_CLASS_LABEL,
        block_size=512
    )

    print("\n" + "=" * 70)
    print("HOÀN TẤT THÀNH CÔNG QUY TRÌNH RANDOM FOREST CHO ĐỀ TÀI TRƯỢT LỞ ĐẤT!")
    print(f"Các file kết quả được lưu tại: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
