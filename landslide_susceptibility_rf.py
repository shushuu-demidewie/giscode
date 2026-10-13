# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH ĐÁNH GIÁ ĐỘ NHẠY CẢM SẠT LỞ ĐẤT (LANDSLIDE SUSCEPTIBILITY MAPPING - LSM)
Áp dụng Học máy Random Forest kết hợp GIS (Rasterio, GeoPandas, Scikit-learn)
================================================================================

Các tính năng cốt lõi:
1. Đồng bộ (Reproject/Resample) và gộp (Stack) 8 nhân tố trượt lở (Conditioning Factors):
   - Địa hình: Slope, Aspect, Curvature, TWI
   - Môi trường & Lớp phủ: NDVI, LULC
   - Nhân tố nhân sinh / Thủy văn: Distance to rivers, Distance to roads
2. Đọc Shapefile điểm sạt lở (Nhãn 1) và tự động sinh mẫu ngẫu nhiên không sạt lở (Nhãn 0)
   trong vùng dốc thấp (low slope) có kiểm soát khoảng cách an toàn.
3. Trích xuất giá trị đa tầng (Multi-band extraction), làm sạch NoData/NaN/Inf và cân bằng mẫu (50:50).
4. Phân chia Train / Val / Test (70 / 15 / 15) và chuẩn hóa bằng StandardScaler (chống rò rỉ dữ liệu).
5. Huấn luyện Random Forest (n_estimators=200) và trực quan hóa Feature Importance.
6. Đánh giá toàn diện: Accuracy, F1-score, Cohen's Kappa, Confusion Matrix, ROC Curve & AUC.
7. Dự đoán theo từng khối cửa sổ (Block-wise / Windowed predict_proba) để tránh tràn RAM,
   xuất ra GeoTIFF Chỉ số Nhạy cảm Sạt lở đất (LSI: 0 -> 1) và bản đồ phân cấp màu chuyên nghiệp.
"""

import os
import sys

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi font/mã hóa
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Khắc phục xung đột PROJ trên Windows (do PostgreSQL/PostGIS/QGIS đặt PROJ_LIB cũ vào biến môi trường hệ thống)
try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ['PROJ_DATA'] = proj_dir
    os.environ['PROJ_LIB'] = proj_dir
except Exception:
    pass

# Cho phép GDAL tự động phục hồi nếu thiếu file .shx
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import Window
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

# Định vị thư mục làm việc mặc định
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'lsm_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU ĐẦU VÀO (TÙY CHỈNH THEO DỰ ÁN CỦA BẠN)
# ==============================================================================
# 8 nhân tố trượt lở (Conditioning Factors)
FACTOR_RASTERS = {
    "Slope": os.path.join(BASE_DIR, "factors", "slope.tif"),
    "Aspect": os.path.join(BASE_DIR, "factors", "aspect.tif"),
    "Curvature": os.path.join(BASE_DIR, "factors", "curvature.tif"),
    "TWI": os.path.join(BASE_DIR, "factors", "twi.tif"),
    "NDVI": os.path.join(BASE_DIR, "factors", "ndvi.tif"),
    "LULC": os.path.join(BASE_DIR, "factors", "lulc.tif"),
    "Dist_Rivers": os.path.join(BASE_DIR, "factors", "dist_rivers.tif"),
    "Dist_Roads": os.path.join(BASE_DIR, "factors", "dist_roads.tif"),
}

# Shapefile chứa các điểm / vùng đã xảy ra sạt lở (Landslide Inventory - Nhãn 1)
LANDSLIDE_SHP_PATH = os.path.join(BASE_DIR, "inventory", "landslides.shp")

# Đường dẫn file raster gộp (Stacked Multi-band Raster)
STACKED_RASTER_PATH = os.path.join(OUTPUT_DIR, "stacked_conditioning_factors.tif")

# File bản đồ đầu ra chỉ số nhạy cảm trượt lở (Landslide Susceptibility Index - LSI)
LSI_OUTPUT_RASTER_PATH = os.path.join(OUTPUT_DIR, "landslide_susceptibility_index.tif")
LSI_CLASSIFIED_RASTER_PATH = os.path.join(OUTPUT_DIR, "landslide_susceptibility_classified.tif")

# File lưu mô hình và bộ chuẩn hóa
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "lsm_random_forest_model.pkl")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "lsm_standard_scaler.pkl")


# ==============================================================================
# BƯỚC 1: ĐỌC, ĐỒNG BỘ (CRS, EXTENT, RESOLUTION) VÀ GỘP (STACK) CÁC RASTER NHÂN TỐ
# ==============================================================================
def align_and_stack_factors(factor_dict, reference_factor="Slope", output_path=STACKED_RASTER_PATH):
    """
    Đồng bộ tất cả raster nhân tố về cùng CRS, độ phân giải (pixel size) và phạm vi (extent)
    dựa trên một raster tham chiếu (mặc định là Slope), sau đó gộp thành 1 GeoTIFF đa kênh (Multi-band).
    """
    print("\n" + "="*70)
    print("BƯỚC 1: ĐỒNG BỘ VÀ GỘP (STACK) CÁC RASTER NHÂN TỐ TRƯỢT LỞ")
    print("="*70)

    # 1. Xác định raster tham chiếu (Reference Raster)
    if reference_factor not in factor_dict or not os.path.exists(factor_dict[reference_factor]):
        # Lấy file đầu tiên tồn tại làm tham chiếu
        existing = [k for k, v in factor_dict.items() if os.path.exists(v)]
        if not existing:
            raise FileNotFoundError("Không tìm thấy bất kỳ file raster nhân tố nào trong đường dẫn cấu hình!")
        reference_factor = existing[0]

    ref_path = factor_dict[reference_factor]
    print(f"[*] Sử dụng '{reference_factor}' làm raster tham chiếu chuẩn: {ref_path}")

    with rasterio.open(ref_path) as ref_src:
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_width = ref_src.width
        ref_height = ref_src.height
        ref_meta = ref_src.meta.copy()

    factor_names = list(factor_dict.keys())
    n_factors = len(factor_names)
    print(f"[*] Tổng số nhân tố cần gộp: {n_factors} ({', '.join(factor_names)})")
    print(f"    - CRS chuẩn: {ref_crs}")
    print(f"    - Kích thước pixel chuẩn: {ref_width} x {ref_height} (dx={ref_transform[0]:.2f}, dy={ref_transform[4]:.2f})")

    # Cập nhật metadata cho file ảnh gộp
    ref_meta.update({
        'count': n_factors,
        'dtype': 'float32',
        'nodata': -9999.0,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with rasterio.open(output_path, 'w', **ref_meta) as dst:
        for idx, name in enumerate(factor_names, start=1):
            fpath = factor_dict[name]
            print(f"[*] Đang xử lý kênh {idx}/{n_factors} [{name}] từ: {fpath}")

            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Không tìm thấy file nhân tố '{name}': {fpath}")

            # Đặt phương pháp nội suy: Nearest cho LULC (dữ liệu phân loại), Bilinear cho biến liên tục
            resample_alg = Resampling.nearest if name.upper() == 'LULC' else Resampling.bilinear

            destination_data = np.full((ref_height, ref_width), -9999.0, dtype=np.float32)

            with rasterio.open(fpath) as src:
                # Kiểm tra nếu đã hoàn toàn khớp CRS, Extent, Shape
                is_identical = (
                    src.crs == ref_crs and
                    src.width == ref_width and
                    src.height == ref_height and
                    src.transform == ref_transform
                )

                if is_identical:
                    data = src.read(1).astype(np.float32)
                    # Gán giá trị NoData chuẩn
                    if src.nodata is not None:
                        data[data == src.nodata] = -9999.0
                    destination_data = data
                else:
                    print(f"    -> Khác biệt không gian phát hiện! Tiến hành Reproject/Resample ({resample_alg.name})...")
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=destination_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src.nodata if src.nodata is not None else -9999.0,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        dst_nodata=-9999.0,
                        resampling=resample_alg
                    )

            # Xử lý các giá trị không hợp lệ (NaN/Inf) thành NoData
            destination_data[~np.isfinite(destination_data)] = -9999.0

            # Ghi vào kênh tương ứng của raster gộp
            dst.write(destination_data, idx)
            dst.set_band_description(idx, name)

    print(f"[+] Gộp thành công {n_factors} nhân tố vào file: {output_path}")
    return output_path, factor_names


# ==============================================================================
# BƯỚC 2: ĐỌC ĐIỂM SẠT LỞ (NHÃN 1) VÀ SINH ĐIỂM KHÔNG SẠT LỞ (NHÃN 0) Ở VÙNG DỐC THẤP
# ==============================================================================
def sample_landslide_and_non_landslide(
    landslide_shp_path,
    stacked_raster_path,
    factor_names,
    slope_factor_name="Slope",
    max_slope_deg=5.0,
    buffer_distance=200.0,
    random_state=42
):
    """
    Đọc các điểm sạt lở (Nhãn 1), tạo vùng đệm an toàn và sinh ngẫu nhiên số lượng điểm
    không sạt lở (Nhãn 0) tương đương ở những khu vực có độ dốc thấp (slope <= max_slope_deg).
    """
    print("\n" + "="*70)
    print("BƯỚC 2: CHUẨN BỊ MẪU HUẤN LUYỆN (SẠT LỞ = 1 & KHÔNG SẠT LỞ = 0)")
    print("="*70)

    # 1. Đọc điểm sạt lở (Nhãn 1)
    if not os.path.exists(landslide_shp_path):
        raise FileNotFoundError(f"Không tìm thấy file Shapefile sạt lở: {landslide_shp_path}")

    print(f"[*] Đọc dữ liệu sạt lở từ Shapefile: {landslide_shp_path}")
    ls_gdf = gpd.read_file(landslide_shp_path)

    with rasterio.open(stacked_raster_path) as src:
        target_crs = src.crs
        transform = src.transform
        width, height = src.width, src.height
        bounds = src.bounds

        # Đồng bộ CRS nếu khác nhau
        if ls_gdf.crs != target_crs:
            print(f"[*] Chuyển đổi hệ tọa độ Shapefile ({ls_gdf.crs}) -> Raster CRS ({target_crs})...")
            ls_gdf = ls_gdf.to_crs(target_crs)

        # Trích xuất tọa độ điểm (nếu là Polygon thì lấy tâm Centroid)
        ls_points = []
        for geom in ls_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type in ['Point', 'MultiPoint']:
                if geom.geom_type == 'MultiPoint':
                    for pt in geom.geoms:
                        ls_points.append(pt)
                else:
                    ls_points.append(geom)
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                ls_points.append(geom.centroid)

        n_landslides = len(ls_points)
        print(f"[+] Tìm thấy {n_landslides} điểm sạt lở thực tế (Nhãn 1).")

        # 2. Sinh điểm không sạt lở (Nhãn 0) ở vùng dốc thấp
        # Tìm chỉ số kênh của Slope trong file ảnh gộp
        slope_band_idx = factor_names.index(slope_factor_name) + 1
        print(f"[*] Đọc kênh độ dốc '{slope_factor_name}' (kênh {slope_band_idx}) để xác định vùng sạt lở tiềm năng cực thấp (<= {max_slope_deg}°)...")
        slope_data = src.read(slope_band_idx)

        # Tạo mặt nạ vùng sạt lở hiện có và vùng đệm an toàn (Buffer)
        ls_multipoint = gpd.GeoSeries(ls_points, crs=target_crs)
        buffered_zones = ls_multipoint.buffer(buffer_distance).unary_union

        # Lọc ra các pixel hợp lệ: Độ dốc <= max_slope_deg và không phải NoData (-9999)
        current_slope_threshold = float(max_slope_deg)
        valid_low_slope_mask = (slope_data != -9999.0) & (slope_data >= 0) & (slope_data <= current_slope_threshold) & np.isfinite(slope_data)
        low_slope_indices = np.argwhere(valid_low_slope_mask)

        # Cơ chế thích ứng tự động (Adaptive Slope Thresholding):
        # Nếu địa hình đồi núi dốc đứng có ít pixel <= 5 độ, tự động nới lỏng dần ngưỡng dốc an toàn
        while len(low_slope_indices) < n_landslides * 3 and current_slope_threshold < 30.0:
            current_slope_threshold += 2.0
            valid_low_slope_mask = (slope_data != -9999.0) & (slope_data >= 0) & (slope_data <= current_slope_threshold) & np.isfinite(slope_data)
            low_slope_indices = np.argwhere(valid_low_slope_mask)
            print(f"    -> Tự động điều chỉnh ngưỡng dốc an toàn lên <= {current_slope_threshold:.1f}° (tìm thấy {len(low_slope_indices)} pixel ứng viên)...")

        print(f"[*] Số lượng pixel vùng dốc thấp tiềm năng: {len(low_slope_indices):,}")

        np.random.seed(random_state)
        # Shuffle các pixel dốc thấp để lấy mẫu ngẫu nhiên
        np.random.shuffle(low_slope_indices)

        non_ls_points = []
        # Lấy dư ra 20% để bù trừ các điểm rơi vào vùng đệm buffer hoặc rìa NoData
        target_non_ls_count = n_landslides

        for row, col in low_slope_indices:
            x, y = rasterio.transform.xy(transform, row, col, offset='center')
            pt = Point(x, y)

            # Kiểm tra xem điểm này có nằm trong vùng đệm sạt lở hay không
            if buffered_zones is not None and buffered_zones.contains(pt):
                continue

            non_ls_points.append(pt)
            if len(non_ls_points) >= target_non_ls_count:
                break

        print(f"[+] Đã sinh ngẫu nhiên {len(non_ls_points)} điểm không sạt lở (Nhãn 0) ở độ dốc <= {max_slope_deg}° (cách sạt lở tối thiểu {buffer_distance}m).")

    # Tạo GeoDataFrame hợp nhất (Gồm cả điểm 1 và điểm 0)
    all_points = ls_points + non_ls_points
    all_labels = [1] * len(ls_points) + [0] * len(non_ls_points)

    sample_gdf = gpd.GeoDataFrame({
        'label': all_labels,
        'geometry': all_points
    }, crs=target_crs)

    return sample_gdf


# ==============================================================================
# BƯỚC 3: TRÍCH XUẤT GIÁ TRỊ ĐA TẦNG, LÀM SẠCH VÀ CÂN BẰNG MẪU (50:50)
# ==============================================================================
def extract_and_balance_samples(sample_gdf, stacked_raster_path, factor_names, random_state=42):
    """
    Trích xuất giá trị của tất cả các kênh nhân tố tại các điểm mẫu, loại bỏ các điểm NoData,
    và đảm bảo cân bằng chính xác 50% trượt lở (1) và 50% không trượt lở (0).
    """
    print("\n" + "="*70)
    print("BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG ĐA TẦNG & CÂN BẰNG MẪU (50:50)")
    print("="*70)

    coords = [(geom.x, geom.y) for geom in sample_gdf.geometry]

    with rasterio.open(stacked_raster_path) as src:
        nodata_val = src.nodata if src.nodata is not None else -9999.0
        print(f"[*] Đang lấy mẫu giá trị từ {len(coords)} điểm qua {len(factor_names)} kênh...")
        # src.sample trả về generator của các array kích thước (n_bands,)
        sampled_values = np.array(list(src.sample(coords)), dtype=np.float32)

    df_samples = pd.DataFrame(sampled_values, columns=factor_names)
    df_samples['label'] = sample_gdf['label'].values

    # Kiểm tra và làm sạch NoData, NaN, Inf
    print("[*] Đang kiểm tra và loại bỏ NoData, NaN, Inf...")
    invalid_mask = (df_samples[factor_names] == nodata_val).any(axis=1) | \
                   (~np.isfinite(df_samples[factor_names])).any(axis=1)

    df_clean = df_samples[~invalid_mask].copy()
    print(f"    - Tổng số mẫu ban đầu: {len(df_samples)}")
    print(f"    - Số mẫu bị loại bỏ (rơi vào NoData/Rìa bản đồ): {np.sum(invalid_mask)}")
    print(f"    - Số mẫu hợp lệ còn lại: {len(df_clean)}")

    # Cân bằng tỷ lệ 50:50 giữa 2 lớp
    df_pos = df_clean[df_clean['label'] == 1]
    df_neg = df_clean[df_clean['label'] == 0]

    min_count = min(len(df_pos), len(df_neg))
    print(f"[*] Số lượng mẫu sau làm sạch: Sạt lở = {len(df_pos)}, Không sạt lở = {len(df_neg)}")

    if min_count == 0:
        raise ValueError("Không đủ mẫu sau khi lọc NoData! Vui lòng kiểm tra lại phạm vi chồng lấn giữa điểm sạt lở và ảnh.")

    df_pos_balanced = df_pos.sample(n=min_count, random_state=random_state)
    df_neg_balanced = df_neg.sample(n=min_count, random_state=random_state)

    df_balanced = pd.concat([df_pos_balanced, df_neg_balanced]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    X = df_balanced[factor_names].values
    y = df_balanced['label'].values

    print(f"[+] Tập dữ liệu hoàn chỉnh sau cân bằng: {len(df_balanced)} mẫu (50% Sạt lở : 50% Không sạt lở).")
    return X, y, factor_names


# ==============================================================================
# BƯỚC 4: PHÂN CHIA TẬP DỮ LIỆU (70/15/15) VÀ CHUẨN HÓA STANDARDSCALER
# ==============================================================================
def split_and_scale_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Phân chia Train / Validation / Test theo tỷ lệ 70% / 15% / 15% (Stratified)
    Chuẩn hóa StandardScaler: fit trên tập Train, sau đó transform cho Val, Test và toàn bộ Raster.
    """
    print("\n" + "="*70)
    print("BƯỚC 4: PHÂN CHIA TẬP DỮ LIỆU (70/15/15) & CHUẨN HÓA DỮ LIỆU")
    print("="*70)

    # 1. Tách tập Train (70%) và tập tạm Temp (30% gồm Val + Test)
    temp_ratio = val_size + test_size  # 0.30
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_ratio, random_state=random_state, stratify=y
    )

    # 2. Tách tập Temp thành Validation (15%) và Test (15%)
    relative_test_ratio = test_size / temp_ratio  # 0.15 / 0.30 = 0.50
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"[*] Phân bố mẫu sau khi chia:")
    print(f"    - Tập Huấn luyện (Train):   {len(X_train):>5} mẫu ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    - Tập Đánh giá (Val):       {len(X_val):>5} mẫu ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    - Tập Kiểm thử (Test):      {len(X_test):>5} mẫu ({len(X_test)/len(X)*100:.1f}%)")

    # 3. Chuẩn hóa đặc trưng bằng StandardScaler (Tránh Data Leakage)
    print("[*] Áp dụng StandardScaler (Fit chỉ trên tập Train để tránh rò rỉ dữ liệu)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Lưu scaler để tái sử dụng trong suy luận không gian
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"[+] Đã lưu bộ chuẩn hóa StandardScaler tại: {SCALER_SAVE_PATH}")

    data_splits = {
        'X_train': X_train_scaled, 'y_train': y_train,
        'X_val': X_val_scaled, 'y_val': y_val,
        'X_test': X_test_scaled, 'y_test': y_test,
        'scaler': scaler
    }
    return data_splits


# ==============================================================================
# BƯỚC 5: HUẤN LUYỆN RANDOM FOREST (n_estimators=200) & FEATURE IMPORTANCE
# ==============================================================================
def train_random_forest_lsm(data_splits, factor_names, n_estimators=200, random_state=42):
    """
    Huấn luyện Random Forest với 200 cây quyết định, trích xuất và vẽ đồ thị Feature Importance.
    """
    print("\n" + "="*70)
    print(f"BƯỚC 5: HUẤN LUYỆN RANDOM FOREST ({n_estimators} CÂY) & ĐỘ QUAN TRỌNG NHÂN TỐ")
    print("="*70)

    X_train = data_splits['X_train']
    y_train = data_splits['y_train']

    rf_model = RandomForestClassifier(
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

    print(f"[*] Đang huấn luyện Random Forest trên {len(X_train)} mẫu...")
    rf_model.fit(X_train, y_train)

    if hasattr(rf_model, "oob_score_"):
        print(f"[+] Out-Of-Bag (OOB) Score: {rf_model.oob_score_ * 100:.2f}%")

    # Lưu mô hình
    joblib.dump(rf_model, MODEL_SAVE_PATH)
    print(f"[+] Đã lưu mô hình huấn luyện tại: {MODEL_SAVE_PATH}")

    # Trích xuất Feature Importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n" + "-"*50)
    print("BẢNG XẾP HẠNG ĐỘ QUAN TRỌNG CỦA CÁC NHÂN TỐ TRƯỢT LỞ:")
    print("-"*50)
    for rank, idx in enumerate(indices, start=1):
        print(f"  {rank:>2}. {factor_names[idx]:<15}: {importances[idx]*100:>6.2f}%")

    # Vẽ biểu đồ Feature Importance
    plot_feature_importance(factor_names, importances, indices)

    return rf_model


def plot_feature_importance(factor_names, importances, indices, output_path=os.path.join(OUTPUT_DIR, "feature_importance.png")):
    """Vẽ và lưu biểu đồ thanh Feature Importance dạng đồ họa đẹp mắt."""
    sorted_names = [factor_names[i] for i in indices][::-1]
    sorted_importances = [importances[i] * 100 for i in indices][::-1]

    plt.figure(figsize=(9, 5), dpi=300)
    bars = plt.barh(sorted_names, sorted_importances, color='#2c7bb6', edgecolor='#1a4c73', height=0.6)
    
    # Ghi giá trị phần trăm trên từng thanh
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                 ha='left', va='center', fontsize=9, fontweight='bold', color='#111111')

    plt.title("Độ quan trọng của các nhân tố sạt lở đất (Random Forest Feature Importance)", fontsize=11, fontweight='bold', pad=15)
    plt.xlabel("Độ quan trọng đóng góp (%)", fontsize=10)
    plt.xlim(0, max(sorted_importances) + 5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Đã xuất biểu đồ Feature Importance tại: {output_path}")


# ==============================================================================
# BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH TOÀN DIỆN (ACCURACY, F1, KAPPA, CONFUSION MATRIX, ROC/AUC)
# ==============================================================================
def evaluate_lsm_model(model, data_splits):
    """
    Đánh giá mô hình trên tập Validation và tập Test độc lập:
    - Accuracy, Precision, Recall, F1-Score
    - Cohen's Kappa Score
    - Confusion Matrix
    - ROC Curve & AUC Score
    """
    print("\n" + "="*70)
    print("BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION VÀ TẬP TEST ĐỘC LẬP")
    print("="*70)

    for set_name, X_set, y_set in [
        ("VALIDATION", data_splits['X_val'], data_splits['y_val']),
        ("TEST (ĐỘC LẬP)", data_splits['X_test'], data_splits['y_test'])
    ]:
        y_pred = model.predict(X_set)
        y_prob = model.predict_proba(X_set)[:, 1]

        acc = accuracy_score(y_set, y_pred)
        prec = precision_score(y_set, y_pred)
        rec = recall_score(y_set, y_pred)
        f1 = f1_score(y_set, y_pred)
        kappa = cohen_kappa_score(y_set, y_pred)
        auc = roc_auc_score(y_set, y_prob)
        cm = confusion_matrix(y_set, y_pred)

        print("\n" + "="*50)
        print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP {set_name}")
        print("="*50)
        print(f"  - Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%")
        print(f"  - Precision:                        {prec * 100:.2f}%")
        print(f"  - Recall:                           {rec * 100:.2f}%")
        print(f"  - F1-Score:                         {f1:.4f}")
        print(f"  - Hệ số Cohen's Kappa:              {kappa:.4f}")
        print(f"  - Diện tích dưới đường cong (AUC):  {auc:.4f}")
        print("\nMa trận nhầm lẫn (Confusion Matrix):")
        print(f"  [[TN={cm[0,0]:<4} FP={cm[0,1]:<4}]")
        print(f"   [FN={cm[1,0]:<4} TP={cm[1,1]:<4}]]")
        print("\nBáo cáo phân loại chi tiết:")
        print(classification_report(y_set, y_pred, target_names=["Không sạt lở (0)", "Sạt lở (1)"]))

    # Vẽ ROC Curve & Confusion Matrix cho tập Test
    plot_evaluation_charts(model, data_splits['X_test'], data_splits['y_test'])


def plot_evaluation_charts(model, X_test, y_test, output_path=os.path.join(OUTPUT_DIR, "model_evaluation_metrics.png")):
    """Vẽ biểu đồ ROC Curve và Ma trận nhầm lẫn (Confusion Matrix) trên cùng một hình ảnh."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Vẽ ROC Curve
    ax_roc = axes[0]
    ax_roc.plot(fpr, tpr, color='#d7191c', lw=2.5, label=f'Random Forest (AUC = {auc_val:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label='Ngẫu nhiên (AUC = 0.50)')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Tỷ lệ Dương tính giả (False Positive Rate)', fontsize=9)
    ax_roc.set_ylabel('Tỷ lệ Dương tính thật (True Positive Rate)', fontsize=9)
    ax_roc.set_title('Đường cong ROC (Receiver Operating Characteristic)', fontsize=10, fontweight='bold')
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, linestyle=':', alpha=0.6)

    # 2. Vẽ Confusion Matrix Heatmap
    ax_cm = axes[1]
    cax = ax_cm.matshow(cm, cmap='Blues', alpha=0.85)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax_cm.text(j, i, str(val), ha='center', va='center', fontsize=14, fontweight='bold',
                       color='white' if val > cm.max()/2 else 'black')
    
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Không sạt lở (0)', 'Sạt lở (1)'], fontsize=9)
    ax_cm.set_yticklabels(['Không sạt lở (0)', 'Sạt lở (1)'], fontsize=9)
    ax_cm.set_xlabel('Nhãn Dự đoán (Predicted)', fontsize=9, labelpad=10)
    ax_cm.set_ylabel('Nhãn Thực tế (Actual)', fontsize=9)
    ax_cm.set_title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=10, fontweight='bold', pad=15)
    fig.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Đã xuất biểu đồ đánh giá (ROC Curve & Confusion Matrix) tại: {output_path}")


# ==============================================================================
# BƯỚC 7: DỰ ĐOÁN XÁC SUẤT THEO KHỐI (BLOCK-WISE) VÀ XUẤT BẢN ĐỒ GEOTIFF (LSI: 0 - 1)
# ==============================================================================
def predict_lsm_raster(stacked_raster_path, model, scaler, output_lsi_path=LSI_OUTPUT_RASTER_PATH, block_size=1024):
    """
    Dự đoán xác suất trượt lở (predict_proba[:, 1]) cho toàn vùng nghiên cứu theo từng cửa sổ (Block)
    để tránh quá tải bộ nhớ RAM (OOM), xuất ra GeoTIFF chỉ số nhạy cảm (LSI) có giá trị từ 0.0 đến 1.0.
    """
    print("\n" + "="*70)
    print("BƯỚC 7: DỰ ĐOÁN TOÀN VÙNG (BLOCK-WISE PREDICTION) VÀ XUẤT BẢN ĐỒ LSI GEOTIFF")
    print("="*70)

    print(f"[*] File ảnh đầu vào:  {stacked_raster_path}")
    print(f"[*] File bản đồ xuất:  {output_lsi_path}")
    print(f"[*] Kích thước khối (Block size): {block_size} x {block_size} pixels")

    with rasterio.open(stacked_raster_path) as src:
        meta = src.meta.copy()
        height = src.height
        width = src.width
        n_bands = src.count
        nodata_val = src.nodata if src.nodata is not None else -9999.0

        # Cấu hình metadata cho file bản đồ LSI đầu ra (Float32, giá trị 0.0 -> 1.0)
        meta.update({
            'count': 1,
            'dtype': 'float32',
            'nodata': -9999.0,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        })

        total_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)
        block_counter = 0

        with rasterio.open(output_lsi_path, 'w', **meta) as dst:
            dst.set_band_description(1, "Landslide_Susceptibility_Index")

            for row in range(0, height, block_size):
                h_chunk = min(block_size, height - row)

                for col in range(0, width, block_size):
                    w_chunk = min(block_size, width - col)
                    block_counter += 1

                    window = Window(col, row, w_chunk, h_chunk)

                    # Đọc dữ liệu khối: (bands, h_chunk, w_chunk)
                    block_data = src.read(window=window)

                    # Chuyển đổi về dạng bảng 2D: (số pixel, số đặc trưng)
                    # block_data: shape (n_bands, h_chunk, w_chunk) -> transpose -> reshape
                    reshaped = block_data.reshape(n_bands, -1).T

                    # Tìm các pixel hợp lệ (không phải NoData, NaN, Inf ở bất kỳ kênh nào)
                    valid_mask = (reshaped != nodata_val).all(axis=1) & np.isfinite(reshaped).all(axis=1)

                    # Tạo mảng kết quả mặc định là NoData
                    lsi_block = np.full(reshaped.shape[0], -9999.0, dtype=np.float32)

                    if np.any(valid_mask):
                        valid_pixels = reshaped[valid_mask]

                        # QUAN TRỌNG: Chuẩn hóa pixel hợp lệ bằng StandardScaler đã huấn luyện
                        valid_pixels_scaled = scaler.transform(valid_pixels)

                        # Dự đoán xác suất thuộc lớp 1 (Sạt lở đất)
                        prob_landslide = model.predict_proba(valid_pixels_scaled)[:, 1]
                        lsi_block[valid_mask] = prob_landslide.astype(np.float32)

                    # Đưa về dạng 2D ảnh và ghi vào file kết quả
                    lsi_block_2d = lsi_block.reshape((1, h_chunk, w_chunk))
                    dst.write(lsi_block_2d, window=window)

                    if block_counter % 10 == 0 or block_counter == total_blocks:
                        print(f"    - Tiến độ dự đoán: {block_counter}/{total_blocks} khối ({block_counter/total_blocks*100:.1f}%)")

    print(f"[+] Đã xuất thành công bản đồ Chỉ số Nhạy cảm Sạt lở đất (LSI) tại: {output_lsi_path}")

    # Phân cấp và tạo bản đồ màu trực quan chuyên nghiệp
    classify_and_export_map(output_lsi_path, LSI_CLASSIFIED_RASTER_PATH)


# ==============================================================================
# HÀM BỔ TRỢ: PHÂN CẤP ĐỘ NHẠY CẢM VÀ XUẤT BẢN ĐỒ MÀU (PNG + .CLR)
# ==============================================================================
def classify_and_export_map(lsi_raster_path, classified_raster_path):
    """
    Phân loại chỉ số LSI thành 5 cấp độ chuẩn theo quy chuẩn quốc tế:
      1: Rất thấp (Very Low)   [0.0 - 0.2] -> Xanh lá đậm  [56, 168, 0]
      2: Thấp (Low)            [0.2 - 0.4] -> Xanh lá nhạt [139, 209, 0]
      3: Trung bình (Moderate) [0.4 - 0.6] -> Vàng         [255, 255, 0]
      4: Cao (High)            [0.6 - 0.8] -> Cam          [255, 128, 0]
      5: Rất cao (Very High)   [0.8 - 1.0] -> Đỏ           [255, 0, 0]
    """
    print("\n[*] Đang phân cấp độ nhạy cảm sạt lở đất thành 5 mức độ...")

    color_dict = {
        1: ([56, 168, 0],   "Rất thấp (0.0 - 0.2)"),
        2: ([139, 209, 0],  "Thấp (0.2 - 0.4)"),
        3: ([255, 255, 0],  "Trung bình (0.4 - 0.6)"),
        4: ([255, 128, 0],  "Cao (0.6 - 0.8)"),
        5: ([255, 0, 0],    "Rất cao (0.8 - 1.0)")
    }

    with rasterio.open(lsi_raster_path) as src:
        lsi_data = src.read(1)
        meta = src.meta.copy()

    classified_data = np.full(lsi_data.shape, 255, dtype=np.uint8)  # 255 là NoData cho Byte

    valid_mask = (lsi_data != -9999.0) & np.isfinite(lsi_data) & (lsi_data >= 0.0) & (lsi_data <= 1.0)

    # Phân 5 cấp độ
    classified_data[valid_mask & (lsi_data < 0.2)] = 1
    classified_data[valid_mask & (lsi_data >= 0.2) & (lsi_data < 0.4)] = 2
    classified_data[valid_mask & (lsi_data >= 0.4) & (lsi_data < 0.6)] = 3
    classified_data[valid_mask & (lsi_data >= 0.6) & (lsi_data < 0.8)] = 4
    classified_data[valid_mask & (lsi_data >= 0.8)] = 5

    meta.update({
        'dtype': 'uint8',
        'nodata': 255,
        'count': 1,
        'compress': 'lzw'
    })

    with rasterio.open(classified_raster_path, 'w', **meta) as dst:
        dst.write(classified_data, 1)
        dst.set_band_description(1, "Landslide_Susceptibility_Classified")

    # Tạo file bảng màu ArcMap (.clr)
    clr_path = os.path.splitext(classified_raster_path)[0] + ".clr"
    with open(clr_path, "w", encoding="utf-8") as f:
        for val, (rgb, _) in color_dict.items():
            f.write(f"{val} {rgb[0]} {rgb[1]} {rgb[2]}\n")
    print(f"[+] Đã tạo file cấu hình màu cho ArcMap/QGIS (.clr) tại: {clr_path}")

    # Xuất ảnh PNG trực quan
    png_path = os.path.splitext(classified_raster_path)[0] + "_preview.png"
    h, w = classified_data.shape
    rgb_img = np.full((h, w, 3), 255, dtype=np.uint8)  # Nền trắng cho vùng NoData

    for val, (rgb, _) in color_dict.items():
        rgb_img[classified_data == val] = rgb

    from PIL import Image
    im = Image.fromarray(rgb_img)
    # Tự động scale vừa khung hình xem
    scale = max(1, 1000 // max(h, w))
    if scale > 1:
        im = im.resize((w * scale, h * scale), Image.NEAREST)
    im.save(png_path)
    print(f"[+] Đã xuất file ảnh màu trực quan kết quả tại: {png_path}")

    # Thống kê diện tích từng cấp
    total_valid = np.sum(valid_mask)
    if total_valid > 0:
        print("\n" + "-"*50)
        print("THỐNG KÊ PHÂN BỐ TỶ LỆ DIỆN TÍCH TỪNG CẤP ĐỘ NHẠY CẢM:")
        print("-"*50)
        for val, (_, name) in color_dict.items():
            cnt = np.sum(classified_data == val)
            pct = (cnt / total_valid) * 100
            print(f"  Cấp {val} [{name:<22}]: {cnt:>8} pixels ({pct:>5.2f}%)")


# ==============================================================================
# HÀM TẠO DỮ LIỆU THỬ NGHIỆM GIẢ LẬP (NẾU CHƯA CÓ SẴN 8 FILE RASTER & SHAPEFILE)
# ==============================================================================
def generate_synthetic_demo_data(base_test_dir):
    """
    Tự động sinh dữ liệu địa hình và điểm sạt lở giả lập để kiểm tra toàn bộ pipeline
    chạy thông suốt ngay lập tức trên máy nếu người dùng chưa kịp copy dữ liệu thực tế.
    """
    print("\n" + "="*70)
    print("CHẾ ĐỘ TỰ ĐỘNG SINH DỮ LIỆU THỬ NGHIỆM (SYNTHETIC BENCHMARK DATA)")
    print("="*70)
    factor_dir = os.path.join(base_test_dir, "demo_factors")
    inventory_dir = os.path.join(base_test_dir, "demo_inventory")
    os.makedirs(factor_dir, exist_ok=True)
    os.makedirs(inventory_dir, exist_ok=True)

    height, width = 400, 400
    transform = rasterio.transform.from_origin(580000.0, 2350000.0, 30.0, 30.0) # UTM 48N
    crs = "EPSG:32648"

    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    xx, yy = np.meshgrid(x, y)

    # Sinh địa hình đồi núi thực tế bằng sóng sin/cos
    elevation = 200 + 400 * (np.sin(xx/2) * np.cos(yy/2) + 0.5 * np.sin(xx + yy))
    gy, gx = np.gradient(elevation)
    slope = np.clip(np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))), 0, 75)
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360
    curvature = np.clip(np.gradient(gx)[1] + np.gradient(gy)[0], -5, 5)
    twi = np.clip(np.log(np.abs(elevation) + 1.0) / (np.tan(np.radians(slope + 1.0))), 2, 18)
    ndvi = np.clip(0.3 + 0.5 * np.sin(xx/3), -0.2, 0.9)
    lulc = np.random.choice([1, 2, 3, 4, 5], size=(height, width), p=[0.4, 0.25, 0.15, 0.1, 0.1]).astype(np.float32)
    dist_rivers = np.abs(xx - 5) * 500
    dist_roads = np.abs(yy - 3) * 400

    factors_data = {
        "Slope": slope.astype(np.float32),
        "Aspect": aspect.astype(np.float32),
        "Curvature": curvature.astype(np.float32),
        "TWI": twi.astype(np.float32),
        "NDVI": ndvi.astype(np.float32),
        "LULC": lulc.astype(np.float32),
        "Dist_Rivers": dist_rivers.astype(np.float32),
        "Dist_Roads": dist_roads.astype(np.float32)
    }

    demo_paths = {}
    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': -9999.0,
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }

    for name, data in factors_data.items():
        fpath = os.path.join(factor_dir, f"{name.lower()}.tif")
        with rasterio.open(fpath, 'w', **meta) as dst:
            dst.write(data, 1)
        demo_paths[name] = fpath

    # Sinh các điểm sạt lở tập trung ở sườn dốc cao (> 25 độ)
    high_slope_idx = np.argwhere(slope > 25)
    np.random.seed(42)
    selected_idx = high_slope_idx[np.random.choice(len(high_slope_idx), 120, replace=False)]

    ls_pts = [Point(rasterio.transform.xy(transform, r, c)) for r, c in selected_idx]
    demo_shp_path = os.path.join(inventory_dir, "landslides_demo.shp")
    gdf_demo = gpd.GeoDataFrame({'id': range(1, len(ls_pts)+1), 'label': [1]*len(ls_pts), 'geometry': ls_pts}, crs=crs)
    gdf_demo.to_file(demo_shp_path)

    print(f"[+] Đã tạo 8 raster nhân tố giả lập và {len(ls_pts)} điểm sạt lở mẫu tại: {base_test_dir}")
    return demo_paths, demo_shp_path


# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH (PIPELINE EXECUTION)
# ==============================================================================
def main():
    print("="*75)
    print("  HỆ THỐNG ĐÁNH GIÁ ĐỘ NHẠY CẢM SẠT LỞ ĐẤT (LANDSLIDE SUSCEPTIBILITY)")
    print("  Ứng dụng Học máy Random Forest & Xử lý dữ liệu không gian Raster/Vector")
    print("="*75)

    # Kiểm tra xem dữ liệu thực tế của người dùng đã sẵn sàng hay chưa
    real_factors_exist = all(os.path.exists(p) for p in FACTOR_RASTERS.values())
    real_shp_exist = os.path.exists(LANDSLIDE_SHP_PATH)

    if real_factors_exist and real_shp_exist:
        print("[*] Phát hiện đầy đủ 8 file raster nhân tố và Shapefile sạt lở thực tế.")
        current_factors = FACTOR_RASTERS
        current_shp = LANDSLIDE_SHP_PATH
    else:
        print("[!] Chưa tìm thấy đủ file dữ liệu thực tế tại cấu hình mặc định.")
        print("[*] Đang khởi tạo dữ liệu mô phỏng chuẩn (Benchmark Demo) để kiểm tra toàn bộ pipeline...")
        demo_dir = os.path.join(OUTPUT_DIR, "demo_workspace")
        current_factors, current_shp = generate_synthetic_demo_data(demo_dir)

    # BƯỚC 1: ĐỒNG BỘ VÀ GỘP (STACK) CÁC RASTER NHÂN TỐ
    stacked_raster, factor_names = align_and_stack_factors(
        factor_dict=current_factors,
        reference_factor="Slope",
        output_path=STACKED_RASTER_PATH
    )

    # BƯỚC 2: CHUẨN BỊ MẪU (SẠT LỞ = 1 & TỰ ĐỘNG SINH KHÔNG SẠT LỞ = 0 Ở VÙNG DỐC THẤP)
    sample_gdf = sample_landslide_and_non_landslide(
        landslide_shp_path=current_shp,
        stacked_raster_path=stacked_raster,
        factor_names=factor_names,
        slope_factor_name="Slope",
        max_slope_deg=5.0,     # Ngưỡng dốc thấp an toàn: <= 5 độ
        buffer_distance=200.0, # Cách xa điểm sạt lở tối thiểu 200m
        random_state=42
    )

    # BƯỚC 3: TRÍCH XUẤT GIÁ TRỊ ĐA TẦNG, LÀM SẠCH VÀ CÂN BẰNG MẪU (50:50)
    X, y, feature_names = extract_and_balance_samples(
        sample_gdf=sample_gdf,
        stacked_raster_path=stacked_raster,
        factor_names=factor_names,
        random_state=42
    )

    # BƯỚC 4: PHÂN CHIA TRAIN/VAL/TEST (70/15/15) VÀ CHUẨN HÓA STANDARDSCALER
    data_splits = split_and_scale_data(
        X=X,
        y=y,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    # BƯỚC 5: HUẤN LUYỆN RANDOM FOREST (n_estimators=200) & TRÍCH XUẤT FEATURE IMPORTANCE
    rf_model = train_random_forest_lsm(
        data_splits=data_splits,
        factor_names=feature_names,
        n_estimators=200,
        random_state=42
    )

    # BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH TOÀN DIỆN (ACCURACY, F1, KAPPA, CONFUSION MATRIX, ROC/AUC)
    evaluate_lsm_model(
        model=rf_model,
        data_splits=data_splits
    )

    # BƯỚC 7: DỰ ĐOÁN XÁC SUẤT THEO KHỐI (BLOCK-WISE) VÀ XUẤT FILE GEOTIFF LSI (0 -> 1)
    predict_lsm_raster(
        stacked_raster_path=stacked_raster,
        model=rf_model,
        scaler=data_splits['scaler'],
        output_lsi_path=LSI_OUTPUT_RASTER_PATH,
        block_size=1024
    )

    print("\n" + "="*75)
    print("HOÀN TẤT TOÀN BỘ QUY TRÌNH ĐÁNH GIÁ ĐỘ NHẠY CẢM SẠT LỞ ĐẤT (LSM)!")
    print(f"Các tệp kết quả đã được lưu tại thư mục: {OUTPUT_DIR}")
    print("="*75)


if __name__ == "__main__":
    main()
