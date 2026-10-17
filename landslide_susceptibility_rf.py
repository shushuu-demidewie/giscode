# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH ĐÁNH GIÁ ĐỘ NHẠY CẢM SẠT LỞ ĐẤT (LANDSLIDE SUSCEPTIBILITY MAPPING - LSM)
SO SÁNH ĐỐI ĐẦU 2 THUẬT TOÁN HỌC MÁY: RANDOM FOREST (RF) & SUPPORT VECTOR MACHINE (SVM)
================================================================================

Đề tài: Nghiên cứu ứng dụng học máy/học sâu và ảnh viễn thám Landsat 8 
thành lập bản đồ / WebGIS khoanh vùng nguy cơ trượt lở.

Các tính năng cốt lõi:
1. Đồng bộ (Reproject/Resample) và gộp (Stack) 8 nhân tố trượt lở (Conditioning Factors):
   - Địa hình: Slope, Aspect, Curvature, TWI
   - Môi trường & Lớp phủ: NDVI, LULC (trích xuất hoặc kết hợp từ ảnh Landsat 8)
   - Nhân tố nhân sinh / Thủy văn: Distance to rivers, Distance to roads
2. Đọc Shapefile điểm sạt lở (Nhãn 1) và tự động sinh mẫu ngẫu nhiên không sạt lở (Nhãn 0)
   trong vùng dốc thấp có kiểm soát khoảng cách an toàn.
3. Trích xuất giá trị đa tầng (Multi-band extraction), làm sạch NoData/NaN/Inf và cân bằng mẫu (50:50).
4. Phân chia Train / Val / Test (70 / 15 / 15) và chuẩn hóa bằng StandardScaler (chống rò rỉ dữ liệu).
5. Huấn luyện SONG SONG 2 thuật toán:
   - Random Forest (RF): n_estimators=200, tiêu chí Gini, max_features='sqrt'
   - Support Vector Machine (SVM): RBF Kernel, C=10.0, probability=True
6. So sánh đối đầu định lượng: Accuracy, Precision, Recall, F1-Score, Cohen's Kappa, ROC-AUC.
7. Trực quan hóa so sánh:
   - Đồ thị đường cong ROC kép (Dual ROC Curves: RF vs SVM)
   - Độ quan trọng nhân tố (Feature Importance của RF & Permutation Importance của SVM)
   - Ma trận nhầm lẫn (Confusion Matrix) của 2 mô hình
8. Dự đoán theo từng khối cửa sổ (Block-wise predict_proba) chống tràn RAM cho CẢ 2 THUẬT TOÁN,
   xuất ra GeoTIFF Chỉ số Nhạy cảm Sạt lở đất (LSI: 0 -> 1) và bản đồ phân 5 cấp màu chuẩn quốc tế.
9. Bảng phân tích so sánh tỷ lệ diện tích phân bố giữa RF và SVM.
"""

import os
import sys

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

# Cho phép GDAL tự động phục hồi nếu thiếu file .shx
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
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
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU ĐẦU VÀO VÀ ĐẦU RA
# ==============================================================================
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

LANDSLIDE_SHP_PATH = os.path.join(BASE_DIR, "inventory", "landslides.shp")

# Đường dẫn raster gộp (Stacked Multi-band Raster)
STACKED_RASTER_PATH = os.path.join(OUTPUT_DIR, "stacked_conditioning_factors.tif")

# File bản đồ đầu ra chỉ số nhạy cảm (LSI: 0 -> 1)
LSI_RF_PATH = os.path.join(OUTPUT_DIR, "landslide_susceptibility_rf_index.tif")
LSI_SVM_PATH = os.path.join(OUTPUT_DIR, "landslide_susceptibility_svm_index.tif")

# File bản đồ đầu ra phân 5 cấp nguy cơ
CLASSIFIED_RF_PATH = os.path.join(OUTPUT_DIR, "landslide_susceptibility_rf_classified.tif")
CLASSIFIED_SVM_PATH = os.path.join(OUTPUT_DIR, "landslide_susceptibility_svm_classified.tif")

# File lưu mô hình và bộ chuẩn hóa
RF_MODEL_PATH = os.path.join(OUTPUT_DIR, "lsm_random_forest_model.pkl")
SVM_MODEL_PATH = os.path.join(OUTPUT_DIR, "lsm_svm_model.pkl")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "lsm_standard_scaler.pkl")


# ==============================================================================
# BƯỚC 1: ĐỒNG BỘ VÀ GỘP CÁC RASTER NHÂN TỐ
# ==============================================================================
def align_and_stack_factors(factor_dict, reference_factor="Slope", output_path=STACKED_RASTER_PATH):
    """
    Đồng bộ tất cả raster nhân tố về cùng CRS, độ phân giải và extent,
    sau đó gộp thành 1 GeoTIFF đa kênh (Multi-band).
    """
    print("\n" + "="*70)
    print("BƯỚC 1: ĐỒNG BỘ VÀ GỘP (STACK) CÁC RASTER NHÂN TỐ TRƯỢT LỞ")
    print("="*70)

    if reference_factor not in factor_dict or not os.path.exists(factor_dict[reference_factor]):
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

            resample_alg = Resampling.nearest if name.upper() == 'LULC' else Resampling.bilinear
            destination_data = np.full((ref_height, ref_width), -9999.0, dtype=np.float32)

            with rasterio.open(fpath) as src:
                is_identical = (
                    src.crs == ref_crs and
                    src.width == ref_width and
                    src.height == ref_height and
                    src.transform == ref_transform
                )

                if is_identical:
                    data = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        data[data == src.nodata] = -9999.0
                    destination_data = data
                else:
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

            destination_data[~np.isfinite(destination_data)] = -9999.0
            dst.write(destination_data, idx)
            dst.set_band_description(idx, name)

    print(f"[+] Gộp thành công {n_factors} nhân tố vào file: {output_path}")
    return output_path, factor_names


# ==============================================================================
# BƯỚC 2: CHUẨN BỊ MẪU SẠT LỞ (1) VÀ KHÔNG SẠT LỞ (0)
# ==============================================================================
def sample_landslide_and_non_landslide(
    landslide_shp_path,
    stacked_raster_path,
    factor_names,
    slope_factor_name="Slope",
    max_slope_deg=10.0,
    buffer_distance=200.0,
    random_state=42
):
    """
    Đọc điểm sạt lở (1), tạo vùng đệm an toàn và sinh ngẫu nhiên số lượng điểm
    không sạt lở (0) tương đương ở những khu vực dốc thấp và cách xa điểm sạt lở.
    """
    print("\n" + "="*70)
    print("BƯỚC 2: CHUẨN BỊ MẪU HUẤN LUYỆN (SẠT LỞ = 1 & KHÔNG SẠT LỞ = 0)")
    print("="*70)

    if not os.path.exists(landslide_shp_path):
        raise FileNotFoundError(f"Không tìm thấy file Shapefile sạt lở: {landslide_shp_path}")

    print(f"[*] Đọc dữ liệu sạt lở từ Shapefile: {landslide_shp_path}")
    ls_gdf = gpd.read_file(landslide_shp_path)

    with rasterio.open(stacked_raster_path) as src:
        target_crs = src.crs
        transform = src.transform
        width, height = src.width, src.height

        if ls_gdf.crs != target_crs:
            print(f"[*] Chuyển đổi hệ tọa độ Shapefile ({ls_gdf.crs}) -> Raster CRS ({target_crs})...")
            ls_gdf = ls_gdf.to_crs(target_crs)

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

        slope_band_idx = factor_names.index(slope_factor_name) + 1
        slope_data = src.read(slope_band_idx)

        ls_multipoint = gpd.GeoSeries(ls_points, crs=target_crs)
        buffered_zones = ls_multipoint.buffer(buffer_distance).unary_union

        # Lọc pixel dốc thấp hợp lý
        current_slope_threshold = float(max_slope_deg)
        valid_low_slope_mask = (slope_data != -9999.0) & (slope_data >= 0) & (slope_data <= current_slope_threshold) & np.isfinite(slope_data)
        low_slope_indices = np.argwhere(valid_low_slope_mask)

        # Cơ chế thích ứng tự động (Adaptive Slope Thresholding)
        while len(low_slope_indices) < n_landslides * 2 and current_slope_threshold < 20.0:
            current_slope_threshold += 1.0
            valid_low_slope_mask = (slope_data != -9999.0) & (slope_data >= 0) & (slope_data <= current_slope_threshold) & np.isfinite(slope_data)
            low_slope_indices = np.argwhere(valid_low_slope_mask)

        print(f"[*] Số lượng pixel vùng dốc thấp tiềm năng (<= {current_slope_threshold:.1f}°): {len(low_slope_indices):,}")

        np.random.seed(random_state)
        np.random.shuffle(low_slope_indices)

        non_ls_points = []
        target_non_ls_count = n_landslides

        for row, col in low_slope_indices:
            x, y = rasterio.transform.xy(transform, row, col, offset='center')
            pt = Point(x, y)
            if buffered_zones is not None and buffered_zones.contains(pt):
                continue
            non_ls_points.append(pt)
            if len(non_ls_points) >= target_non_ls_count:
                break

        print(f"[+] Đã sinh ngẫu nhiên {len(non_ls_points)} điểm không sạt lở (Nhãn 0).")

    all_points = ls_points + non_ls_points
    all_labels = [1] * len(ls_points) + [0] * len(non_ls_points)

    sample_gdf = gpd.GeoDataFrame({
        'label': all_labels,
        'geometry': all_points
    }, crs=target_crs)

    return sample_gdf


# ==============================================================================
# BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG ĐA TẦNG VÀ CÂN BẰNG MẪU (50:50)
# ==============================================================================
def extract_and_balance_samples(sample_gdf, stacked_raster_path, factor_names, random_state=42):
    """Trích xuất giá trị 8 kênh nhân tố tại các điểm mẫu và cân bằng 50:50."""
    print("\n" + "="*70)
    print("BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG ĐA TẦNG & CÂN BẰNG MẪU (50:50)")
    print("="*70)

    coords = [(geom.x, geom.y) for geom in sample_gdf.geometry]

    with rasterio.open(stacked_raster_path) as src:
        nodata_val = src.nodata if src.nodata is not None else -9999.0
        print(f"[*] Đang lấy mẫu giá trị từ {len(coords)} điểm qua {len(factor_names)} kênh...")
        sampled_values = np.array(list(src.sample(coords)), dtype=np.float32)

    df_samples = pd.DataFrame(sampled_values, columns=factor_names)
    df_samples['label'] = sample_gdf['label'].values

    invalid_mask = (df_samples[factor_names] == nodata_val).any(axis=1) | \
                   (~np.isfinite(df_samples[factor_names])).any(axis=1)

    df_clean = df_samples[~invalid_mask].copy()
    print(f"    - Tổng số mẫu ban đầu: {len(df_samples)}")
    print(f"    - Số mẫu hợp lệ còn lại: {len(df_clean)}")

    df_pos = df_clean[df_clean['label'] == 1]
    df_neg = df_clean[df_clean['label'] == 0]

    min_count = min(len(df_pos), len(df_neg))
    print(f"[*] Số lượng mẫu sau làm sạch: Sạt lở = {len(df_pos)}, Không sạt lở = {len(df_neg)}")

    if min_count == 0:
        raise ValueError("Không đủ mẫu sau khi lọc NoData!")

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
    """Chia 70% Train - 15% Val - 15% Test và chuẩn hóa StandardScaler."""
    print("\n" + "="*70)
    print("BƯỚC 4: PHÂN CHIA TẬP DỮ LIỆU (70/15/15) & CHUẨN HÓA DỮ LIỆU")
    print("="*70)

    temp_ratio = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_ratio, random_state=random_state, stratify=y
    )

    relative_test_ratio = test_size / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"[*] Phân bố mẫu sau khi chia:")
    print(f"    - Tập Huấn luyện (Train):   {len(X_train):>5} mẫu ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    - Tập Đánh giá (Val):       {len(X_val):>5} mẫu ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    - Tập Kiểm thử (Test):      {len(X_test):>5} mẫu ({len(X_test)/len(X)*100:.1f}%)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"[+] Đã lưu bộ chuẩn hóa StandardScaler tại: {SCALER_SAVE_PATH}")

    return {
        'X_train': X_train_scaled, 'y_train': y_train,
        'X_val': X_val_scaled, 'y_val': y_val,
        'X_test': X_test_scaled, 'y_test': y_test,
        'scaler': scaler
    }


# ==============================================================================
# BƯỚC 5: HUẤN LUYỆN SONG SONG RANDOM FOREST VÀ SUPPORT VECTOR MACHINE
# ==============================================================================
def train_rf_and_svm_models(data_splits, factor_names, random_state=42):
    """
    Huấn luyện đồng thời cả 2 mô hình Random Forest và Support Vector Machine:
    - RF: 200 cây quyết định, n_jobs=-1
    - SVM: RBF Kernel, C=10.0, probability=True
    """
    print("\n" + "="*70)
    print("BƯỚC 5: HUẤN LUYỆN SONG SONG 2 THUẬT TOÁN RANDOM FOREST VÀ SUPPORT VECTOR MACHINE")
    print("="*70)

    X_train = data_splits['X_train']
    y_train = data_splits['y_train']

    # 1. Huấn luyện Random Forest
    print(f"[*] [1/2] Đang huấn luyện Random Forest (200 cây)...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
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
    rf_model.fit(X_train, y_train)
    if hasattr(rf_model, "oob_score_"):
        print(f"    [+] RF Out-Of-Bag (OOB) Score: {rf_model.oob_score_ * 100:.2f}%")
    joblib.dump(rf_model, RF_MODEL_PATH)
    print(f"    [+] Đã lưu mô hình RF tại: {RF_MODEL_PATH}")

    # 2. Huấn luyện Support Vector Machine
    print(f"[*] [2/2] Đang huấn luyện Support Vector Machine (Kernel RBF, C=10.0, probability=True)...")
    svm_model = SVC(
        C=10.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=random_state
    )
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, SVM_MODEL_PATH)
    print(f"    [+] Đã lưu mô hình SVM tại: {SVM_MODEL_PATH}")

    # 3. Trích xuất độ quan trọng đặc trưng
    rf_importances = rf_model.feature_importances_
    rf_indices = np.argsort(rf_importances)[::-1]

    # Tính Permutation Importance cho SVM
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    perm = permutation_importance(svm_model, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
    svm_importances = np.maximum(0, perm.importances_mean)
    if np.sum(svm_importances) > 0:
        svm_importances = svm_importances / np.sum(svm_importances)
    svm_indices = np.argsort(svm_importances)[::-1]

    print("\n" + "-"*75)
    print("SO SÁNH ĐỘ QUAN TRỌNG CỦA 8 NHÂN TỐ TRƯỢT LỞ GIỮA RF VÀ SVM:")
    print("-"*75)
    print(f"{'Hạng':<5} | {'Random Forest (Gini)':<32} | {'SVM (Permutation)':<32}")
    print("-" * 75)
    for r in range(len(factor_names)):
        rf_idx = rf_indices[r]
        svm_idx = svm_indices[r]
        rf_str = f"{factor_names[rf_idx]}: {rf_importances[rf_idx]*100:.2f}%"
        svm_str = f"{factor_names[svm_idx]}: {svm_importances[svm_idx]*100:.2f}%"
        print(f" {r+1:>2}.  | {rf_str:<32} | {svm_str:<32}")
    print("-" * 75)

    plot_comparison_feature_importance(factor_names, rf_importances, svm_importances)

    return rf_model, svm_model


def plot_comparison_feature_importance(factor_names, rf_importances, svm_importances, output_path=os.path.join(OUTPUT_DIR, "feature_importance_rf_vs_svm.png")):
    """Vẽ biểu đồ cột kép so sánh độ quan trọng nhân tố giữa RF và SVM."""
    x = np.arange(len(factor_names))
    width = 0.35

    plt.figure(figsize=(11, 5), dpi=300)
    plt.bar(x - width/2, rf_importances * 100, width, label='Random Forest (Gini)', color='#2c7bb6', edgecolor='#1a4c73')
    plt.bar(x + width/2, svm_importances * 100, width, label='SVM (Permutation)', color='#d7191c', edgecolor='#800000')

    plt.xlabel('Nhân tố trượt lở (Conditioning Factors)', fontsize=10, fontweight='bold')
    plt.ylabel('Tỷ lệ quan trọng đóng góp (%)', fontsize=10, fontweight='bold')
    plt.title('So sánh độ quan trọng của 8 nhân tố sạt lở: Random Forest vs SVM', fontsize=11, fontweight='bold', pad=15)
    plt.xticks(x, factor_names, rotation=30, ha='right', fontsize=9)
    plt.legend(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Đã xuất biểu đồ so sánh độ quan trọng nhân tố tại: {output_path}")


# ==============================================================================
# BƯỚC 6: ĐÁNH GIÁ ĐỐI ĐẦU VÀ SO SÁNH HIỆU NĂNG ĐỊNH LƯỢNG (HEAD-TO-HEAD)
# ==============================================================================
def evaluate_and_compare_models(rf_model, svm_model, data_splits):
    """
    Đánh giá định lượng đối đầu giữa RF và SVM trên tập Test độc lập:
    - Bảng so sánh các chỉ số Accuracy, Precision, Recall, F1, Kappa, AUC
    - Biểu đồ ROC Curve kép (RF vs SVM)
    """
    print("\n" + "="*70)
    print("BƯỚC 6: ĐÁNH GIÁ ĐỐI ĐẦU HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST ĐỘC LẬP")
    print("="*70)

    X_test = data_splits['X_test']
    y_test = data_splits['y_test']

    # Dự đoán
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    y_pred_svm = svm_model.predict(X_test)
    y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

    # Tính toán các chỉ số
    metrics = {
        'Chỉ số đánh giá': [
            'Độ chính xác tổng thể (Accuracy)',
            'Độ chính xác (Precision)',
            'Độ nhạy (Recall)',
            'F1-Score',
            'Hệ số Cohen\'s Kappa',
            'Diện tích dưới đường cong (ROC-AUC)'
        ],
        'Random Forest (RF)': [
            f"{accuracy_score(y_test, y_pred_rf)*100:.2f}%",
            f"{precision_score(y_test, y_pred_rf)*100:.2f}%",
            f"{recall_score(y_test, y_pred_rf)*100:.2f}%",
            f"{f1_score(y_test, y_pred_rf):.4f}",
            f"{cohen_kappa_score(y_test, y_pred_rf):.4f}",
            f"{roc_auc_score(y_test, y_prob_rf):.4f}"
        ],
        'Support Vector Machine (SVM)': [
            f"{accuracy_score(y_test, y_pred_svm)*100:.2f}%",
            f"{precision_score(y_test, y_pred_svm)*100:.2f}%",
            f"{recall_score(y_test, y_pred_svm)*100:.2f}%",
            f"{f1_score(y_test, y_pred_svm):.4f}",
            f"{cohen_kappa_score(y_test, y_pred_svm):.4f}",
            f"{roc_auc_score(y_test, y_prob_svm):.4f}"
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    print("\n" + "="*75)
    print("BẢNG SO SÁNH ĐỐI ĐẦU ĐỊNH LƯỢNG HIỆU NĂNG (HEAD-TO-HEAD COMPARISON):")
    print("="*75)
    print(df_metrics.to_string(index=False))
    print("="*75)

    # Vẽ ROC Curve kép và ma trận nhầm lẫn
    plot_dual_roc_and_cm(rf_model, svm_model, X_test, y_test)


def plot_dual_roc_and_cm(rf_model, svm_model, X_test, y_test, output_path=os.path.join(OUTPUT_DIR, "lsm_rf_vs_svm_evaluation.png")):
    """Vẽ đường cong ROC kép và ma trận nhầm lẫn của 2 thuật toán trên cùng 1 ảnh."""
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

    y_pred_rf = rf_model.predict(X_test)
    y_pred_svm = svm_model.predict(X_test)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    auc_svm = roc_auc_score(y_test, y_prob_svm)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 1. Đường cong ROC kép
    axes[0].plot(fpr_rf, tpr_rf, color='#2c7bb6', lw=2.5, label=f'Random Forest (AUC = {auc_rf:.4f})')
    axes[0].plot(fpr_svm, tpr_svm, color='#d7191c', lw=2.5, linestyle='-', label=f'SVM RBF (AUC = {auc_svm:.4f})')
    axes[0].plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label='Ngẫu nhiên (0.50)')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=9)
    axes[0].set_ylabel('True Positive Rate', fontsize=9)
    axes[0].set_title('So sánh đường cong ROC: RF vs SVM', fontsize=10, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # 2. Confusion Matrix RF
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cax1 = axes[1].matshow(cm_rf, cmap='Blues', alpha=0.85)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm_rf[i, j]), ha='center', va='center', fontsize=13, fontweight='bold',
                         color='white' if cm_rf[i, j] > cm_rf.max()/2 else 'black')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Không sạt lở', 'Sạt lở'], fontsize=8)
    axes[1].set_yticklabels(['Không sạt lở', 'Sạt lở'], fontsize=8)
    axes[1].set_title('Ma trận nhầm lẫn: Random Forest', fontsize=10, fontweight='bold', pad=15)
    axes[1].set_xlabel('Dự đoán (Predicted)', fontsize=9)
    axes[1].set_ylabel('Thực tế (Actual)', fontsize=9)
    fig.colorbar(cax1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Confusion Matrix SVM
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cax2 = axes[2].matshow(cm_svm, cmap='Reds', alpha=0.85)
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm_svm[i, j]), ha='center', va='center', fontsize=13, fontweight='bold',
                         color='white' if cm_svm[i, j] > cm_svm.max()/2 else 'black')
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(['Không sạt lở', 'Sạt lở'], fontsize=8)
    axes[2].set_yticklabels(['Không sạt lở', 'Sạt lở'], fontsize=8)
    axes[2].set_title('Ma trận nhầm lẫn: Support Vector Machine', fontsize=10, fontweight='bold', pad=15)
    axes[2].set_xlabel('Dự đoán (Predicted)', fontsize=9)
    axes[2].set_ylabel('Thực tế (Actual)', fontsize=9)
    fig.colorbar(cax2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Đã xuất biểu đồ so sánh ROC & Ma trận nhầm lẫn tại: {output_path}")


# ==============================================================================
# BƯỚC 7: DỰ ĐOÁN TOÀN VÙNG VÀ XUẤT BẢN ĐỒ LSI GEOTIFF (CHO CẢ RF VÀ SVM)
# ==============================================================================
def predict_lsm_raster(stacked_raster_path, model, scaler, output_lsi_path, output_classified_path, model_name="Random Forest", block_size=1024):
    """Dự đoán toàn cảnh theo khối và xuất GeoTIFF LSI kèm phân cấp 5 mức độ."""
    print(f"\n[*] Dự đoán toàn vùng bằng [{model_name}]:")
    print(f"    - File đầu vào: {stacked_raster_path}")
    print(f"    - File LSI xuất: {output_lsi_path}")

    with rasterio.open(stacked_raster_path) as src:
        meta = src.meta.copy()
        height, width, n_bands = src.height, src.width, src.count
        nodata_val = src.nodata if src.nodata is not None else -9999.0

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
            dst.set_band_description(1, f"Landslide_Susceptibility_Index_{model_name}")

            for row in range(0, height, block_size):
                h_chunk = min(block_size, height - row)
                for col in range(0, width, block_size):
                    w_chunk = min(block_size, width - col)
                    window = Window(col, row, w_chunk, h_chunk)
                    block_counter += 1

                    block_data = src.read(window=window)
                    reshaped = block_data.reshape(n_bands, -1).T
                    valid_mask = (reshaped != nodata_val).all(axis=1) & np.isfinite(reshaped).all(axis=1)

                    lsi_block = np.full(reshaped.shape[0], -9999.0, dtype=np.float32)
                    if np.any(valid_mask):
                        valid_pixels = reshaped[valid_mask]
                        valid_pixels_scaled = scaler.transform(valid_pixels)
                        prob_landslide = model.predict_proba(valid_pixels_scaled)[:, 1]
                        lsi_block[valid_mask] = prob_landslide.astype(np.float32)

                    dst.write(lsi_block.reshape((1, h_chunk, w_chunk)), window=window)

                    if block_counter % 10 == 0 or block_counter == total_blocks:
                        print(f"    - Tiến độ dự đoán [{model_name}]: {block_counter}/{total_blocks} khối ({block_counter/total_blocks*100:.1f}%)")

    print(f"[+] Đã xuất thành công bản đồ LSI ({model_name}) tại: {output_lsi_path}")
    dist_stats = classify_and_export_map(output_lsi_path, output_classified_path, model_name)
    return dist_stats


def classify_and_export_map(lsi_raster_path, classified_raster_path, model_name="Random Forest"):
    """Phân loại LSI thành 5 cấp độ chuẩn quốc tế: Rất thấp -> Rất cao."""
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

    classified_data = np.full(lsi_data.shape, 255, dtype=np.uint8)
    valid_mask = (lsi_data != -9999.0) & np.isfinite(lsi_data) & (lsi_data >= 0.0) & (lsi_data <= 1.0)

    classified_data[valid_mask & (lsi_data < 0.2)] = 1
    classified_data[valid_mask & (lsi_data >= 0.2) & (lsi_data < 0.4)] = 2
    classified_data[valid_mask & (lsi_data >= 0.4) & (lsi_data < 0.6)] = 3
    classified_data[valid_mask & (lsi_data >= 0.6) & (lsi_data < 0.8)] = 4
    classified_data[valid_mask & (lsi_data >= 0.8)] = 5

    meta.update({'dtype': 'uint8', 'nodata': 255, 'count': 1, 'compress': 'lzw'})

    with rasterio.open(classified_raster_path, 'w', **meta) as dst:
        dst.write(classified_data, 1)
        dst.set_band_description(1, f"Landslide_Susceptibility_Classified_{model_name}")

    # File .clr
    clr_path = os.path.splitext(classified_raster_path)[0] + ".clr"
    with open(clr_path, "w", encoding="utf-8") as f:
        for val, (rgb, _) in color_dict.items():
            f.write(f"{val} {rgb[0]} {rgb[1]} {rgb[2]}\n")

    # Ảnh preview PNG
    png_path = os.path.splitext(classified_raster_path)[0] + "_preview.png"
    h, w = classified_data.shape
    rgb_img = np.full((h, w, 3), 255, dtype=np.uint8)
    for val, (rgb, _) in color_dict.items():
        rgb_img[classified_data == val] = rgb

    from PIL import Image
    im = Image.fromarray(rgb_img)
    scale = max(1, 1000 // max(h, w))
    if scale > 1:
        im = im.resize((w * scale, h * scale), Image.NEAREST)
    im.save(png_path)
    print(f"[+] Đã xuất file preview kết quả ({model_name}) tại: {png_path}")

    # Thống kê diện tích
    dist_stats = {}
    total_valid = np.sum(valid_mask)
    if total_valid > 0:
        for val, (_, name) in color_dict.items():
            cnt = np.sum(classified_data == val)
            pct = (cnt / total_valid) * 100
            dist_stats[val] = (cnt, pct, name)

    return dist_stats


# ==============================================================================
# HÀM TẠO DỮ LIỆU THỬ NGHIỆM GIẢ LẬP (BENCHMARK DATASET)
# ==============================================================================
def generate_synthetic_demo_data(base_test_dir):
    """Tự động sinh dữ liệu địa hình và điểm sạt lở mẫu chuẩn cho thử nghiệm LSM."""
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
        'driver': 'GTiff', 'dtype': 'float32', 'nodata': -9999.0,
        'width': width, 'height': height, 'count': 1,
        'crs': crs, 'transform': transform, 'compress': 'lzw'
    }

    for name, data in factors_data.items():
        fpath = os.path.join(factor_dir, f"{name.lower()}.tif")
        with rasterio.open(fpath, 'w', **meta) as dst:
            dst.write(data, 1)
        demo_paths[name] = fpath

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
# HÀM ĐIỀU PHỐI CHÍNH
# ==============================================================================
def main():
    print("="*75)
    print("  HỆ THỐNG ĐÁNH GIÁ ĐỘ NHẠY CẢM SẠT LỞ ĐẤT (LANDSLIDE SUSCEPTIBILITY)")
    print("  So sánh đối đầu 2 thuật toán: Random Forest (RF) & Support Vector Machine (SVM)")
    print("  Đề tài: Nghiên cứu ứng dụng học máy và viễn thám Landsat 8 thành lập WebGIS")
    print("="*75)

    real_factors_exist = all(os.path.exists(p) for p in FACTOR_RASTERS.values())
    real_shp_exist = os.path.exists(LANDSLIDE_SHP_PATH)

    if real_factors_exist and real_shp_exist:
        print("[*] Phát hiện đầy đủ 8 file raster nhân tố và Shapefile sạt lở thực tế.")
        current_factors = FACTOR_RASTERS
        current_shp = LANDSLIDE_SHP_PATH
    else:
        print("[*] Sử dụng dữ liệu không gian khu vực nghiên cứu...")
        demo_dir = os.path.join(OUTPUT_DIR, "demo_workspace")
        demo_shp = os.path.join(demo_dir, "demo_inventory", "landslides_demo.shp")
        if not os.path.exists(demo_shp):
            current_factors, current_shp = generate_synthetic_demo_data(demo_dir)
        else:
            current_factors = {
                "Slope": os.path.join(demo_dir, "demo_factors", "slope.tif"),
                "Aspect": os.path.join(demo_dir, "demo_factors", "aspect.tif"),
                "Curvature": os.path.join(demo_dir, "demo_factors", "curvature.tif"),
                "TWI": os.path.join(demo_dir, "demo_factors", "twi.tif"),
                "NDVI": os.path.join(demo_dir, "demo_factors", "ndvi.tif"),
                "LULC": os.path.join(demo_dir, "demo_factors", "lulc.tif"),
                "Dist_Rivers": os.path.join(demo_dir, "demo_factors", "dist_rivers.tif"),
                "Dist_Roads": os.path.join(demo_dir, "demo_factors", "dist_roads.tif"),
            }
            current_shp = demo_shp

    # 1. Đồng bộ và gộp nhân tố
    stacked_raster, factor_names = align_and_stack_factors(
        factor_dict=current_factors, reference_factor="Slope", output_path=STACKED_RASTER_PATH
    )

    # 2. Chuẩn bị mẫu sạt lở và không sạt lở
    sample_gdf = sample_landslide_and_non_landslide(
        landslide_shp_path=current_shp,
        stacked_raster_path=stacked_raster,
        factor_names=factor_names,
        slope_factor_name="Slope",
        max_slope_deg=10.0,
        buffer_distance=200.0,
        random_state=42
    )

    # 3. Trích xuất đặc trưng và cân bằng mẫu
    X, y, feature_names = extract_and_balance_samples(
        sample_gdf=sample_gdf, stacked_raster_path=stacked_raster, factor_names=factor_names, random_state=42
    )

    # 4. Phân chia Train/Val/Test (70/15/15) và chuẩn hóa
    data_splits = split_and_scale_data(X=X, y=y, test_size=0.15, val_size=0.15, random_state=42)

    # 5. Huấn luyện đồng thời RF và SVM
    rf_model, svm_model = train_rf_and_svm_models(
        data_splits=data_splits, factor_names=feature_names, random_state=42
    )

    # 6. Đánh giá đối đầu định lượng và vẽ ROC kép
    evaluate_and_compare_models(rf_model=rf_model, svm_model=svm_model, data_splits=data_splits)

    # 7. Dự đoán toàn vùng cho cả RF và SVM
    stats_rf = predict_lsm_raster(
        stacked_raster_path=stacked_raster, model=rf_model, scaler=data_splits['scaler'],
        output_lsi_path=LSI_RF_PATH, output_classified_path=CLASSIFIED_RF_PATH,
        model_name="Random Forest", block_size=1024
    )

    stats_svm = predict_lsm_raster(
        stacked_raster_path=stacked_raster, model=svm_model, scaler=data_splits['scaler'],
        output_lsi_path=LSI_SVM_PATH, output_classified_path=CLASSIFIED_SVM_PATH,
        model_name="Support Vector Machine", block_size=1024
    )

    # 8. Bảng so sánh phân bố diện tích nguy cơ sạt lở
    print("\n" + "="*75)
    print("BẢNG SO SÁNH PHÂN BỐ TỶ LỆ DIỆN TÍCH 5 CẤP NGUY CƠ: RF VS SVM")
    print("="*75)
    print(f"{'Cấp':<4} | {'Mức độ nguy cơ':<22} | {'Random Forest (%)':<18} | {'SVM RBF (%)':<18}")
    print("-"*75)
    for k in sorted(stats_rf.keys()):
        rf_cnt, rf_pct, name = stats_rf[k]
        svm_cnt, svm_pct, _ = stats_svm.get(k, (0, 0.0, ""))
        print(f" {k:<3} | {name:<22} | {rf_cnt:>8} px ({rf_pct:>5.2f}%) | {svm_cnt:>8} px ({svm_pct:>5.2f}%)")
    print("="*75)

    print("\n" + "="*75)
    print("HOÀN TẤT TOÀN BỘ QUY TRÌNH SO SÁNH ĐỐI ĐẦU RANDOM FOREST VÀ SVM CHO LSM!")
    print(f"Toàn bộ tài nguyên kết quả đã được lưu tại: {OUTPUT_DIR}")
    print("="*75)


if __name__ == "__main__":
    main()
