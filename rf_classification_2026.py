# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH XỬ LÝ DỮ LIỆU VIỄN THÁM VÀ PHÂN LOẠI RANDOM FOREST NĂM 2026
Tích hợp từ: test.py, rf_classification.py và tương thích lulc_change_detection.py
--------------------------------------------------------------------------------
Các bước thực hiện:
  1. Đọc ảnh viễn thám Landsat 8 năm 2026 (kênh B1-B7 hoặc ảnh cắt dp2026Extrac.tif).
  2. Tính toán bộ 4 chỉ số phổ chuyên sâu: NDVI, NDBI, MNDWI, NDWI (tạo stack 11 kênh).
  3. Cắt theo mặt nạ ranh giới xã (ketquatach.shp) và vẽ biểu đồ 4 chỉ số phổ.
  4. Trích xuất mẫu huấn luyện từ Shapefile DKT2026.shp (cột 'Nhan').
  5. Cân bằng số lượng mẫu giữa các lớp (Stratified Sampling).
  6. Huấn luyện mô hình Random Forest (chia 70% Train - 15% Val - 15% Test).
  7. Đánh giá toàn diện: Overall Accuracy, Cohen's Kappa, Confusion Matrix, Classification Report.
  8. Lưu mô hình rf_model_2026.pkl.
  9. Phân loại toàn bộ ảnh raster theo khối và xuất file GeoTIFF: anh_phan_loai_2026.tif.
  10. Xuất bảng màu ArcMap (.clr), ảnh màu xem trước (.png) và thống kê diện tích (ha).
================================================================================
"""

import os
import sys

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi font mã hóa
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Tự động tạo lại file .shx nếu bị thiếu khi đọc Shapefile
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Khắc phục cảnh báo xung đột thư viện PROJ database trên Windows
try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    if os.path.exists(proj_dir):
        os.environ['PROJ_LIB'] = proj_dir
except Exception:
    pass

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import joblib

# Hỗ trợ hiển thị biểu đồ chỉ số phổ nếu có thư viện matplotlib
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from PIL import Image


# ==============================================================================
# CẤU HÌNH THÔNG SỐ VÀ ĐƯỜNG DẪN DỮ LIỆU NĂM 2026
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 1. Đường dẫn ảnh đầu vào 2026 (Ưu tiên ảnh đã cắt theo xã dp2026Extrac.tif)
INPUT_RASTER_2026 = os.path.join(BASE_DIR, 'data', 'dp2026Extrac.tif')
RAW_BAND_FOLDER_2026 = os.path.join(BASE_DIR, 'LC8_HaNoi_2026')

# 2. File ranh giới khu vực nghiên cứu / xã (Shapefile)
BOUNDARY_SHP_PATH = os.path.join(BASE_DIR, 'data', 'ketquatach.shp')

# 3. File mẫu huấn luyện năm 2026 từ ArcMap
ROI_SHP_2026 = os.path.join(BASE_DIR, 'DKT2026', 'DKT2026.shp')
CLASS_COLUMN_NAME = "Nhan"

# 4. Định nghĩa danh mục lớp phủ năm 2026
# Mã lớp thực tế trong DKT2026: 1 (Nước), 10 (Nông nghiệp), 13 (Xây dựng/Dân cư), 19 (Thực vật)
CLASS_NAMES_2026 = {
    1: "Đất mặt nước",
    10: "Đất nông nghiệp / Đất trống",
    13: "Đất dân cư và xây dựng",
    19: "Thảm thực vật"
}

# Bảng màu hiển thị (RGB)
CLASS_COLORS_2026 = {
    1: [0, 92, 230],       # Mặt nước (Xanh dương)
    10: [255, 255, 115],   # Đất nông nghiệp / trống (Vàng)
    13: [230, 0, 0],       # Dân cư / Xây dựng (Đỏ)
    19: [38, 115, 0],      # Thảm thực vật (Xanh lá)
}

# 5. Đường dẫn lưu trữ kết quả đầu ra
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CLASSIFIED_TIF = os.path.join(OUTPUT_DIR, 'anh_phan_loai_2026.tif')
OUTPUT_CLASSIFIED_CLR = os.path.join(OUTPUT_DIR, 'anh_phan_loai_2026.clr')
OUTPUT_CLASSIFIED_PNG = os.path.join(OUTPUT_DIR, 'anh_phan_loai_2026_preview.png')
OUTPUT_INDICES_PNG = os.path.join(OUTPUT_DIR, 'chi_so_pho_2026.png')
OUTPUT_MODEL_PKL = os.path.join(BASE_DIR, 'giscode', 'rf_model_2026.pkl')


# ==============================================================================
# HÀM 1: ĐỌC VÀ TIỀN XỬ LÝ ẢNH VIỄN THÁM NĂM 2026 (7 KÊNH + 4 CHỈ SỐ PHỔ)
# ==============================================================================
def load_and_preprocess_2026():
    """
    Đọc dữ liệu ảnh năm 2026, tính toán 4 chỉ số phổ:
    NDVI, NDBI, MNDWI, NDWI tạo thành stack 11 kênh đặc trưng.
    """
    print("=" * 70)
    print("BƯỚC 1: ĐỌC DỮ LIỆU ẢNH NĂM 2026 VÀ TÍNH TOÁN BỘ CHỈ SỐ PHỔ")
    print("=" * 70)

    if os.path.exists(INPUT_RASTER_2026):
        print(f"[*] Đang đọc ảnh viễn thám đa phổ từ: {INPUT_RASTER_2026}")
        src = rasterio.open(INPUT_RASTER_2026)
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        shape = src.shape
        nodata_val = src.nodata if src.nodata is not None else 65536.0

        b1 = src.read(1).astype(float)
        b2 = src.read(2).astype(float)
        b3 = src.read(3).astype(float)
        b4 = src.read(4).astype(float)
        b5 = src.read(5).astype(float)
        b6 = src.read(6).astype(float)
        b7 = src.read(7).astype(float)
    else:
        print(f"[*] Không tìm thấy {INPUT_RASTER_2026}, đọc từ thư mục kênh: {RAW_BAND_FOLDER_2026}")
        def find_band(b_num):
            for f in os.listdir(RAW_BAND_FOLDER_2026):
                u = f.upper()
                if u.endswith(f"_B{b_num}.TIF") or u.endswith(f"_SR_B{b_num}.TIF"):
                    return os.path.join(RAW_BAND_FOLDER_2026, f)
            raise FileNotFoundError(f"Không tìm thấy kênh B{b_num} trong {RAW_BAND_FOLDER_2026}")

        with rasterio.open(find_band(1)) as s1:
            profile = s1.profile.copy()
            crs = s1.crs
            transform = s1.transform
            shape = s1.shape
            nodata_val = s1.nodata if s1.nodata is not None else 0.0
            b1 = s1.read(1).astype(float)

        b2 = rasterio.open(find_band(2)).read(1).astype(float)
        b3 = rasterio.open(find_band(3)).read(1).astype(float)
        b4 = rasterio.open(find_band(4)).read(1).astype(float)
        b5 = rasterio.open(find_band(5)).read(1).astype(float)
        b6 = rasterio.open(find_band(6)).read(1).astype(float)
        b7 = rasterio.open(find_band(7)).read(1).astype(float)

    # Lọc NoData trước khi tính toán chỉ số
    valid_mask = (b1 != nodata_val) & (b2 != nodata_val) & (b3 != nodata_val) & \
                 (b4 != nodata_val) & (b5 != nodata_val) & (b6 != nodata_val) & (b7 != nodata_val)

    print("[*] Đang tính toán 4 chỉ số phổ chuyên sâu (NDVI, NDBI, MNDWI, NDWI)...")
    np.seterr(divide='ignore', invalid='ignore')

    # 1. NDVI (Thực vật): (B5 - B4) / (B5 + B4)
    ndvi = np.nan_to_num((b5 - b4) / (b5 + b4), nan=0.0)

    # 2. NDBI (Đô thị / Bê tông / Đất trống): (B6 - B5) / (B6 + B5)
    ndbi = np.nan_to_num((b6 - b5) / (b6 + b5), nan=0.0)

    # 3. MNDWI (Mặt nước sạch - Tách biệt đất xây dựng): (B3 - B6) / (B3 + B6)
    mndwi = np.nan_to_num((b3 - b6) / (b3 + b6), nan=0.0)

    # 4. NDWI (Độ ẩm mặt nước truyền thống): (B3 - B5) / (B3 + B5)
    ndwi = np.nan_to_num((b3 - b5) / (b3 + b5), nan=0.0)

    # Ghép thành 11 kênh đặc trưng (7 kênh phổ + 4 chỉ số)
    stacked_image = np.stack([b1, b2, b3, b4, b5, b6, b7, ndvi, ndbi, mndwi, ndwi])
    print(f"[+] Hoàn thành stack ảnh 11 kênh: Shape = {stacked_image.shape}")

    # Áp dụng mặt nạ ranh giới xã nếu có
    mask_boundary = np.ones(shape, dtype=bool)
    if os.path.exists(BOUNDARY_SHP_PATH):
        print(f"[*] Áp dụng ranh giới nghiên cứu từ: {BOUNDARY_SHP_PATH}")
        b_gdf = gpd.read_file(BOUNDARY_SHP_PATH)
        if b_gdf.crs != crs:
            b_gdf = b_gdf.to_crs(crs)
        mask_boundary = geometry_mask(b_gdf.geometry, transform=transform, invert=True, out_shape=shape)

    # Kết hợp mặt nạ hợp lệ: pixel vừa nằm trong ranh giới vừa không phải NoData
    final_valid_mask = valid_mask & mask_boundary

    return {
        'stacked': stacked_image,
        'b_indices': (ndvi, ndbi, mndwi, ndwi),
        'profile': profile,
        'crs': crs,
        'transform': transform,
        'shape': shape,
        'valid_mask': final_valid_mask,
        'nodata_val': nodata_val
    }


# ==============================================================================
# HÀM 2: VẼ VÀ XUẤT BIỂU ĐỒ TRỰC QUAN 4 CHỈ SỐ PHỔ (TỪ test.py)
# ==============================================================================
def plot_spectral_indices(b_indices, valid_mask, output_png):
    """Vẽ và lưu biểu đồ phân bố 4 chỉ số phổ năm 2026."""
    if not HAS_MATPLOTLIB:
        print("[!] Thư viện matplotlib chưa có, bỏ qua xuất biểu đồ chỉ số phổ.")
        return

    ndvi, ndbi, mndwi, ndwi = b_indices
    ndvi_p = np.where(valid_mask, ndvi, np.nan)
    ndbi_p = np.where(valid_mask, ndbi, np.nan)
    mndwi_p = np.where(valid_mask, mndwi, np.nan)
    ndwi_p = np.where(valid_mask, ndwi, np.nan)

    print(f"[*] Đang lưu biểu đồ 4 chỉ số phổ năm 2026 tại: {output_png}")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    def add_cbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(im, cax=cax)

    # 1. NDVI
    im1 = axes[0].imshow(ndvi_p, cmap='RdYlGn', vmin=float(np.nanmin(ndvi_p)), vmax=float(np.nanmax(ndvi_p)))
    axes[0].set_title(f"1. NDVI 2026 (Thực vật)\n[{np.nanmin(ndvi_p):.2f} ~ {np.nanmax(ndvi_p):.2f}]", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    add_cbar(im1, axes[0])

    # 2. NDWI
    im2 = axes[1].imshow(ndwi_p, cmap='Blues', vmin=float(np.nanmin(ndwi_p)), vmax=float(np.nanmax(ndwi_p)))
    axes[1].set_title(f"2. NDWI 2026 (Nước TC)\n[{np.nanmin(ndwi_p):.2f} ~ {np.nanmax(ndwi_p):.2f}]", fontsize=10, fontweight='bold')
    axes[1].axis('off')
    add_cbar(im2, axes[1])

    # 3. MNDWI
    im3 = axes[2].imshow(mndwi_p, cmap='Blues', vmin=float(np.nanmin(mndwi_p)), vmax=float(np.nanmax(mndwi_p)))
    axes[2].set_title(f"3. MNDWI 2026 (Nước NC)\n[{np.nanmin(mndwi_p):.2f} ~ {np.nanmax(mndwi_p):.2f}]", fontsize=10, fontweight='bold')
    axes[2].axis('off')
    add_cbar(im3, axes[2])

    # 4. NDBI
    im4 = axes[3].imshow(ndbi_p, cmap='coolwarm', vmin=float(np.nanmin(ndbi_p)), vmax=float(np.nanmax(ndbi_p)))
    axes[3].set_title(f"4. NDBI 2026 (Xây dựng)\n[{np.nanmin(ndbi_p):.2f} ~ {np.nanmax(ndbi_p):.2f}]", fontsize=10, fontweight='bold')
    axes[3].axis('off')
    add_cbar(im4, axes[3])

    plt.tight_layout(pad=1.5)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Đã lưu ảnh phân tích 4 chỉ số phổ: {output_png}")


# ==============================================================================
# HÀM 3: TRÍCH XUẤT MẪU VÀ CÂN BẰNG MẪU (TỪ DKT2026.shp)
# ==============================================================================
def extract_and_balance_samples(stacked_img, transform, shape, crs):
    """Trích xuất giá trị 11 kênh tại các điểm mẫu và cân bằng số lượng mẫu."""
    print("\n" + "=" * 70)
    print("BƯỚC 2: TRÍCH XUẤT MẪU ĐẶC TRƯNG TỪ SHAPEFILE DKT2026.shp")
    print("=" * 70)

    if not os.path.exists(ROI_SHP_2026):
        raise FileNotFoundError(f"Không tìm thấy file mẫu: {ROI_SHP_2026}")

    roi_gdf = gpd.read_file(ROI_SHP_2026)
    print(f"[*] Đọc thành công {len(roi_gdf)} mẫu từ: {ROI_SHP_2026}")

    if roi_gdf.crs != crs:
        print(f"[*] Chuyển hệ tọa độ từ {roi_gdf.crs} sang {crs}...")
        roi_gdf = roi_gdf.to_crs(crs)

    col = CLASS_COLUMN_NAME
    if col not in roi_gdf.columns:
        for cand in ['Nhan', 'nhan', 'NHAN', 'label', 'Classvalue', 'Id']:
            if cand in roi_gdf.columns and roi_gdf[cand].nunique() > 1:
                col = cand
                break

    print(f"[*] Sử dụng cột nhãn: '{col}' (Số lớp: {roi_gdf[col].nunique()})")

    X_list = []
    y_list = []

    for _, row in roi_gdf.iterrows():
        geom = row['geometry']
        lbl = row[col]

        try:
            if geom.geom_type in ['Point', 'MultiPoint']:
                points = [geom] if geom.geom_type == 'Point' else list(geom.geoms)
                for pt in points:
                    r, c = rasterio.transform.rowcol(transform, pt.x, pt.y)
                    if 0 <= r < shape[0] and 0 <= c < shape[1]:
                        pixel_val = stacked_img[:, r, c]
                        if not np.any(np.isnan(pixel_val)):
                            X_list.append(pixel_val)
                            y_list.append(lbl)
            else:
                # Dạng Polygon
                m_2d = geometry_mask([geom], transform=transform, invert=True, out_shape=shape)
                rows, cols = np.where(m_2d)
                for r, c in zip(rows, cols):
                    pixel_val = stacked_img[:, r, c]
                    if not np.any(np.isnan(pixel_val)):
                        X_list.append(pixel_val)
                        y_list.append(lbl)
        except Exception:
            continue

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"[+] Trích xuất được tổng cộng: {len(X)} pixel mẫu.")
    df_samples = pd.DataFrame(X)
    df_samples['label'] = y
    print("[*] Cơ cấu mẫu ban đầu:\n", df_samples['label'].value_counts())

    # Cân bằng mẫu (Stratified sample balancing)
    min_count = df_samples['label'].value_counts().min()
    n_sample = min(min_count, 1500)
    df_balanced = df_samples.groupby('label').sample(n=n_sample, random_state=42).reset_index(drop=True)

    X_balanced = df_balanced.drop('label', axis=1).values
    y_balanced = df_balanced['label'].values

    print(f"[+] Sau khi cân bằng ({n_sample} mẫu/lớp), tổng mẫu đưa vào học: {len(X_balanced)}")
    return X_balanced, y_balanced


# ==============================================================================
# HÀM 4: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST
# ==============================================================================
def train_and_evaluate_rf(X, y):
    """Huấn luyện Random Forest và đánh giá Train / Val (15%) / Test (15%)."""
    print("\n" + "=" * 70)
    print("BƯỚC 3: HUẤN LUYỆN RANDOM FOREST VÀ ĐÁNH GIÁ ĐỘ CHÍNH XÁC (2026)")
    print("=" * 70)

    # Tách 70% Train và 30% (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Tách 15% Val và 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[*] Phân chia tập dữ liệu: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    print("[*] Đang huấn luyện Random Forest (100 cây, đa luồng n_jobs=-1)...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        oob_score=True
    )
    rf.fit(X_train, y_train)

    # 1. Đánh giá trên tập Validation
    val_pred = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_kappa = cohen_kappa_score(y_val, val_pred)

    # 2. Đánh giá trên tập Test độc lập
    test_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_kappa = cohen_kappa_score(y_test, test_pred)

    print("\n" + "-" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH NĂM 2026")
    print("-" * 50)
    print(f"-> Độ chính xác Out-of-Bag (OOB Score): {rf.oob_score_ * 100:.2f}%")
    print(f"-> Độ chính xác Validation (Accuracy):  {val_acc * 100:.2f}% | Kappa: {val_kappa:.4f}")
    print(f"-> Độ chính xác Test độc lập (Accuracy): {test_acc * 100:.2f}% | Kappa: {test_kappa:.4f}")

    target_names = [f"Lớp {c} ({CLASS_NAMES_2026.get(c, 'Chưa rõ')})" for c in np.unique(y)]
    print("\n[+] BÁO CÁO PHÂN LOẠI CHI TIẾT (TEST SET):")
    print(classification_report(y_test, test_pred, target_names=target_names))

    print("[+] MA TRẬN NHẦM LẪN (CONFUSION MATRIX - TEST SET):")
    print(confusion_matrix(y_test, test_pred))

    # Lưu mô hình .pkl
    joblib.dump(rf, OUTPUT_MODEL_PKL)
    print(f"\n[*] Đã lưu mô hình Random Forest tại: {OUTPUT_MODEL_PKL}")

    return rf


# ==============================================================================
# HÀM 5: PHÂN LOẠI TOÀN BỘ RASTER, XUẤT GEOTIFF, BẢNG MÀU .CLR VÀ PREVIEW .PNG
# ==============================================================================
def classify_and_export(rf_model, data_dict):
    """Phân loại toàn bộ ảnh raster 2026 và xuất các định dạng kết quả chuẩn GIS."""
    print("\n" + "=" * 70)
    print("BƯỚC 4: PHÂN LOẠI TOÀN BỘ ẢNH VÀ XUẤT FILE KẾT QUẢ GEOTIFF 2026")
    print("=" * 70)

    stacked_img = data_dict['stacked']
    profile = data_dict['profile']
    valid_mask = data_dict['valid_mask']
    shape = data_dict['shape']

    n_features, h, w = stacked_img.shape
    classified_raster = np.full(shape, 127, dtype=np.int16)  # 127: Mã NoData ngoài phạm vi

    # Chuẩn bị dữ liệu pixel hợp lệ để dự báo
    pixels = stacked_img.reshape(n_features, -1).T
    flat_valid = valid_mask.reshape(-1)

    print(f"[*] Đang dự báo cho {np.sum(flat_valid):,} pixel hợp lệ trong khu vực nghiên cứu...")
    valid_pixels = pixels[flat_valid]

    # Dự báo phân loại
    predictions = rf_model.predict(valid_pixels)
    classified_raster.reshape(-1)[flat_valid] = predictions

    # 1. Xuất file GeoTIFF kết quả phân loại
    profile.update(
        dtype=rasterio.int16,
        count=1,
        nodata=127,
        compress='lzw'
    )

    with rasterio.open(OUTPUT_CLASSIFIED_TIF, 'w', **profile) as dst:
        dst.write(classified_raster, 1)

    print(f"[+] ĐÃ XUẤT THÀNH CÔNG RASTER PHÂN LOẠI NĂM 2026 (MÃ GỐC): {OUTPUT_CLASSIFIED_TIF}")

    # Xuất thêm file GeoTIFF chuẩn hóa mã (1: Nước, 2: Xây dựng, 3: Nông nghiệp, 4: Cây xanh) để chạy trực tiếp Change Detection
    std_tif_path = os.path.join(OUTPUT_DIR, 'anh_phan_loai_2026_std.tif')
    mapping_to_std = {1: 1, 13: 2, 10: 3, 19: 4}
    std_raster = np.full(shape, -9999, dtype=np.int16)
    for orig_c, std_c in mapping_to_std.items():
        std_raster[classified_raster == orig_c] = std_c

    std_profile = profile.copy()
    std_profile.update(nodata=-9999)
    with rasterio.open(std_tif_path, 'w', **std_profile) as dst_std:
        dst_std.write(std_raster, 1)
    print(f"[+] ĐÃ XUẤT THÀNH CÔNG RASTER CHUẨN HÓA (MÃ 1-4) CHO LULC: {std_tif_path}")

    # 2. Tạo file bảng màu ArcMap (.clr)
    try:
        with open(OUTPUT_CLASSIFIED_CLR, 'w', encoding='utf-8') as f_clr:
            for val, col in sorted(CLASS_COLORS_2026.items()):
                f_clr.write(f"{val} {col[0]} {col[1]} {col[2]}\n")
            f_clr.write("127 255 255 255\n")  # NoData màu trắng
        print(f"[+] Đã tạo file bảng màu ArcMap: {OUTPUT_CLASSIFIED_CLR}")
    except Exception as e:
        print(f"[!] Lỗi khi tạo .clr: {e}")

    # 3. Xuất file ảnh màu xem trước trực quan (.png)
    try:
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_img[:, :] = [255, 255, 255]  # Mặc định nền trắng

        for val, col in CLASS_COLORS_2026.items():
            rgb_img[classified_raster == val] = col

        img_pil = Image.fromarray(rgb_img)
        scale = max(1, 600 // max(h, w))
        if scale > 1:
            img_pil = img_pil.resize((w * scale, h * scale), Image.NEAREST)
        img_pil.save(OUTPUT_CLASSIFIED_PNG)
        print(f"[+] Đã tạo ảnh màu xem trước trực quan: {OUTPUT_CLASSIFIED_PNG}")
    except Exception as e:
        print(f"[!] Lỗi khi xuất ảnh preview: {e}")

    # 4. Thống kê diện tích các lớp phủ năm 2026
    print("\n" + "=" * 70)
    print("THỐNG KÊ DIỆN TÍCH CÁC LỚP PHỦ MẶT ĐẤT NĂM 2026")
    print("=" * 70)
    pixel_area_ha = (30.0 * 30.0) / 10000.0  # 0.09 ha/pixel (Landsat 30m)
    total_valid_pixels = np.sum(flat_valid)

    stats = []
    for c_id, c_name in CLASS_NAMES_2026.items():
        cnt = np.sum(classified_raster == c_id)
        area_ha = cnt * pixel_area_ha
        pct = (cnt / total_valid_pixels * 100.0) if total_valid_pixels > 0 else 0.0
        stats.append({
            "Mã lớp": c_id,
            "Tên lớp phủ": c_name,
            "Số Pixel": cnt,
            "Diện tích (ha)": round(area_ha, 2),
            "Tỷ lệ (%)": round(pct, 2)
        })

    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(index=False))
    print(f"[*] Tổng diện tích nghiên cứu: {df_stats['Diện tích (ha)'].sum():.2f} ha ({total_valid_pixels:,} pixels)")


# ==============================================================================
# HÀM MAIN THỰC THI TOÀN BỘ QUY TRÌNH 2026
# ==============================================================================
def main():
    print("*" * 75)
    print("*   KHỞI CHẠY QUY TRÌNH XỬ LÝ VÀ PHÂN LOẠI DỮ LIỆU VIỄN THÁM NĂM 2026   *")
    print("*" * 75)

    # 1. Đọc và tính toán chỉ số
    data_dict = load_and_preprocess_2026()

    # 2. Xuất biểu đồ 4 chỉ số phổ
    plot_spectral_indices(data_dict['b_indices'], data_dict['valid_mask'], OUTPUT_INDICES_PNG)

    # 3. Trích xuất mẫu huấn luyện từ DKT2026.shp
    X, y = extract_and_balance_samples(
        data_dict['stacked'], data_dict['transform'], data_dict['shape'], data_dict['crs']
    )

    # 4. Huấn luyện Random Forest
    rf_model = train_and_evaluate_rf(X, y)

    # 5. Phân loại và xuất kết quả
    classify_and_export(rf_model, data_dict)

    print("\n" + "*" * 75)
    print("*      HOÀN TẤT TOÀN BỘ QUY TRÌNH XỬ LÝ DỮ LIỆU NĂM 2026 THÀNH CÔNG!     *")
    print("*" * 75)


if __name__ == "__main__":
    main()
