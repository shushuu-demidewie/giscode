# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH PHÂN LOẠI ẢNH VIỄN THÁM BẰNG THUẬT TOÁN SVM (SUPPORT VECTOR MACHINE)
Nâng cấp từ test.py: Đầy đủ tính năng tương đương Random Forest (rf_classification)
Hỗ trợ xử lý và phân loại cả 2 mốc thời gian: Năm 2015 và Năm 2026
--------------------------------------------------------------------------------
Các bước thực hiện:
  1. Đọc ảnh viễn thám đa phổ Landsat 8 (7 kênh quang học B1 - B7).
  2. Tính toán bộ 4 chỉ số phổ chuyên sâu: NDVI, NDBI, MNDWI, NDWI (stack 11 kênh).
  3. Áp dụng mặt nạ ranh giới nghiên cứu (ketquatach.shp) & xuất biểu đồ 4 chỉ số.
  4. Trích xuất mẫu pixel từ Shapefile (DKT2015.shp hoặc DKT2026.shp).
  5. Cân bằng mẫu giữa các lớp (Stratified Balancing) chống thiên vị lớp.
  6. Tiêu chuẩn hóa dữ liệu bằng StandardScaler (bắt buộc đối với SVM RBF kernel).
  7. Huấn luyện SVM (kernel='rbf', C=1.0, gamma='scale') với chia tập Train 70% / Val 15% / Test 15%.
  8. Đánh giá toàn diện: Overall Accuracy, Cohen's Kappa, Confusion Matrix, Classification Report.
  9. Lưu mô hình svm_model_<year>.pkl và bộ chuẩn hóa svm_scaler_<year>.pkl.
  10. Phân loại toàn bộ ảnh raster và xuất file GeoTIFF kết quả.
  11. Xuất file bảng màu ArcMap (.clr), ảnh màu xem trước (.png) và thống kê diện tích (ha).
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ==============================================================================
# CẤU HÌNH HỆ THỐNG VÀ THƯ MỤC
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_SHP_PATH = os.path.join(BASE_DIR, 'data', 'ketquatach.shp')

# Cấu hình danh mục lớp và màu sắc cho từng năm
DATASET_CONFIG = {
    2015: {
        'raster': os.path.join(BASE_DIR, 'data', 'dp2015Extrac.tif'),
        'raw_folder': os.path.join(BASE_DIR, 'LC8_HaNoi_2015'),
        'shp': os.path.join(BASE_DIR, 'Dkt2015', 'DKT2015.shp'),
        'class_col': 'Nhan',
        'classes': {
            1: "Đất mặt nước",
            10: "Đất nông nghiệp / Đất trống",
            14: "Đất dân cư và xây dựng",
            22: "Thảm thực vật"
        },
        'colors': {
            1: [0, 92, 230],       # Xanh dương
            10: [255, 255, 115],   # Vàng
            14: [230, 0, 0],       # Đỏ
            22: [38, 115, 0],      # Xanh lá
        },
        'std_mapping': {1: 1, 14: 2, 10: 3, 22: 4}
    },
    2026: {
        'raster': os.path.join(BASE_DIR, 'data', 'dp2026Extrac.tif'),
        'raw_folder': os.path.join(BASE_DIR, 'LC8_HaNoi_2026'),
        'shp': os.path.join(BASE_DIR, 'DKT2026', 'DKT2026.shp'),
        'class_col': 'Nhan',
        'classes': {
            1: "Đất mặt nước",
            10: "Đất nông nghiệp / Đất trống",
            13: "Đất dân cư và xây dựng",
            19: "Thảm thực vật"
        },
        'colors': {
            1: [0, 92, 230],       # Xanh dương
            10: [255, 255, 115],   # Vàng
            13: [230, 0, 0],       # Đỏ
            19: [38, 115, 0],      # Xanh lá
        },
        'std_mapping': {1: 1, 13: 2, 10: 3, 19: 4}
    }
}


# ==============================================================================
# HÀM 1: ĐỌC VÀ TIỀN XỬ LÝ ẢNH VIỄN THÁM (7 KÊNH + 4 CHỈ SỐ PHỔ = 11 KÊNH)
# ==============================================================================
def load_and_preprocess(year):
    """Đọc ảnh theo năm, tính 4 chỉ số phổ: NDVI, NDBI, MNDWI, NDWI tạo stack 11 kênh."""
    cfg = DATASET_CONFIG[year]
    print(f"\n[*] Đang nạp dữ liệu ảnh viễn thám năm {year}...")

    if os.path.exists(cfg['raster']):
        print(f"[*] Đang đọc ảnh viễn thám đa phổ từ: {cfg['raster']}")
        src = rasterio.open(cfg['raster'])
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
        print(f"[*] Đọc từ thư mục kênh rời: {cfg['raw_folder']}")
        def find_band(b_num):
            for f in os.listdir(cfg['raw_folder']):
                u = f.upper()
                if u.endswith(f"_B{b_num}.TIF") or u.endswith(f"_SR_B{b_num}.TIF"):
                    return os.path.join(cfg['raw_folder'], f)
            raise FileNotFoundError(f"Không tìm thấy kênh B{b_num} trong {cfg['raw_folder']}")

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

    valid_mask = (b1 != nodata_val) & (b2 != nodata_val) & (b3 != nodata_val) & \
                 (b4 != nodata_val) & (b5 != nodata_val) & (b6 != nodata_val) & (b7 != nodata_val)

    print(f"[*] Đang tính toán 4 chỉ số phổ chuyên sâu (NDVI, NDBI, MNDWI, NDWI) năm {year}...")
    np.seterr(divide='ignore', invalid='ignore')

    ndvi = np.nan_to_num((b5 - b4) / (b5 + b4), nan=0.0)
    ndbi = np.nan_to_num((b6 - b5) / (b6 + b5), nan=0.0)
    mndwi = np.nan_to_num((b3 - b6) / (b3 + b6), nan=0.0)
    ndwi = np.nan_to_num((b3 - b5) / (b3 + b5), nan=0.0)

    stacked_image = np.stack([b1, b2, b3, b4, b5, b6, b7, ndvi, ndbi, mndwi, ndwi])
    print(f"[+] Hoàn thành stack ảnh 11 kênh: Shape = {stacked_image.shape}")

    mask_boundary = np.ones(shape, dtype=bool)
    if os.path.exists(BOUNDARY_SHP_PATH):
        b_gdf = gpd.read_file(BOUNDARY_SHP_PATH)
        if b_gdf.crs != crs:
            b_gdf = b_gdf.to_crs(crs)
        mask_boundary = geometry_mask(b_gdf.geometry, transform=transform, invert=True, out_shape=shape)

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
# HÀM 2: VẼ VÀ XUẤT BIỂU ĐỒ 4 CHỈ SỐ PHỔ
# ==============================================================================
def plot_spectral_indices(b_indices, valid_mask, output_png, year):
    """Vẽ và lưu biểu đồ phân bố 4 chỉ số phổ."""
    if not HAS_MATPLOTLIB:
        print("[!] Thư viện matplotlib chưa có, bỏ qua xuất biểu đồ chỉ số phổ.")
        return

    ndvi, ndbi, mndwi, ndwi = b_indices
    ndvi_p = np.where(valid_mask, ndvi, np.nan)
    ndbi_p = np.where(valid_mask, ndbi, np.nan)
    mndwi_p = np.where(valid_mask, mndwi, np.nan)
    ndwi_p = np.where(valid_mask, ndwi, np.nan)

    print(f"[*] Đang lưu biểu đồ 4 chỉ số phổ năm {year} tại: {output_png}")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    def add_cbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(im, cax=cax)

    im1 = axes[0].imshow(ndvi_p, cmap='RdYlGn', vmin=float(np.nanmin(ndvi_p)), vmax=float(np.nanmax(ndvi_p)))
    axes[0].set_title(f"Chỉ số thực vật (NDVI) - Năm {year}", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    add_cbar(im1, axes[0])

    im2 = axes[1].imshow(ndwi_p, cmap='Blues', vmin=float(np.nanmin(ndwi_p)), vmax=float(np.nanmax(ndwi_p)))
    axes[1].set_title(f"Chỉ số nước (NDWI) - Năm {year}", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    add_cbar(im2, axes[1])

    im3 = axes[2].imshow(mndwi_p, cmap='Blues', vmin=float(np.nanmin(mndwi_p)), vmax=float(np.nanmax(mndwi_p)))
    axes[2].set_title(f"Chỉ số nước tăng cường (MNDWI) - Năm {year}", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    add_cbar(im3, axes[2])

    im4 = axes[3].imshow(ndbi_p, cmap='coolwarm', vmin=float(np.nanmin(ndbi_p)), vmax=float(np.nanmax(ndbi_p)))
    axes[3].set_title(f"Chỉ số xây dựng (NDBI) - Năm {year}", fontsize=11, fontweight='bold')
    axes[3].axis('off')
    add_cbar(im4, axes[3])

    plt.tight_layout(pad=1.5)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[+] Đã lưu biểu đồ chỉ số phổ: {output_png}")
    print(f"[*] Đang hiển thị cửa sổ biểu đồ 4 chỉ số phổ năm {year} (Hãy đóng cửa sổ hình ảnh Figure để tiếp tục)...")
    try:
        plt.show()
    except Exception as e:
        print(f"[!] Không thể mở cửa sổ hiển thị đồ họa: {e}")
    plt.close()


# ==============================================================================
# HÀM 3: TRÍCH XUẤT MẪU VÀ CÂN BẰNG MẪU TỪ SHAPEFILE
# ==============================================================================
def extract_and_balance_samples(stacked_img, transform, shape, crs, year):
    """Trích xuất giá trị 11 kênh tại các điểm mẫu và cân bằng số lượng mẫu."""
    cfg = DATASET_CONFIG[year]
    shp_path = cfg['shp']
    col = cfg['class_col']

    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Không tìm thấy file mẫu: {shp_path}")

    roi_gdf = gpd.read_file(shp_path)
    print(f"[*] Đọc thành công {len(roi_gdf)} mẫu từ: {shp_path}")

    if roi_gdf.crs != crs:
        roi_gdf = roi_gdf.to_crs(crs)

    if col not in roi_gdf.columns:
        for cand in ['Nhan', 'nhan', 'NHAN', 'label', 'Classvalue', 'Id']:
            if cand in roi_gdf.columns and roi_gdf[cand].nunique() > 1:
                col = cand
                break

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

    df_samples = pd.DataFrame(X)
    df_samples['label'] = y

    min_count = df_samples['label'].value_counts().min()
    n_sample = min(min_count, 1500)
    df_balanced = df_samples.groupby('label').sample(n=n_sample, random_state=42).reset_index(drop=True)

    X_balanced = df_balanced.drop('label', axis=1).values
    y_balanced = df_balanced['label'].values

    print(f"[+] Trích xuất {len(X)} pixel mẫu ban đầu -> Cân bằng ({n_sample} mẫu/lớp), tổng: {len(X_balanced)}")
    return X_balanced, y_balanced


# ==============================================================================
# HÀM 4: CHUẨN HÓA VÀ HUẤN LUYỆN MÔ HÌNH SVM
# ==============================================================================
def train_and_evaluate_svm(X, y, year):
    """Huấn luyện SVM (RBF kernel) với StandardScaler và đánh giá Train / Val (15%) / Test (15%)."""
    cfg = DATASET_CONFIG[year]
    print(f"\n[*] Đang huấn luyện mô hình SVM cho năm {year}...")

    # Chia 70% Train - 15% Val - 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[*] Phân chia tập dữ liệu: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # Chuẩn hóa đặc trưng bằng StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Đánh giá
    val_pred = svm.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    val_kappa = cohen_kappa_score(y_val, val_pred)

    test_pred = svm.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    test_kappa = cohen_kappa_score(y_test, test_pred)

    print("\n" + "-" * 55)
    print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH SVM NĂM {year}")
    print("-" * 55)
    print(f"-> Độ chính xác Validation (Accuracy):  {val_acc * 100:.2f}% | Kappa: {val_kappa:.4f}")
    print(f"-> Độ chính xác Test độc lập (Accuracy): {test_acc * 100:.2f}% | Kappa: {test_kappa:.4f}")

    target_names = [f"Lớp {c} ({cfg['classes'].get(c, 'Chưa rõ')})" for c in np.unique(y)]
    print(f"\n[+] BÁO CÁO PHÂN LOẠI CHI TIẾT (TEST SET SVM {year}):")
    print(classification_report(y_test, test_pred, target_names=target_names))

    print(f"[+] MA TRẬN NHẦM LẪN (TEST SET SVM {year}):")
    print(confusion_matrix(y_test, test_pred))

    # Lưu mô hình và scaler
    model_path = os.path.join(BASE_DIR, 'giscode', f'svm_model_{year}.pkl')
    scaler_path = os.path.join(BASE_DIR, 'giscode', f'svm_scaler_{year}.pkl')
    joblib.dump(svm, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[*] Đã lưu mô hình SVM: {model_path}")
    print(f"[*] Đã lưu Scaler: {scaler_path}")

    return svm, scaler, test_acc, test_kappa


# ==============================================================================
# HÀM 5: PHÂN LOẠI TOÀN BỘ RASTER, XUẤT GEOTIFF, BẢNG MÀU VÀ PREVIEW
# ==============================================================================
def classify_and_export(svm_model, scaler, data_dict, year):
    """Phân loại toàn bộ ảnh raster bằng SVM và xuất kết quả chuẩn GIS."""
    cfg = DATASET_CONFIG[year]
    stacked_img = data_dict['stacked']
    profile = data_dict['profile']
    valid_mask = data_dict['valid_mask']
    shape = data_dict['shape']

    n_features, h, w = stacked_img.shape
    classified_raster = np.full(shape, 127, dtype=np.int16)

    pixels = stacked_img.reshape(n_features, -1).T
    flat_valid = valid_mask.reshape(-1)

    print(f"[*] Đang phân loại SVM cho {np.sum(flat_valid):,} pixel hợp lệ năm {year}...")
    valid_pixels = pixels[flat_valid]
    valid_pixels_scaled = scaler.transform(valid_pixels)

    predictions = svm_model.predict(valid_pixels_scaled)
    classified_raster.reshape(-1)[flat_valid] = predictions

    # 1. Xuất GeoTIFF mã gốc
    out_tif = os.path.join(OUTPUT_DIR, f'anh_phan_loai_svm_{year}.tif')
    profile.update(
        dtype=rasterio.int16,
        count=1,
        nodata=127,
        compress='lzw'
    )

    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(classified_raster, 1)
    print(f"[+] ĐÃ XUẤT RASTER SVM NĂM {year}: {out_tif}")

    # 2. Xuất GeoTIFF chuẩn hóa 1-4
    out_std_tif = os.path.join(OUTPUT_DIR, f'anh_phan_loai_svm_{year}_std.tif')
    std_raster = np.full(shape, -9999, dtype=np.int16)
    for orig_c, std_c in cfg['std_mapping'].items():
        std_raster[classified_raster == orig_c] = std_c

    std_profile = profile.copy()
    std_profile.update(nodata=-9999)
    with rasterio.open(out_std_tif, 'w', **std_profile) as dst_std:
        dst_std.write(std_raster, 1)
    print(f"[+] ĐÃ XUẤT RASTER SVM CHUẨN HÓA CHO LULC: {out_std_tif}")

    # 3. Xuất bảng màu .clr
    out_clr = os.path.join(OUTPUT_DIR, f'anh_phan_loai_svm_{year}.clr')
    try:
        with open(out_clr, 'w', encoding='utf-8') as f_clr:
            for val, col in sorted(cfg['colors'].items()):
                f_clr.write(f"{val} {col[0]} {col[1]} {col[2]}\n")
            f_clr.write("127 255 255 255\n")
        print(f"[+] Đã tạo file bảng màu ArcMap: {out_clr}")
    except Exception as e:
        print(f"[!] Lỗi khi tạo .clr: {e}")

    # 4. Xuất ảnh preview .png
    out_png = os.path.join(OUTPUT_DIR, f'anh_phan_loai_svm_{year}_preview.png')
    try:
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_img[:, :] = [255, 255, 255]

        for val, col in cfg['colors'].items():
            rgb_img[classified_raster == val] = col

        img_pil = Image.fromarray(rgb_img)
        scale = max(1, 600 // max(h, w))
        if scale > 1:
            img_pil = img_pil.resize((w * scale, h * scale), Image.NEAREST)
        img_pil.save(out_png)
        print(f"[+] Đã tạo ảnh xem trước trực quan: {out_png}")
    except Exception as e:
        print(f"[!] Lỗi khi xuất preview: {e}")

    # 5. Thống kê diện tích
    print(f"\n--- THỐNG KÊ DIỆN TÍCH CÁC LỚP PHỦ NĂM {year} (SVM) ---")
    pixel_area_ha = (30.0 * 30.0) / 10000.0
    total_valid_pixels = np.sum(flat_valid)

    stats = []
    for c_id, c_name in cfg['classes'].items():
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
    print(f"[*] Tổng diện tích: {df_stats['Diện tích (ha)'].sum():.2f} ha ({total_valid_pixels:,} pixels)")


# ==============================================================================
# HÀM THỰC THI PIPELINE CHO MỘT NĂM CỤ THỂ
# ==============================================================================
def process_svm_year(year):
    """Chạy toàn bộ quy trình SVM hoàn chỉnh cho một năm (2015 hoặc 2026)."""
    print("=" * 80)
    print(f"    KHỞI CHẠY QUY TRÌNH PHÂN LOẠI SVM NĂM {year}")
    print("=" * 80)

    # 1. Đọc và tính chỉ số
    data_dict = load_and_preprocess(year)

    # 2. Xuất biểu đồ chỉ số phổ
    out_indices_png = os.path.join(OUTPUT_DIR, f'chi_so_pho_svm_{year}.png')
    plot_spectral_indices(data_dict['b_indices'], data_dict['valid_mask'], out_indices_png, year)

    # 3. Trích xuất mẫu
    X, y = extract_and_balance_samples(
        data_dict['stacked'], data_dict['transform'], data_dict['shape'], data_dict['crs'], year
    )

    # 4. Huấn luyện SVM
    svm_model, scaler, test_acc, test_kappa = train_and_evaluate_svm(X, y, year)

    # 5. Phân loại và xuất GeoTIFF
    classify_and_export(svm_model, scaler, data_dict, year)

    return test_acc, test_kappa


# ==============================================================================
# HÀM MAIN
# ==============================================================================
def main():
    print("*" * 80)
    print("*    CHƯƠNG TRÌNH PHÂN LOẠI ẢNH VIỄN THÁM SVM ĐỐI CHỨNG VỚI RANDOM FOREST    *")
    print("*" * 80)

    # Chạy lần lượt cả 2 năm 2015 và 2026 để đối sánh khoa học
    results = {}
    for yr in [2015, 2026]:
        acc, kappa = process_svm_year(yr)
        results[yr] = (acc, kappa)

    print("\n" + "=" * 80)
    print("TỔNG HỢP ĐỘ CHÍNH XÁC MÔ HÌNH SVM ĐÃ THỰC THI")
    print("=" * 80)
    for yr, (acc, kappa) in results.items():
        print(f" - Năm {yr}: Overall Accuracy = {acc*100:.2f}% | Cohen's Kappa = {kappa:.4f}")
    print("=" * 80)
    print("[HOÀN TẤT] Quy trình phân loại SVM đã hoàn thành thành công!")


if __name__ == "__main__":
    main()