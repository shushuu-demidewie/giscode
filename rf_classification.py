# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH PHÂN LOẠI ẢNH VIỄN THÁM (GEOTIFF) BẰNG RANDOM FOREST
Hỗ trợ dữ liệu năm 2015 và 2026 với mẫu huấn luyện từ ArcMap (Shapefile)
================================================================================
Các chức năng chính:
  1. Tự động liên kết các đường dẫn dữ liệu chuẩn xác trong dự án (d:\\testgis):
     - Dữ liệu năm 2015: data/dp2015Extrac.tif, Dkt2015/DKT2015.shp, data/ketquatach.shp
     - Dữ liệu năm 2026: data/dp2026Extrac.tif, DKT2026/DKT2026.shp, data/ketquatach.shp
  2. Trích xuất giá trị phổ tại các điểm mẫu và huấn luyện Random Forest.
  3. Đánh giá độ chính xác độc lập (Train 70% - Validation 15% - Test 15%).
  4. Phân loại toàn bộ ảnh theo khối (block processing), cắt theo mặt nạ ranh giới xã.
  5. Xuất kết quả GeoTIFF (.tif), bảng màu ArcMap (.clr), ảnh xem trước (.png) và thống kê diện tích (ha).
================================================================================
"""

import os
import sys

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi font/mã hóa
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
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import joblib
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU CHUẨN XÁC TRONG DỰ ÁN (d:\testgis)
# ==============================================================================
# Định vị thư mục gốc dự án (d:\testgis)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bảng cấu hình chi tiết cho từng năm dữ liệu
DATASET_CONFIG = {
    2015: {
        'name': "Năm 2015",
        # 1. Đường dẫn ảnh GeoTIFF đầu vào (ảnh 7 kênh đã trích xuất theo khu vực nghiên cứu)
        'raster': os.path.join(BASE_DIR, 'data', 'dp2015Extrac.tif'),
        'raw_folder': os.path.join(BASE_DIR, 'LC8_HaNoi_2015'),
        # 2. File ranh giới xã nghiên cứu
        'boundary_shp': os.path.join(BASE_DIR, 'data', 'ketquatach.shp'),
        # 3. File Shapefile điểm mẫu huấn luyện số hóa từ ArcMap
        'roi_shp': os.path.join(BASE_DIR, 'Dkt2015', 'DKT2015.shp'),
        'class_col': "Nhan",
        # 4. Danh mục 4 lớp phủ mặt đất thực tế trong DKT2015
        'classes': {
            1: "Đất mặt nước",
            10: "Đất nông nghiệp / Đất trống",
            14: "Đất dân cư và xây dựng",
            22: "Thảm thực vật"
        },
        # 5. Bảng màu hiển thị chuẩn theo hệ thống ArcMap (RGB)
        'colors': {
            1: [0, 92, 230],       # Mặt nước (Xanh dương)
            10: [255, 255, 115],   # Đất nông nghiệp / trống (Vàng)
            14: [230, 0, 0],       # Dân cư / Xây dựng (Đỏ)
            22: [38, 115, 0],      # Thảm thực vật (Xanh lá)
        },
        # 6. File đầu ra
        'output_raster': os.path.join(OUTPUT_DIR, 'anh_phan_loai_2015.tif'),
        'output_model': os.path.join(BASE_DIR, 'giscode', 'rf_model_2015.pkl')
    },
    2026: {
        'name': "Năm 2026",
        # 1. Đường dẫn ảnh GeoTIFF đầu vào (ảnh 7 kênh đã trích xuất theo khu vực nghiên cứu)
        'raster': os.path.join(BASE_DIR, 'data', 'dp2026Extrac.tif'),
        'raw_folder': os.path.join(BASE_DIR, 'LC8_HaNoi_2026'),
        # 2. File ranh giới xã nghiên cứu
        'boundary_shp': os.path.join(BASE_DIR, 'data', 'ketquatach.shp'),
        # 3. File Shapefile điểm mẫu huấn luyện số hóa từ ArcMap
        'roi_shp': os.path.join(BASE_DIR, 'DKT2026', 'DKT2026.shp'),
        'class_col': "Nhan",
        # 4. Danh mục 4 lớp phủ mặt đất thực tế trong DKT2026
        'classes': {
            1: "Đất mặt nước",
            10: "Đất nông nghiệp / Đất trống",
            13: "Đất dân cư và xây dựng",
            19: "Thảm thực vật"
        },
        # 5. Bảng màu hiển thị chuẩn theo hệ thống ArcMap (RGB)
        'colors': {
            1: [0, 92, 230],       # Mặt nước (Xanh dương)
            10: [255, 255, 115],   # Đất nông nghiệp / trống (Vàng)
            13: [230, 0, 0],       # Dân cư / Xây dựng (Đỏ)
            19: [38, 115, 0],      # Thảm thực vật (Xanh lá)
        },
        # 6. File đầu ra
        'output_raster': os.path.join(OUTPUT_DIR, 'anh_phan_loai_2026.tif'),
        'output_model': os.path.join(BASE_DIR, 'giscode', 'rf_model_2026.pkl')
    }
}

# ------------------------------------------------------------------------------
# LỰA CHỌN NĂM CẦN CHẠY PHÂN LOẠI
# Người dùng có thể đổi TARGET_YEAR thành: 2015, 2026, hoặc 'ALL' để chạy cả hai năm
# Hoặc truyền qua dòng lệnh: python rf_classification.py 2015
# ------------------------------------------------------------------------------
TARGET_YEAR = 2015  # Mặc định: 2015 (hoặc 2026, hoặc 'ALL')

# Các biến đường dẫn tương thích ngược mặc định trỏ theo TARGET_YEAR
ACTIVE_CFG = DATASET_CONFIG.get(TARGET_YEAR if TARGET_YEAR in DATASET_CONFIG else 2015)
IMAGE_PATH = ACTIVE_CFG['raster']
ROI_SHAPEFILE_PATH = ACTIVE_CFG['roi_shp']
BOUNDARY_SHP_PATH = ACTIVE_CFG['boundary_shp']
CLASS_COLUMN_NAME = ACTIVE_CFG['class_col']
OUTPUT_RASTER_PATH = ACTIVE_CFG['output_raster']
MODEL_SAVE_PATH = ACTIVE_CFG['output_model']
CLASS_NAMES = ACTIVE_CFG['classes']
CLASS_COLORS = ACTIVE_CFG['colors']


# ==============================================================================
# BƯỚC 1: TRÍCH XUẤT MẪU ĐẶC TRƯNG TỪ SHAPEFILE VÀ ẢNH TIF
# ==============================================================================
def extract_training_data(image_path, roi_path, class_col):
    """
    Trích xuất giá trị phổ (pixel values) từ các mẫu POINT (Điểm) hoặc POLYGON (Vùng đa giác).
    """
    print(f"[*] Đang đọc ảnh viễn thám: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}\n--> Vui lòng kiểm tra lại đường dẫn!")
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"Không tìm thấy file Shapefile mẫu: {roi_path}")

    # Đọc Shapefile mẫu tạo từ ArcMap
    print(f"[*] Đang đọc Shapefile mẫu huấn luyện: {roi_path}")
    roi_gdf = gpd.read_file(roi_path)

    if class_col not in roi_gdf.columns:
        for candidate in ["Nhan", "nhan", "NHAN", "label", "Classvalue", "Classname", "Id"]:
            if candidate in roi_gdf.columns and roi_gdf[candidate].nunique() > 1:
                print(f"[*] Cột '{class_col}' không tồn tại, tự động chuyển sang cột: '{candidate}'")
                class_col = candidate
                break
        else:
            raise ValueError(f"Không tìm thấy cột '{class_col}' trong Shapefile. Các cột hiện có: {list(roi_gdf.columns)}")

    X_list = []
    y_list = []

    with rasterio.open(image_path) as src:
        # Tự động đồng bộ hệ tọa độ (CRS) giữa Shapefile và ảnh nếu khác nhau
        if roi_gdf.crs != src.crs:
            print(f"[*] Chuyển đổi hệ quy chiếu của Shapefile ({roi_gdf.crs}) sang của ảnh ({src.crs})...")
            roi_gdf = roi_gdf.to_crs(src.crs)

        geom_types = roi_gdf.geometry.geom_type.unique()
        print(f"[*] Kiểu hình học của mẫu (Geometry Types): {geom_types}")

        # Trường hợp 1: Dữ liệu mẫu là dạng POINT (Điểm)
        if any("Point" in gt for gt in geom_types):
            points = [(geom.x, geom.y) for geom in roi_gdf.geometry if geom.geom_type in ["Point", "MultiPoint"]]
            labels = [row[class_col] for idx, row in roi_gdf.iterrows() if row["geometry"].geom_type in ["Point", "MultiPoint"]]

            # Lấy mẫu giá trị pixel trực tiếp tại tọa độ điểm
            sampled_values = list(src.sample(points))
            sampled_values = np.array(sampled_values)

            # Lọc bỏ giá trị NoData
            if src.nodata is not None:
                valid_mask = (sampled_values != src.nodata).all(axis=1)
            else:
                valid_mask = (sampled_values != 0).any(axis=1)

            X_list.append(sampled_values[valid_mask])
            y_list.append(np.array(labels)[valid_mask])

        # Trường hợp 2: Dữ liệu mẫu là dạng POLYGON (Vùng đa giác)
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
                        X_list.append(valid_pixels)
                        y_list.append(np.full(len(valid_pixels), class_label))
                except Exception as e:
                    print(f"[!] Cảnh báo: Bỏ qua vùng mẫu số {idx} do lỗi: {e}")

    if not X_list:
        raise ValueError("Không trích xuất được pixel mẫu nào! Hãy kiểm tra lại độ trùng khớp tọa độ giữa ảnh và shapefile.")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print(f"[+] Trích xuất thành công {X.shape[0]} điểm mẫu với {X.shape[1]} kênh phổ.")
    unique_classes, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(unique_classes, counts):
        c_name = CLASS_NAMES.get(c, f"Lớp {c}")
        print(f"    - Mã {c} ({c_name}): {cnt} điểm mẫu")

    return X, y


# ==============================================================================
# BƯỚC 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ THUẬT TOÁN RANDOM FOREST
# ==============================================================================
def train_random_forest(X, y, n_estimators=100, max_depth=None, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Chia tập dữ liệu theo tỷ lệ Train / Val / Test (mặc định 70% / 15% / 15%),
    huấn luyện Random Forest và đánh giá mô hình trên cả 2 tập Validation và Test.
    """
    print(f"\n[*] Đang chia tập dữ liệu ({int(train_ratio*100)}% Train - {int(val_ratio*100)}% Val - {int(test_ratio*100)}% Test)...")

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

    print(f"\n[*] Đang huấn luyện Random Forest ({n_estimators} cây)...")
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        oob_score=True
    )
    rf_clf.fit(X_train, y_train)

    # 1. Đánh giá trên tập Validation
    y_val_pred = rf_clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_kappa = cohen_kappa_score(y_val, y_val_pred)
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP VALIDATION")
    print("=" * 60)
    print(f"Độ chính xác Validation (Overall Accuracy): {val_acc * 100:.2f}%")
    print(f"Hệ số Cohen's Kappa Validation:           {val_kappa:.4f}")
    print("\nBáo cáo chi tiết tập Validation (Classification Report):")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    print("Ma trận nhầm lẫn Validation (Confusion Matrix):")
    print(confusion_matrix(y_val, y_val_pred))

    # 2. Đánh giá trên tập Test độc lập
    y_test_pred = rf_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_kappa = cohen_kappa_score(y_test, y_test_pred)
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (KIỂM TRA ĐỘC LẬP)")
    print("=" * 60)
    print(f"Độ chính xác tổng thể Test (Overall Accuracy): {test_acc * 100:.2f}%")
    print(f"Hệ số Cohen's Kappa tập Test:                {test_kappa:.4f}")
    if hasattr(rf_clf, "oob_score_"):
        print(f"Điểm Out-Of-Bag (OOB Score):                 {rf_clf.oob_score_ * 100:.2f}%")

    print("\nBáo cáo chi tiết tập Test (Classification Report):")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    print("Ma trận nhầm lẫn tập Test (Confusion Matrix):")
    print(confusion_matrix(y_test, y_test_pred))

    return rf_clf


# ==============================================================================
# BƯỚC 3: PHÂN LOẠI TOÀN BỘ ẢNH TIF VÀ XUẤT FILE GEOTIFF
# ==============================================================================
def classify_and_export_raster(image_path, model, output_path, boundary_shp=None, class_colors=None, class_names=None, block_size=1024):
    """
    Phân loại toàn bộ ảnh GeoTIFF theo từng khối (block) để tránh tràn bộ nhớ RAM,
    áp dụng mặt nạ ranh giới xã (NoData = 127) và xuất file GeoTIFF chuẩn.
    """
    print(f"\n[*] Bắt đầu phân loại toàn bộ ảnh: {image_path}")
    print(f"[*] File kết quả sẽ được lưu tại: {output_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        height = src.height
        width = src.width
        n_bands = src.count

        # Tạo mặt nạ ranh giới xã từ file Shapefile nếu có
        boundary_mask = None
        if boundary_shp and os.path.exists(boundary_shp):
            print(f"[*] Áp dụng mặt nạ ranh giới xã từ: {boundary_shp}")
            try:
                bgdf = gpd.read_file(boundary_shp)
                if bgdf.crs != crs:
                    bgdf = bgdf.to_crs(crs)
                # True = nằm ngoài ranh giới
                boundary_mask = geometry_mask(bgdf.geometry, out_shape=(height, width), transform=transform, invert=False)
            except Exception as e:
                print(f"[!] Cảnh báo: Không thể tạo mặt nạ ranh giới: {e}")
                boundary_mask = None

        # Cập nhật thông số metadata cho file kết quả (1 kênh phân loại)
        # Giá trị NoData 127 tương thích chuẩn ArcMap (pl_T2015v.tif / pl_T2026.tif)
        nodata_out = 127
        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=nodata_out,
            compress='lzw'
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            for row in range(0, height, block_size):
                row_end = min(row + block_size, height)
                h_chunk = row_end - row

                for col in range(0, width, block_size):
                    col_end = min(col + block_size, width)
                    w_chunk = col_end - col

                    window = rasterio.windows.Window(col, row, w_chunk, h_chunk)
                    block_data = src.read(window=window)
                    reshaped = block_data.reshape(n_bands, -1).T

                    # Kiểm tra pixel NoData đầu vào
                    if src.nodata is not None:
                        valid_mask = (reshaped != src.nodata).all(axis=1)
                    else:
                        valid_mask = (reshaped != 0).any(axis=1)

                    # Áp dụng mặt nạ ranh giới xã nếu có
                    if boundary_mask is not None:
                        sub_mask = boundary_mask[row:row_end, col:col_end].ravel()
                        valid_mask = valid_mask & (~sub_mask)

                    classified_block = np.full(reshaped.shape[0], nodata_out, dtype=np.int16)

                    if np.any(valid_mask):
                        classified_block[valid_mask] = model.predict(reshaped[valid_mask])

                    classified_block = classified_block.reshape((1, h_chunk, w_chunk))
                    dst.write(classified_block, window=window)

    print(f"[+] Hoàn tất! Ảnh phân loại đã được lưu thành công tại: {output_path}")

    # Xuất ảnh màu PNG và file bảng màu ArcMap .clr
    export_preview_and_stats(output_path, class_colors=class_colors, class_names=class_names, nodata_val=nodata_out)


# ==============================================================================
# BƯỚC 4: XUẤT ẢNH XEM TRƯỚC (.PNG), BẢNG MÀU ARCMAP (.CLR) & THỐNG KÊ DIỆN TÍCH
# ==============================================================================
def export_preview_and_stats(raster_path, class_colors=None, class_names=None, nodata_val=127):
    """
    Xuất ảnh màu RGB (.png) để xem trực quan, tạo file .clr tương thích với ArcMap,
    và thống kê diện tích từng lớp (pixel và hecta).
    """
    if class_colors is None:
        class_colors = CLASS_COLORS
    if class_names is None:
        class_names = CLASS_NAMES

    # 1. Tạo file bảng màu ArcMap (.clr)
    clr_path = os.path.splitext(raster_path)[0] + ".clr"
    try:
        with open(clr_path, "w", encoding="utf-8") as f_clr:
            for val, col in sorted(class_colors.items()):
                f_clr.write(f"{val} {col[0]} {col[1]} {col[2]}\n")
            f_clr.write(f"{nodata_val} 255 255 255\n")
        print(f"[+] Đã tạo file bảng màu ArcMap (.clr) tại: {clr_path}")
    except Exception as e:
        print(f"[!] Không thể tạo file .clr: {e}")

    # 2. Xuất ảnh màu xem trước .png và thống kê diện tích
    png_path = os.path.splitext(raster_path)[0] + "_preview.png"
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            res_x, res_y = abs(src.res[0]), abs(src.res[1])
            pixel_area_ha = (res_x * res_y) / 10000.0

        h, w = data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :] = [255, 255, 255]  # Nền ngoài ranh giới màu trắng

        print("\n" + "=" * 60)
        print("THỐNG KÊ DIỆN TÍCH CÁC LỚP PHỦ SAU PHÂN LOẠI")
        print("=" * 60)
        print(f"{'Mã':<6}{'Tên lớp phủ':<30}{'Số Pixel':<12}{'Diện tích (ha)':<15}{'Tỷ lệ (%)'}")
        print("-" * 70)

        valid_total = np.sum((data != nodata_val) & (data > 0))

        for val, col in class_colors.items():
            mask_val = (data == val)
            count = np.sum(mask_val)
            rgb[mask_val] = col
            c_name = class_names.get(val, f"Lớp {val}")
            area_ha = count * pixel_area_ha
            pct = (count / valid_total * 100) if valid_total > 0 else 0
            print(f"{val:<6}{c_name:<30}{count:<12}{area_ha:<15.2f}{pct:.2f}%")

        print("-" * 70)
        total_ha = valid_total * pixel_area_ha
        print(f"{'TỔNG':<36}{valid_total:<12}{total_ha:<15.2f}100.00%")
        print("=" * 60)

        if HAS_PIL:
            img = Image.fromarray(rgb)
            scale = max(1, 600 // max(h, w))
            if scale > 1:
                img = img.resize((w * scale, h * scale), Image.NEAREST)

            img.save(png_path)
            print(f"[+] Đã tạo ảnh xem trước trực quan tại: {png_path}")
        else:
            print("[*] Bỏ qua xuất ảnh preview .png (Cần thư viện pillow nếu muốn xuất ảnh PNG).")
    except Exception as e:
        print(f"[!] Không thể xuất ảnh xem trước: {e}")


# ==============================================================================
# HÀM THỰC THI CHO MỘT NĂM CỤ THỂ
# ==============================================================================
def run_classification_for_year(year):
    if year not in DATASET_CONFIG:
        raise ValueError(f"Năm không hợp lệ: {year}. Chỉ hỗ trợ: {list(DATASET_CONFIG.keys())}")

    cfg = DATASET_CONFIG[year]
    print("\n" + "#" * 70)
    print(f"# TIẾN TRÌNH PHÂN LOẠI RANDOM FOREST - {cfg['name'].upper()}")
    print("#" * 70)
    print(f"[*] Ảnh đầu vào:   {cfg['raster']}")
    print(f"[*] Shapefile mẫu: {cfg['roi_shp']}")
    print(f"[*] Ranh giới xã:  {cfg['boundary_shp']}")
    print(f"[*] File kết quả:  {cfg['output_raster']}")
    print(f"[*] Model lưu tại: {cfg['output_model']}")
    print("-" * 70)

    # 1. Trích xuất mẫu
    X, y = extract_training_data(
        image_path=cfg['raster'],
        roi_path=cfg['roi_shp'],
        class_col=cfg['class_col']
    )

    # 2. Huấn luyện Random Forest
    rf_model = train_random_forest(
        X=X,
        y=y,
        n_estimators=100,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    # 3. Lưu mô hình .pkl
    os.makedirs(os.path.dirname(os.path.abspath(cfg['output_model'])), exist_ok=True)
    joblib.dump(rf_model, cfg['output_model'])
    print(f"[*] Đã lưu mô hình Random Forest tại: {cfg['output_model']}")

    # 4. Phân loại và xuất GeoTIFF
    classify_and_export_raster(
        image_path=cfg['raster'],
        model=rf_model,
        output_path=cfg['output_raster'],
        boundary_shp=cfg['boundary_shp'],
        class_colors=cfg['colors'],
        class_names=cfg['classes'],
        block_size=1024
    )


# ==============================================================================
# HÀM THỰC THI CHÍNH
# ==============================================================================
def main():
    print("=" * 70)
    print("    CHƯƠNG TRÌNH PHÂN LOẠI LỚP PHỦ BẰNG RANDOM FOREST")
    print("=" * 70)

    # Kiểm tra nếu người dùng truyền tham số dòng lệnh: python rf_classification.py 2015/2026/ALL
    selected_year = TARGET_YEAR
    if len(sys.argv) > 1:
        arg_val = sys.argv[1].strip().upper()
        if arg_val in ['2015', '2026']:
            selected_year = int(arg_val)
        elif arg_val == 'ALL':
            selected_year = 'ALL'

    if selected_year == 'ALL':
        print("[*] Chế độ thực thi: Phân loại cả 2 mốc thời gian 2015 và 2026...")
        run_classification_for_year(2015)
        run_classification_for_year(2026)
    else:
        run_classification_for_year(selected_year)

    print("\n[HOÀN TẤT] Toàn bộ quy trình phân loại Random Forest đã hoàn thành thành công!")


if __name__ == "__main__":
    main()
