# -*- coding: utf-8 -*-
"""
Quy trình phân loại ảnh viễn thám (GeoTIFF) bằng Random Forest với mẫu huấn luyện từ ArcMap (Shapefile).
Các thư viện cần thiết: rasterio, geopandas, shapely, scikit-learn, numpy, joblib
Cài đặt: pip install rasterio geopandas scikit-learn numpy joblib
"""

import os
import sys

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi font/mã hóa
if sys.stdout and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Tự động tạo lại file .shx nếu bị thiếu khi đọc Shapefile
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# CẤU HÌNH ĐƯỜNG DẪN FILE DỮ LIỆU CÓ SẴN TRONG MÁY
# Tự động định vị thư mục gốc dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Đường dẫn ảnh GeoTIFF đầu vào
IMAGE_PATH = os.path.join(BASE_DIR, 'data1', 'outExtractByMask1.tif')

# 2. Đường dẫn file Shapefile (.shp) vùng mẫu trích xuất / số hóa từ ArcMap
ROI_SHAPEFILE_PATH = os.path.join(BASE_DIR, 'DKT', 'DKT.shp')

# 3. Tên cột chứa mã loại / lớp phân loại trong bảng thuộc tính ArcMap
CLASS_COLUMN_NAME = "Nhan"

# 4. Đường dẫn lưu ảnh kết quả phân loại (GeoTIFF)
OUTPUT_RASTER_PATH = os.path.join(BASE_DIR, 'data1', 'anh_phan_loai_rf.tif')

# 5. Đường dẫn lưu mô hình Random Forest sau khi huấn luyện (tùy chọn)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'rf_model.pkl')


# HÀM BỔ TRỢ: TÍNH TOÁN 3 CHỈ SỐ CHUYÊN SÊU
def add_spectral_indices(pixels):
    """
    Tính toán 3 chỉ số NDVI, NDWI, NDBI từ mảng 7 kênh phổ, ghép lại thành mảng 10 kênh.
    pixels: numpy array có shape (N, 7)
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Giả định thứ tự kênh: 0=B1, 1=B2, 2=B3(Green), 3=B4(Red), 4=B5(NIR), 5=B6(SWIR1), 6=B7(SWIR2)
        b3 = pixels[:, 2].astype(float)
        b4 = pixels[:, 3].astype(float)
        b5 = pixels[:, 4].astype(float)
        b6 = pixels[:, 5].astype(float)

        ndvi = np.nan_to_num((b5 - b4) / (b5 + b4), nan=0.0)
        ndwi = np.nan_to_num((b3 - b5) / (b3 + b5), nan=0.0)
        ndbi = np.nan_to_num((b6 - b5) / (b6 + b5), nan=0.0)

        # Nối 3 chỉ số vào cuối mảng gốc -> tạo mảng (N, 10)
        return np.column_stack((pixels, ndvi, ndwi, ndbi))

# BƯỚC 1: TRÍCH XUẤT MẪU ĐẶC TRƯNG TỪ SHAPEFILE VÀ ẢNH TIF
def extract_training_data(image_path, roi_path, class_col):
    """
    Trích xuất giá trị phổ (pixel values) từ các mẫu POINT (Điểm) hoặc POLYGON (Vùng đa giác).
    """
    print(f"[*] Đang đọc ảnh viễn thám: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}\n--> Vui lòng copy file ảnh .tif vào thư mục và cập nhật IMAGE_PATH!")
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"Không tìm thấy file Shapefile: {roi_path}")

    # Đọc Shapefile mẫu tạo từ ArcMap
    print(f"[*] Đang đọc Shapefile mẫu huấn luyện: {roi_path}")
    roi_gdf = gpd.read_file(roi_path)

    # Ưu tiên sử dụng class_col được truyền vào nếu có > 1 lớp
    if class_col in roi_gdf.columns and roi_gdf[class_col].nunique() > 1:
        pass # Hợp lệ, giữ nguyên
    else:
        # Tự động dò tìm cột nhãn phù hợp
        for candidate in [class_col, "Nhan", "nhan", "NHAN", "label", "Classvalue", "Classname", "Id"]:
            if candidate in roi_gdf.columns and roi_gdf[candidate].nunique() > 1:
                print(f"[*] Tự động chuyển sang sử dụng cột nhãn: '{candidate}'")
                class_col = candidate
                break
        else:
            raise ValueError(f"Không tìm thấy cột nào chứa >1 lớp phân loại. Các cột hiện có: {list(roi_gdf.columns)}")

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
    X = add_spectral_indices(X) # Tính thêm 3 chỉ số NDVI, NDWI, NDBI (Tổng cộng 10 kênh)
    y = np.concatenate(y_list)

    print(f"[+] Trích xuất thành công {X.shape[0]} điểm mẫu với {X.shape[1]} kênh phổ.")
    unique_classes, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(unique_classes, counts):
        print(f"    - Lớp {c}: {cnt} pixels")

    return X, y


# BƯỚC 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ THUẬT TOÁN RANDOM FOREST
def train_random_forest(X, y, n_estimators=100, max_depth=None, test_size=0.2, random_state=42):
    """
    Chia tập train/test, huấn luyện Random Forest và in báo cáo độ chính xác.
    """
    print("\n[*] Đang chia tập dữ liệu Train / Val / Test (70% - 15% - 15%)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    print(f"[*] Phân bổ: Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Chuẩn hóa dữ liệu (StandardScaler) giống hệt như trong test.py
    print("[*] Đang chuẩn hóa dữ liệu (StandardScaler)...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"[*] Đang huấn luyện Random Forest ({n_estimators} cây)...")
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,  # Sử dụng tối đa tất cả các nhân CPU để tăng tốc
        random_state=random_state,
        oob_score=True
    )
    rf_clf.fit(X_train, y_train)

    # Đánh giá trên tập Validation
    val_pred = rf_clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    # Đánh giá trên tập kiểm tra độc lập (Test)
    y_pred = rf_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST")
    print("="*50)
    print(f"Độ chính xác trên tập Validation (15%): {val_acc * 100:.2f}%")
    print(f"Độ chính xác tổng thể trên tập Test (15%): {acc * 100:.2f}%")
    print(f"Hệ số Kappa trên tập Test: {kappa:.4f}")
    if hasattr(rf_clf, "oob_score_"):
        print(f"Điểm Out-Of-Bag (OOB Score - từ tập Train): {rf_clf.oob_score_ * 100:.2f}%")

    print("\nBáo cáo chi tiết từng lớp (Classification Report - Test set):")
    print(classification_report(y_test, y_pred))

    print("Ma trận nhầm lẫn (Confusion Matrix - Test set):")
    print(confusion_matrix(y_test, y_pred))

    return rf_clf, scaler

# BƯỚC 3: PHÂN LOẠI TOÀN BỘ ẢNH TIF VÀ XUẤT FILE GEOTIFF
def classify_and_export_raster(image_path, model, scaler, output_path, block_size=1024):
    """
    Phân loại toàn bộ ảnh GeoTIFF theo từng khối (block) để tránh tràn bộ nhớ RAM (OOM)
    và xuất ra file GeoTIFF mới giữ nguyên tọa độ và hệ quy chiếu (CRS).
    """
    print(f"\n[*] Bắt đầu phân loại toàn bộ ảnh: {image_path}")
    print(f"[*] File kết quả sẽ được lưu tại: {output_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        
        # Cập nhật thông số metadata cho file kết quả (1 kênh phân loại)
        profile.update(
            dtype=rasterio.int32,
            count=1,
            nodata=-9999,
            compress='lzw'
        )

        n_bands = src.count
        height = src.height
        width = src.width

        with rasterio.open(output_path, 'w', **profile) as dst:
            # Duyệt theo từng khối (block) ngang và dọc
            for row in range(0, height, block_size):
                row_end = min(row + block_size, height)
                h_chunk = row_end - row

                for col in range(0, width, block_size):
                    col_end = min(col + block_size, width)
                    w_chunk = col_end - col

                    window = rasterio.windows.Window(col, row, w_chunk, h_chunk)
                    
                    # Đọc dữ liệu khối: shape (bands, h_chunk, w_chunk)
                    block_data = src.read(window=window)
                    
                    # Chuyển đổi thành dạng (h_chunk * w_chunk, bands)
                    reshaped = block_data.reshape(n_bands, -1).T

                    # Kiểm tra pixel NoData / pixel rỗng
                    if src.nodata is not None:
                        valid_mask = (reshaped != src.nodata).all(axis=1)
                    else:
                        valid_mask = (reshaped != 0).any(axis=1)

                    # Tạo mảng kết quả mặc định là NoData
                    classified_block = np.full(reshaped.shape[0], -9999, dtype=np.int32)

                    if np.any(valid_mask):
                        # Lấy các pixel hợp lệ và tính thêm 3 chỉ số (7 kênh -> 10 kênh)
                        valid_pixels = reshaped[valid_mask]
                        valid_pixels_enhanced = add_spectral_indices(valid_pixels)
                        # Chuẩn hóa (Scale) giống hệt lúc huấn luyện
                        valid_pixels_enhanced = scaler.transform(valid_pixels_enhanced)
                        # Dự đoán nhãn
                        classified_block[valid_mask] = model.predict(valid_pixels_enhanced)

                    # Định hình lại thành (1, h_chunk, w_chunk) và ghi vào file GeoTIFF
                    classified_block = classified_block.reshape((1, h_chunk, w_chunk))
                    dst.write(classified_block, window=window)

    print(f"[+] Hoàn tất! Ảnh phân loại đã được lưu thành công tại: {output_path}")

    # Tự động xuất ảnh màu PNG để xem trực quan ngay trên Paint / Photo Viewer
    export_preview_image(output_path)


def export_preview_image(raster_path, png_path=None):
    """
    Xuất ảnh màu RGB (.png) để xem trực quan và tạo file .clr tương thích với ArcMap.
    """
    if png_path is None:
        png_path = os.path.splitext(raster_path)[0] + "_preview.png"

    # Bảng màu đại diện cho các lớp chuẩn theo ArcMap
    # 1: Rừng (xanh lá chuối / tươi), 7: Thủy hệ (xanh dương), 10: Dân cư (tím hồng), 14: Đất trống (vàng nhạt)
    color_map = {
        1: [76, 230, 0],       # Rừng (Xanh lá sáng / Lime Green)
        7: [0, 112, 255],      # Thủy hệ (Xanh dương)
        10: [223, 115, 255],   # Dân cư (Tím hồng / Orchid - giống ArcMap)
        14: [255, 255, 115],   # Đất trống (Vàng chanh)
    }

    # 1. Tự động tạo file bảng màu .clr để ArcMap tự nhận đúng màu khi mở file TIF
    clr_path = os.path.splitext(raster_path)[0] + ".clr"
    try:
        with open(clr_path, "w", encoding="utf-8") as f_clr:
            for val, col in sorted(color_map.items()):
                f_clr.write(f"{val} {col[0]} {col[1]} {col[2]}\n")
        print(f"[+] Đã tạo file bảng màu ArcMap (.clr) tại: {clr_path}")
    except Exception as e:
        print(f"[!] Không thể tạo file .clr: {e}")

    # 2. Xuất ảnh màu xem trước .png
    try:
        from PIL import Image
        with rasterio.open(raster_path) as src:
            data = src.read(1)

        h, w = data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :] = [255, 255, 255]  # Nền ngoài ranh giới màu trắng sạch sẽ

        for val, col in color_map.items():
            rgb[data == val] = col

        img = Image.fromarray(rgb)
        # Phóng to theo tỷ lệ phù hợp để xem rõ pixel
        scale = max(1, 600 // max(h, w))
        if scale > 1:
            img = img.resize((w * scale, h * scale), Image.NEAREST)

        img.save(png_path)
        print(f"[+] Đã tạo ảnh màu xem trước trực quan tại: {png_path}")
    except Exception as e:
        print(f"[!] Không thể xuất ảnh xem trước: {e}")


# HÀM THỰC THI CHÍNH
def main():
    print("----------------------------------------------------")
    print("    QUY TRÌNH PHÂN LOẠI RANDOM FOREST ẢNH VIỄN THÁM")
    print("----------------------------------------------------")

    # 1. Trích xuất mẫu từ ảnh và Shapefile
    X, y = extract_training_data(
        image_path=IMAGE_PATH,
        roi_path=ROI_SHAPEFILE_PATH,
        class_col=CLASS_COLUMN_NAME
    )

    # 2. Huấn luyện Random Forest và đánh giá độ chính xác
    rf_model, scaler = train_random_forest(
        X=X,
        y=y,
        n_estimators=100,
        test_size=0.2,
        random_state=42
    )

    # 3. Lưu mô hình đã huấn luyện (.pkl)
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_SAVE_PATH)), exist_ok=True)
    joblib.dump(rf_model, MODEL_SAVE_PATH)
    print(f"[*] Đã lưu mô hình Random Forest tại: {MODEL_SAVE_PATH}")

    # 4. Phân loại toàn bộ ảnh và xuất file GeoTIFF
    classify_and_export_raster(
        image_path=IMAGE_PATH,
        model=rf_model,
        scaler=scaler,
        output_path=OUTPUT_RASTER_PATH,
        block_size=1024
    )


if __name__ == "__main__":
    main()
