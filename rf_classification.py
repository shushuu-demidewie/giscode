# -*- coding: utf-8 -*-
"""
Quy trình phân loại ảnh viễn thám (GeoTIFF) bằng Random Forest với mẫu huấn luyện từ ArcMap (Shapefile).
Các thư viện cần thiết: rasterio, geopandas, shapely, scikit-learn, numpy, joblib
Cài đặt: pip install rasterio geopandas scikit-learn numpy joblib
"""

import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN FILE (ĐIỀN ĐƯỜNG DẪN CỦA BẠN VÀO ĐÂY)
# ==============================================================================
# 1. Đường dẫn ảnh GeoTIFF đầu vào (ảnh Landsat 7 kênh cắt từ khu vực Hà Nội)
IMAGE_PATH = r"c:\ANH_VIEN_THAM\test1\LC8_HANOI_2026\anhcat.tif"

# 2. Đường dẫn file Shapefile (.shp) vùng mẫu trích xuất / số hóa từ ArcMap
ROI_SHAPEFILE_PATH = r"c:\ANH_VIEN_THAM\test1\DKT36.shp"

# 3. Tên cột chứa mã loại / lớp phân loại trong bảng thuộc tính ArcMap (trong file DKT36 là 'Nhan')
CLASS_COLUMN_NAME = "Nhan"

# 4. Đường dẫn lưu ảnh kết quả phân loại (GeoTIFF)
OUTPUT_RASTER_PATH = r"c:\ANH_VIEN_THAM\test1\anh_phan_loai_rf.tif"

# 5. Đường dẫn lưu mô hình Random Forest sau khi huấn luyện (tùy chọn)
MODEL_SAVE_PATH = r"c:\ANH_VIEN_THAM\test1\rf_model.pkl"


# ==============================================================================
# BƯỚC 1: TRÍCH XUẤT MẪU ĐẶC TRƯNG TỪ SHAPEFILE VÀ ẢNH TIF
# ==============================================================================
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

    if class_col not in roi_gdf.columns:
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
        print(f"    - Lớp {c}: {cnt} pixels")

    return X, y


# ==============================================================================
# BƯỚC 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ THUẬT TOÁN RANDOM FOREST
# ==============================================================================
def train_random_forest(X, y, n_estimators=100, max_depth=None, test_size=0.2, random_state=42):
    """
    Chia tập train/test, huấn luyện Random Forest và in báo cáo độ chính xác.
    """
    print("\n[*] Đang chia tập dữ liệu Train / Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[*] Đang huấn luyện Random Forest ({n_estimators} cây)...")
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,  # Sử dụng tối đa tất cả các nhân CPU để tăng tốc
        random_state=random_state,
        oob_score=True
    )
    rf_clf.fit(X_train, y_train)

    # Đánh giá trên tập kiểm tra độc lập
    y_pred = rf_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST")
    print("="*50)
    print(f"Độ chính xác tổng thể (Overall Accuracy): {acc * 100:.2f}%")
    if hasattr(rf_clf, "oob_score_"):
        print(f"Điểm Out-Of-Bag (OOB Score): {rf_clf.oob_score_ * 100:.2f}%")

    print("\nBáo cáo chi tiết từng lớp (Classification Report):")
    print(classification_report(y_test, y_pred))

    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

    return rf_clf


# ==============================================================================
# BƯỚC 3: PHÂN LOẠI TOÀN BỘ ẢNH TIF VÀ XUẤT FILE GEOTIFF
# ==============================================================================
def classify_and_export_raster(image_path, model, output_path, block_size=1024):
    """
    Phân loại toàn bộ ảnh GeoTIFF theo từng khối (block) để tránh tràn bộ nhớ RAM (OOM)
    và xuất ra file GeoTIFF mới giữ nguyên tọa độ và hệ quy chiếu (CRS).
    """
    print(f"\n[*] Bắt đầu phân loại toàn bộ ảnh: {image_path}")
    print(f"[*] File kết quả sẽ được lưu tại: {output_path}")

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
                        # Dự đoán nhãn cho các pixel hợp lệ
                        classified_block[valid_mask] = model.predict(reshaped[valid_mask])

                    # Định hình lại thành (1, h_chunk, w_chunk) và ghi vào file GeoTIFF
                    classified_block = classified_block.reshape((1, h_chunk, w_chunk))
                    dst.write(classified_block, window=window)

    print(f"[+] Hoàn tất! Ảnh phân loại đã được lưu thành công tại: {output_path}")


# ==============================================================================
# HÀM THỰC THI CHÍNH
# ==============================================================================
def main():
    print("==========================================================")
    print("    QUY TRÌNH PHÂN LOẠI RANDOM FOREST ẢNH VIỄN THÁM")
    print("==========================================================")

    # 1. Trích xuất mẫu từ ảnh và Shapefile
    X, y = extract_training_data(
        image_path=IMAGE_PATH,
        roi_path=ROI_SHAPEFILE_PATH,
        class_col=CLASS_COLUMN_NAME
    )

    # 2. Huấn luyện Random Forest và đánh giá độ chính xác
    rf_model = train_random_forest(
        X=X,
        y=y,
        n_estimators=100,
        test_size=0.2,
        random_state=42
    )

    # 3. Lưu mô hình đã huấn luyện (.pkl)
    joblib.dump(rf_model, MODEL_SAVE_PATH)
    print(f"[*] Đã lưu mô hình Random Forest tại: {MODEL_SAVE_PATH}")

    # 4. Phân loại toàn bộ ảnh và xuất file GeoTIFF
    classify_and_export_raster(
        image_path=IMAGE_PATH,
        model=rf_model,
        output_path=OUTPUT_RASTER_PATH,
        block_size=1024
    )


if __name__ == "__main__":
    main()
