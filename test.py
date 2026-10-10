import os
import sys

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

# Tự động tạo lại file .shx nếu bị thiếu khi đọc Shapefile
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Tự động định vị thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn thư mục ảnh viễn thám và file Shapefile trên máy
folder_path = os.path.join(BASE_DIR, 'LC08_L2SP_127045_20260901_20260911_02_T1')
shp_path = os.path.join(BASE_DIR, 'DKT', 'DKT.shp')
xa_tif_path = os.path.join(BASE_DIR, 'data1', 'outExtractByMask1.tif')
xa_shp_path = os.path.join(BASE_DIR, 'data1', 'kq_tach.shp')

def get_band_path(band_name):
    # Tìm kiếm linh hoạt file ảnh tương ứng với kênh trong thư mục
    if os.path.exists(folder_path):
        for fname in os.listdir(folder_path):
            if fname.upper().endswith(f"_{band_name.upper()}.TIF"):
                return os.path.join(folder_path, fname)
    prefix = 'LC08_L2SP_127045_20260901_20260911_02_T1'
    return os.path.join(folder_path, f"{prefix}_{band_name}.TIF")

#
# 2. ĐỌC ẢNH VÀ TÍNH CHỈ SỐ THEO PHẠM VI XÃ
#
if os.path.exists(xa_tif_path):
    print(f"-> Đang đọc ảnh viễn thám theo xã từ: {xa_tif_path}")
    with rasterio.open(xa_tif_path) as src:
        b1 = src.read(1).astype(float)
        b2 = src.read(2).astype(float)
        b3 = src.read(3).astype(float)
        b4 = src.read(4).astype(float)
        b5 = src.read(5).astype(float)
        b6 = src.read(6).astype(float)
        b7 = src.read(7).astype(float)
        src_crs = src.crs
        src_transform = src.transform
        src_shape = src.shape
else:
    print("-> Đang đọc các kênh quang học (B1 - B7) từ thư mục gốc...")
    with rasterio.open(get_band_path('SR_B1')) as src:
        b1 = src.read(1).astype(float)
        src_crs = src.crs
        src_transform = src.transform
        src_shape = src.shape

    with rasterio.open(get_band_path('SR_B2')) as src: b2 = src.read(1).astype(float)
    with rasterio.open(get_band_path('SR_B3')) as src: b3 = src.read(1).astype(float)
    with rasterio.open(get_band_path('SR_B4')) as src: b4 = src.read(1).astype(float)
    with rasterio.open(get_band_path('SR_B5')) as src: b5 = src.read(1).astype(float)
    with rasterio.open(get_band_path('SR_B6')) as src: b6 = src.read(1).astype(float)
    with rasterio.open(get_band_path('SR_B7')) as src: b7 = src.read(1).astype(float)

print("Đang tính toán các chỉ số chuyên sâu (NDVI, NDWI, NDBI)...")
np.seterr(divide='ignore', invalid='ignore')

# Tính các chỉ số dựa trên B3 (Green), B4 (Red), B5 (NIR), B6 (SWIR1)
ndvi = np.nan_to_num((b5 - b4) / (b5 + b4), nan=0.0)
ndwi = np.nan_to_num((b3 - b5) / (b3 + b5), nan=0.0)
ndbi = np.nan_to_num((b6 - b5) / (b6 + b5), nan=0.0)

# Gộp 10 kênh dữ liệu (7 quang học B1-B7 + 3 chỉ số) phục vụ mô hình
stacked_image = np.stack((b1, b2, b3, b4, b5, b6, b7, ndvi, ndwi, ndbi))

# Chuẩn bị dữ liệu hiển thị (áp dụng mặt nạ ranh giới xã để chỉ hiện đúng khuôn viên xã)
ndvi_plot = ndvi.copy()
ndwi_plot = ndwi.copy()
ndbi_plot = ndbi.copy()

if os.path.exists(xa_shp_path):
    print(f"-> Đang áp dụng mặt nạ ranh giới xã từ: {xa_shp_path}")
    xa_gdf = gpd.read_file(xa_shp_path)
    if xa_gdf.crs is None:
        xa_gdf.crs = 'EPSG:4326'
    xa_gdf = xa_gdf.to_crs(src_crs)
    mask_xa = geometry_mask(xa_gdf.geometry, transform=src_transform, invert=True, out_shape=src_shape)

    # Đặt các pixel ngoài ranh giới xã thành NaN để không hiển thị (trong suốt)
    ndvi_plot[~mask_xa] = np.nan
    ndwi_plot[~mask_xa] = np.nan
    ndbi_plot[~mask_xa] = np.nan

# Hiển thị biểu đồ 3 chỉ số phổ bằng matplotlib
import matplotlib.pyplot as plt

print("Đang hiển thị biểu đồ 3 chỉ số phổ của xã...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. NDVI (Thực vật - Thường dùng bảng màu xanh lá cây)
im1 = axes[0].imshow(ndvi_plot, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[0].set_title("Chỉ số thực vật (NDVI)")
axes[0].axis('off')
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# 2. NDWI (Mặt nước - Thường dùng bảng màu xanh dương)
im2 = axes[1].imshow(ndwi_plot, cmap='Blues', vmin=-0.5, vmax=0.5)
axes[1].set_title("Chỉ số nước (NDWI)")
axes[1].axis('off')
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# 3. NDBI (Đô thị / Đất xây dựng - Thường dùng bảng màu đỏ/cam/xám)
im3 = axes[2].imshow(ndbi_plot, cmap='coolwarm', vmin=-0.5, vmax=0.5)
axes[2].set_title("Chỉ số xây dựng (NDBI)")
axes[2].axis('off')
fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

#
# 3. TRÍCH XUẤT ĐẶC TRƯNG TỪ MẪU SHAPEFILE
#
print("\nĐang đọc file mẫu Shapefile...")
# Tự động xử lý nếu có sai khác tên file (tranning vs trainning)
if not os.path.exists(shp_path):
    alt = shp_path.replace('trainning', 'tranning') if 'trainning' in shp_path else shp_path.replace('tranning', 'trainning')
    if os.path.exists(alt):
        shp_path = alt

samples = gpd.read_file(shp_path)

if samples.crs != src_crs:
    print(f"-> Đang đồng bộ hệ tọa độ từ {samples.crs} sang {src_crs}...")
    samples = samples.to_crs(src_crs)

# Xác định cột nhãn phân loại (ưu tiên 'Nhan', 'label', 'Classvalue', 'Classname', 'Id')
priority_cols = ['label', 'Nhan', 'nhan', 'NHAN', 'Classvalue', 'Classname', 'Id']
label_col = None
for col in priority_cols:
    if col in samples.columns and samples[col].nunique() > 1:
        label_col = col
        break

if not label_col:
    # Tìm cột bất kỳ có > 1 giá trị phân loại khác biệt
    for col in samples.columns:
        if col != 'geometry' and samples[col].nunique() > 1:
            label_col = col
            break
    if not label_col:
        label_col = 'Id'

print(f"-> Sử dụng cột nhãn: '{label_col}' (gồm {samples[label_col].nunique()} lớp)")

if 'Classname' in samples.columns and label_col == 'Classvalue':
    mapping = dict(zip(samples['Classvalue'], samples['Classname']))
    print(f"-> Danh mục lớp: {mapping}")


X_raw = []
y_raw = []

print("Đang trích xuất pixel từ tập mẫu...")
for index, row in samples.iterrows():
    geom = row['geometry']
    lbl = row[label_col]

    try:
        if geom.geom_type == 'Point':
            # Với Point, chuyển đổi trực tiếp tọa độ (x, y) sang hàng và cột pixel (cực nhanh và chính xác)
            r, c = rasterio.transform.rowcol(src_transform, geom.x, geom.y)
            if 0 <= r < src_shape[0] and 0 <= c < src_shape[1]:
                pixel_values = stacked_image[:, r, c]
                X_raw.append(pixel_values)
                y_raw.append(lbl)
        else:
            # Với Polygon / MultiPolygon
            mask_2d = geometry_mask([geom], transform=src_transform, invert=True, out_shape=src_shape)
            rows, cols = np.where(mask_2d)

            for r, c in zip(rows, cols):
                pixel_values = stacked_image[:, r, c]
                X_raw.append(pixel_values)
                y_raw.append(lbl)
    except Exception as e:
        continue

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

if len(X_raw) == 0:
    raise ValueError("LỖI: Không lấy được pixel nào! Mẫu vẽ đang nằm ngoài ảnh hoặc sai tên cột.")

#
# 4. CÂN BẰNG SỐ LƯỢNG MẪU (CHỐNG OVERFITTING)
#
print("\nĐang cân bằng lại số lượng mẫu giữa các lớp...")
df = pd.DataFrame(X_raw)
df['label'] = y_raw

# Khống chế số lượng để không bị lệch lớp
min_samples = df['label'].value_counts().min()
n_samples_per_class = min(min_samples, 1500)

df_balanced = df.groupby('label').sample(n=n_samples_per_class, random_state=42).reset_index(drop=True)

X_balanced = df_balanced.drop('label', axis=1).values
y_balanced = df_balanced['label'].values

print("Cơ cấu pixel đưa vào huấn luyện sau khi cân bằng:\n", df_balanced['label'].value_counts())

#
# 5. CHIA TẬP VÀ CHUẨN HÓA DỮ LIỆU
#
X_train, X_temp, y_train, y_temp = train_test_split(X_balanced, y_balanced, test_size=0.30, random_state=42, stratify=y_balanced)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nPhân bổ: Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Tiêu chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#
# 6. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH SVM
#
print("\nĐang huấn luyện mô hình SVM...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
val_preds = svm_model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Độ chính xác (Validation 15%): {val_acc * 100:.2f}%")

test_preds = svm_model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
test_kappa = cohen_kappa_score(y_test, test_preds)
print(f"Độ chính xác (Test 15%): {test_acc * 100:.2f}%")
print(f"Hệ số Kappa (Test): {test_kappa:.4f}")