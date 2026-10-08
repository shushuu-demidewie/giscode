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

folder_path = r'D:\test\LC08_L2SP_127045_20260901_20260911_02_T1'
shp_path = r'D:\test\dong_anh_pl\dong_anh_trainning.shp'

def get_band_path(band_name):
    prefix = 'LC08_L2SP_127045_20260901_20260911_02_T1'
    return os.path.join(folder_path, f"{prefix}_{band_name}.TIF")

#
# 2. ĐỌC ẢNH (B1 ĐẾN B7) VÀ TÍNH CHỈ SỐ
#
print("Đang đọc các kênh quang học (B1 - B7)...")

# Đọc B1 và lấy thông tin không gian gốc
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

# Gộp 10 kênh dữ liệu (7 quang học B1-B7 + 3 chỉ số)
stacked_image = np.stack((b1, b2, b3, b4, b5, b6, b7, ndvi, ndwi, ndbi))

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

# Xác định cột nhãn phân loại (ưu tiên 'label', 'Classvalue', 'Id')
if 'label' in samples.columns and samples['label'].nunique() > 1:
    label_col = 'label'
elif 'Classvalue' in samples.columns and samples['Classvalue'].nunique() > 1:
    label_col = 'Classvalue'
elif 'Classname' in samples.columns and samples['Classname'].nunique() > 1:
    label_col = 'Classname'
else:
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