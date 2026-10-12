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
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, classification_report, precision_score, recall_score, f1_score

# Tự động tạo lại file .shx nếu bị thiếu khi đọc Shapefile
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Tự động định vị thư mục gốc của dự án (d:\User\Desktop\giscode)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn thư mục ảnh viễn thám và file Shapefile trên máy
folder_path = os.path.join(BASE_DIR, 'data')
shp_path = os.path.join(BASE_DIR, 'DKT1', 'DKT.shp')
xa_tif_path = os.path.join(BASE_DIR, 'data', 'outExtractByMask1.tif')
xa_shp_path = os.path.join(BASE_DIR, 'data', 'Export_Output.shp')

# Fallback nếu chạy ở thư mục khác hoặc đường dẫn cố định
if not os.path.exists(folder_path):
    folder_path = r'D:\User\Desktop\giscode\data'
if not os.path.exists(shp_path):
    shp_path = r'D:\User\Desktop\giscode\DKT1\DKT.shp'

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

print("Đang tính toán các chỉ số chuyên sâu (NDVI, NDWI, MNDWI, NDBI) trực tiếp giống QGIS...")
np.seterr(divide='ignore', invalid='ignore')

# Tính các chỉ số dựa trên B3 (Green), B4 (Red), B5 (NIR), B6 (SWIR1)
ndvi = np.nan_to_num((b5 - b4) / (b5 + b4), nan=0.0)
ndwi = np.nan_to_num((b3 - b5) / (b3 + b5), nan=0.0)
mndwi = np.nan_to_num((b3 - b6) / (b3 + b6), nan=0.0)
ndbi = np.nan_to_num((b6 - b5) / (b6 + b5), nan=0.0)

# Gộp 11 kênh dữ liệu (7 quang học B1-B7 đã chuẩn hóa + 4 chỉ số phổ) phục vụ mô hình
stacked_image = np.stack((b1, b2, b3, b4, b5, b6, b7, ndvi, ndwi, mndwi, ndbi))

# Chuẩn bị dữ liệu hiển thị (áp dụng mặt nạ ranh giới xã để chỉ hiện đúng khuôn viên xã)
ndvi_plot = ndvi.copy()
ndwi_plot = ndwi.copy()
mndwi_plot = mndwi.copy()
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
    mndwi_plot[~mask_xa] = np.nan
    ndbi_plot[~mask_xa] = np.nan

# Hiển thị biểu đồ 4 chỉ số phổ bằng matplotlib
import matplotlib.pyplot as plt

print("Đang hiển thị biểu đồ 4 chỉ số phổ của xã...")

# Tính toán giá trị Min / Max thực tế của từng chỉ số trong ranh giới xã
def get_stats(arr):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return -0.5, 0.5
    return float(np.nanmin(valid)), float(np.nanmax(valid))

ndvi_min, ndvi_max = get_stats(ndvi_plot)
ndwi_min, ndwi_max = get_stats(ndwi_plot)
mndwi_min, mndwi_max = get_stats(mndwi_plot)
ndbi_min, ndbi_max = get_stats(ndbi_plot)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))

# 1. NDVI (Thực vật)
im1 = axes[0].imshow(ndvi_plot, cmap='RdYlGn', vmin=ndvi_min, vmax=ndvi_max)
axes[0].set_title(f"1. NDVI (Thực vật)\n[{ndvi_min:.2f} đến {ndvi_max:.2f}]", fontsize=9, fontweight='bold')
axes[0].axis('off')
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.03)

# 2. NDWI (Mặt nước tiêu chuẩn)
im2 = axes[1].imshow(ndwi_plot, cmap='Blues', vmin=ndwi_min, vmax=ndwi_max)
axes[1].set_title(f"2. NDWI (Nước TC)\n[{ndwi_min:.2f} đến {ndwi_max:.2f}]", fontsize=9, fontweight='bold')
axes[1].axis('off')
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.03)

# 3. MNDWI (Mặt nước tăng cường - Tách biệt đất xây dựng)
im3 = axes[2].imshow(mndwi_plot, cmap='Blues', vmin=mndwi_min, vmax=mndwi_max)
axes[2].set_title(f"3. MNDWI (Nước NC)\n[{mndwi_min:.2f} đến {mndwi_max:.2f}]", fontsize=9, fontweight='bold')
axes[2].axis('off')
fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.03)

# 4. NDBI (Đô thị / Đất xây dựng)
im4 = axes[3].imshow(ndbi_plot, cmap='coolwarm', vmin=ndbi_min, vmax=ndbi_max)
axes[3].set_title(f"4. NDBI (Xây dựng)\n[{ndbi_min:.2f} đến {ndbi_max:.2f}]", fontsize=9, fontweight='bold')
axes[3].axis('off')
fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.03)

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

if samples.crs is None:
    # Lấy tọa độ điểm đầu tiên để phán đoán
    geom = samples.geometry.iloc[0]
    # Lấy tọa độ trung tâm hoặc tọa độ điểm
    x0, y0 = (geom.x, geom.y) if geom.geom_type == 'Point' else (geom.centroid.x, geom.centroid.y)
    
    if -180 <= x0 <= 180 and -90 <= y0 <= 90:
        print("-> Shapefile không có CRS, nhưng tọa độ nằm trong khoảng -180 đến 180 (độ). Tự động gán WGS84 (EPSG:4326).")
        samples.set_crs(epsg=4326, inplace=True)
    else:
        print(f"-> Shapefile không có CRS và tọa độ là hệ mét. Tự động gán CRS của ảnh: {src_crs}")
        samples.set_crs(src_crs, inplace=True)

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

print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH SVM ---")
val_preds = svm_model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Độ chính xác tổng thể (Validation 15%): {val_acc * 100:.2f}%")

test_preds = svm_model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
test_kappa = cohen_kappa_score(y_test, test_preds)
print(f"Độ chính xác tổng thể (Test 15%): {test_acc * 100:.2f}%")
print(f"Hệ số Kappa (Test): {test_kappa:.4f}")

#
# 7. PHÂN TÍCH ĐỘ CHÍNH XÁC & MỐI QUAN HỆ GIỮA CHỈ SỐ NƯỚC VÀ XÂY DỰNG
#
print("\n" + "="*68)
print("PHÂN TÍCH ĐỘ CHÍNH XÁC & TƯƠNG QUAN: CHỈ SỐ NƯỚC VS CHỈ SỐ XÂY DỰNG")
print("="*68)

# Lấy các chỉ số từ tập mẫu chưa chuẩn hóa (Kênh 8: NDWI, Kênh 9: MNDWI, Kênh 10: NDBI)
raw_ndwi = X_balanced[:, 8]
raw_mndwi = X_balanced[:, 9]
raw_ndbi = X_balanced[:, 10]
labels_unique = np.unique(y_balanced)

# 1. Bảng giá trị trung bình từng chỉ số theo từng lớp phủ
print("\n[1] GIÁ TRỊ TRUNG BÌNH CỦA CÁC CHỈ SỐ THEO TỪNG LỚP ĐẤT PHỦ:")
print(f"{'Lớp (Label)':<18} | {'NDWI (Nước)':<14} | {'MNDWI (Nước TC)':<16} | {'NDBI (Xây dựng)':<16}")
print("-" * 72)
for lbl in sorted(labels_unique):
    mask_lbl = (y_balanced == lbl)
    lbl_name = f"Lớp {lbl}"
    if 'Classname' in samples.columns and 'Classvalue' in samples.columns:
        names = samples[samples['Classvalue'] == lbl]['Classname'].values
        if len(names) > 0:
            lbl_name = f"{names[0]} ({lbl})"
    print(f"{lbl_name:<18} | {raw_ndwi[mask_lbl].mean():>14.4f} | {raw_mndwi[mask_lbl].mean():>16.4f} | {raw_ndbi[mask_lbl].mean():>16.4f}")

# 2. Kiểm tra tương quan Pearson
corr_ndwi_ndbi = np.corrcoef(raw_ndwi, raw_ndbi)[0, 1]
corr_mndwi_ndbi = np.corrcoef(raw_mndwi, raw_ndbi)[0, 1]

print("\n[2] HỆ SỐ TƯƠNG QUAN TUYẾN TÍNH (PEARSON CORRELATION r):")
print(f"-> r(NDWI, NDBI)   = {corr_ndwi_ndbi:+.4f}")
print(f"-> r(MNDWI, NDBI)  = {corr_mndwi_ndbi:+.4f}")
print("\n[*] Giải thích vì sao KHÔNG tỷ lệ nghịch hoàn toàn:")
print("    - Tại sao tưởng tỷ lệ nghịch: Về mặt logic đối tượng, Nước và Đất xây dựng là 2 thái cực.")
print("    - Nhưng trong viễn thám: Phần lớn diện tích là RỪNG / THỰC VẬT. Ở vùng thực vật, kênh NIR (B5)")
print("      rất lớn nằm ở mẫu số khiến cả NDWI và NDBI đều ÂM SÂU (cùng giảm), tạo tương quan đồng biến!")
print("    - Hơn nữa, vật liệu xây dựng (bê tông, mái tôn) phản xạ mạnh ở cả Green và SWIR1 khiến NDWI")
print("      và NDBI cùng mang giá trị dương ở đô thị. Chỉ có MNDWI mới dập tắt triệt để đất xây dựng.")

# 3. Đánh giá độ chính xác độc lập của từng chỉ số khi phân tách lớp (ngưỡng chuẩn > 0)
water_label = 7
urban_label = 10
for lbl in labels_unique:
    if 'Classname' in samples.columns and 'Classvalue' in samples.columns:
        names = samples[samples['Classvalue'] == lbl]['Classname'].values
        if len(names) > 0:
            cname = str(names[0]).lower()
            if any(w in cname for w in ['thủy', 'thuy', 'nước', 'nuoc', 'water']): water_label = lbl
            if any(u in cname for u in ['dân', 'dan', 'cư', 'cu', 'xây', 'xay', 'urban', 'built']): urban_label = lbl

print("\n[3] ĐỘ CHÍNH XÁC PHÂN TÁCH ĐỘC LẬP TỪNG CHỈ SỐ (Ngưỡng phân tách chuẩn > 0):")
y_water_true = (y_balanced == water_label).astype(int)
y_ndwi_pred = (raw_ndwi > 0).astype(int)
y_mndwi_pred = (raw_mndwi > 0).astype(int)

y_urban_true = (y_balanced == urban_label).astype(int)
y_ndbi_pred = (raw_ndbi > 0).astype(int)

def print_metric(name, y_true, y_pred, target_name):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    print(f"-> {name:<6} (nhận diện {target_name:<13}): Độ chính xác={acc:.2f}% | Precision={prec:.2f}% | Recall={rec:.2f}% | F1-Score={f1:.2f}%")

print_metric("NDWI", y_water_true, y_ndwi_pred, "Mặt nước")
print_metric("MNDWI", y_water_true, y_mndwi_pred, "Mặt nước")
print_metric("NDBI", y_urban_true, y_ndbi_pred, "Đất xây dựng")

# 4. Chi tiết phân loại của mô hình SVM cho từng lớp trên tập Test
print("\n[4] BÁO CÁO ĐỘ CHÍNH XÁC CHI TIẾT TỪNG LỚP CỦA MÔ HÌNH SVM (Tập Test 15%):")
target_names = [f"Lớp {l}" for l in sorted(labels_unique)]
if 'Classname' in samples.columns and 'Classvalue' in samples.columns:
    target_names = [samples[samples['Classvalue'] == l]['Classname'].values[0] if len(samples[samples['Classvalue'] == l]['Classname'].values) > 0 else f"Lớp {l}" for l in sorted(labels_unique)]

print(classification_report(y_test, test_preds, target_names=target_names))
print("Ma trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, test_preds))