# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT TÍNH TOÁN VÀ XUẤT CÁC ẢNH CHỈ SỐ PHỔ VIỄN THÁM (SPECTRAL INDICES)
LANDSAT 8 THEO ĐÚNG ĐỊNH DẠNG BÁO CÁO KHOA HỌC / LUẬN VĂN
================================================================================

Các chỉ số phổ tính toán từ 7 kênh quang học Landsat 8 (B1 - B7):
1. NDVI  (Thực vật)              : (B5 - B4) / (B5 + B4)
2. NDWI  (Độ ẩm bề mặt / Nước)    : (B3 - B5) / (B3 + B5)
3. MNDWI (Nước mặt tăng cường)   : (B3 - B6) / (B3 + B6)
4. NDBI  (Xây dựng / Đất khô)    : (B6 - B5) / (B6 + B5)
5. BSI   (Đất trống / Vết sạt)   : ((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2))

Đầu ra:
- Tấm ghép 4 chỉ số chuẩn báo cáo (NDVI, NDWI, MNDWI, NDBI) 300 DPI
- Tấm ghép 5 chỉ số đầy đủ kèm chỉ số đất trống sạt lở BSI 300 DPI
- Các tệp GeoTIFF 32-bit float (.tif) cho ArcGIS / QGIS
- Các tệp ảnh PNG đơn lẻ từng chỉ số
"""

import os
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Fix tiếng Việt terminal Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Cấu hình Matplotlib phông chữ đẹp, rõ nét
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục xuất kết quả chính
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'spectral_indices')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Danh sách các khu vực / thời kỳ cần xử lý
STUDY_AREAS = [
    {
        'id': 'muonglai_thang4_2024',
        'name': 'Xã Mường Lai (Tháng 4/2024 - Trước bão Yagi)',
        'fig_code': '3.1.8',
        'caption_4': 'Hình 3.1.8 : các chỉ số xã Mường Lai Tháng 4 năm 2024',
        'caption_5': 'Hình 3.1.8 : 5 chỉ số phổ viễn thám xã Mường Lai Tháng 4 năm 2024',
        'image_path': os.path.join(BASE_DIR, 'datath4', 'Extracth4.tif')
    },
    {
        'id': 'muonglai_thang11_2024',
        'name': 'Xã Mường Lai (Tháng 11/2024 - Sau bão Yagi)',
        'fig_code': '3.1.9',
        'caption_4': 'Hình 3.1.9 : các chỉ số xã Mường Lai Tháng 11 năm 2024',
        'caption_5': 'Hình 3.1.9 : 5 chỉ số phổ viễn thám xã Mường Lai Tháng 11 năm 2024',
        'image_path': os.path.join(BASE_DIR, 'datath11', 'Extrac11.tif')
    },
    {
        'id': 'dongphong_extracth8',
        'name': 'Khu vực Đông Phong (Ảnh Extracth8)',
        'fig_code': '3.1.10',
        'caption_4': 'Hình 3.1.10 : các chỉ số xã Đông Phong (Extracth8)',
        'caption_5': 'Hình 3.1.10 : 5 chỉ số phổ viễn thám xã Đông Phong (Extracth8)',
        'image_path': os.path.join(BASE_DIR, 'data', 'Extracth8.tif')
    }
]


def calculate_spectral_indices(src):
    """
    Tính toán 5 chỉ số phổ viễn thám chuẩn từ 7 kênh Landsat 8.
    Xử lý NoData và phép chia cho 0 an toàn.
    """
    data = src.read().astype(np.float32)
    nodata = src.nodata

    # Xác định vùng pixel hợp lệ
    if nodata is not None:
        valid = (data != nodata).all(axis=0)
    else:
        valid = (data != 0).any(axis=0)
    valid = valid & np.isfinite(data).all(axis=0)

    # 7 kênh quang học Landsat 8:
    # 0: B1 (Coastal), 1: B2 (Blue), 2: B3 (Green), 3: B4 (Red),
    # 4: B5 (NIR), 5: B6 (SWIR 1), 6: B7 (SWIR 2)
    b1, b2, b3, b4, b5, b6, b7 = [data[i] for i in range(7)]

    with np.errstate(divide='ignore', invalid='ignore'):
        # 1. NDVI: (B5 - B4) / (B5 + B4)
        denom_ndvi = b5 + b4
        ndvi = np.where(valid & (denom_ndvi != 0), (b5 - b4) / denom_ndvi, np.nan)

        # 2. NDWI (McFeeters / Gao): (B3 - B5) / (B3 + B5)
        denom_ndwi = b3 + b5
        ndwi = np.where(valid & (denom_ndwi != 0), (b3 - b5) / denom_ndwi, np.nan)

        # 3. MNDWI (Modified NDWI - Xu): (B3 - B6) / (B3 + B6)
        denom_mndwi = b3 + b6
        mndwi = np.where(valid & (denom_mndwi != 0), (b3 - b6) / denom_mndwi, np.nan)

        # 4. NDBI (Built-up): (B6 - B5) / (B6 + B5)
        denom_ndbi = b6 + b5
        ndbi = np.where(valid & (denom_ndbi != 0), (b6 - b5) / denom_ndbi, np.nan)

        # 5. BSI (Bare Soil Index): ((B6 + B4) - (B5 + B2)) / ((B6 + B4) + (B5 + B2))
        num_bsi = (b6 + b4) - (b5 + b2)
        denom_bsi = (b6 + b4) + (b5 + b2)
        bsi = np.where(valid & (denom_bsi != 0), num_bsi / denom_bsi, np.nan)

    return {
        'NDVI': ndvi,
        'NDWI': ndwi,
        'MNDWI': mndwi,
        'NDBI': ndbi,
        'BSI': bsi,
        'valid_mask': valid
    }


def export_geotiff(src, array, out_path):
    """Xuất mảng chỉ số ra định dạng GeoTIFF 32-bit Float."""
    meta = src.meta.copy()
    meta.update(
        driver='GTiff',
        dtype=rasterio.float32,
        count=1,
        nodata=-9999.0,
        compress='lzw'
    )
    arr_out = np.nan_to_num(array, nan=-9999.0).astype(np.float32)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(arr_out, 1)


def generate_composite_figure(indices_dict, area_info, mode='4_indices'):
    """
    Vẽ ảnh ghép nằm ngang chất lượng cao 300 DPI theo đúng mẫu báo cáo:
    mode='4_indices': 4 cột (NDVI, NDWI, MNDWI, NDBI) đúng như mẫu trong tài liệu
    mode='5_indices': 5 cột (NDVI, NDWI, MNDWI, NDBI, BSI) đầy đủ cho chuyên đề sạt lở
    """
    if mode == '4_indices':
        configs = [
            ('Chỉ số thực vật (NDVI)', indices_dict['NDVI'], 'RdYlGn', (0.0, 0.5)),
            ('Chỉ số nước (NDWI)', indices_dict['NDWI'], 'Blues', (-0.4, 0.1)),
            ('Chỉ số nước tăng cường (MNDWI)', indices_dict['MNDWI'], 'Blues', (-0.4, 0.3)),
            ('Chỉ số xây dựng (NDBI)', indices_dict['NDBI'], 'coolwarm', (-0.3, 0.2))
        ]
        figsize = (16, 4.6)
        caption = area_info['caption_4']
        filename = f"{area_info['id']}_4_chi_so_bao_cao.png"
    else:
        configs = [
            ('Chỉ số thực vật (NDVI)', indices_dict['NDVI'], 'RdYlGn', (0.0, 0.5)),
            ('Chỉ số nước (NDWI)', indices_dict['NDWI'], 'Blues', (-0.4, 0.1)),
            ('Chỉ số nước tăng cường (MNDWI)', indices_dict['MNDWI'], 'Blues', (-0.4, 0.3)),
            ('Chỉ số xây dựng (NDBI)', indices_dict['NDBI'], 'coolwarm', (-0.3, 0.2)),
            ('Chỉ số đất trống (BSI)', indices_dict['BSI'], 'YlOrBr', (-0.2, 0.3))
        ]
        figsize = (20, 4.6)
        caption = area_info['caption_5']
        filename = f"{area_info['id']}_5_chi_so_chuyen_sau.png"

    n_cols = len(configs)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, dpi=300, facecolor='white')

    for ax, (title, data_arr, cmap_name, default_clim) in zip(axes, configs):
        # Tính min, max thực tế (bỏ 1% percentile ngoại lai)
        val_clean = data_arr[np.isfinite(data_arr)]
        if len(val_clean) > 0:
            vmin = np.percentile(val_clean, 1)
            vmax = np.percentile(val_clean, 99)
            # Làm tròn min, max hợp lý
            vmin = np.floor(vmin * 10) / 10
            vmax = np.ceil(vmax * 10) / 10
            if vmin == vmax:
                vmin, vmax = default_clim
        else:
            vmin, vmax = default_clim

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color='white', alpha=0.0)

        im = ax.imshow(data_arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=10)
        ax.axis('off')

        # Thanh màu Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.72)
        cbar.ax.tick_params(labelsize=8)

    # Thêm chú thích tiêu đề hình ở dưới
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.05, caption, ha='center', va='center', fontsize=13, fontfamily='Times New Roman', fontweight='normal')

    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return out_path


def export_individual_index_png(array, title, cmap_name, caption, out_path):
    """Xuất ảnh PNG đơn lẻ từng chỉ số độ phân giải cao 300 DPI."""
    val_clean = array[np.isfinite(array)]
    if len(val_clean) > 0:
        vmin = np.percentile(val_clean, 1)
        vmax = np.percentile(val_clean, 99)
    else:
        vmin, vmax = -1.0, 1.0

    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300, facecolor='white')
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color='white', alpha=0.0)

    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.03, caption, ha='center', va='center', fontsize=11, fontfamily='Times New Roman')

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def process_all_areas():
    print("=" * 80)
    print("  QUY TRÌNH TÍNH TOÁN & XUẤT ẢNH CHỈ SỐ PHỔ VIỄN THÁM (LANDSAT 8)")
    print("=" * 80)

    summary_results = []

    for area in STUDY_AREAS:
        img_path = area['image_path']
        if not os.path.exists(img_path):
            print(f"[!] Bỏ qua {area['name']}: Không tìm thấy file {img_path}")
            continue

        print(f"\n[*] Đang xử lý: {area['name']}")
        print(f"    - Đường dẫn: {img_path}")

        with rasterio.open(img_path) as src:
            indices = calculate_spectral_indices(src)

            # 1. Xuất ảnh composite 4 chỉ số (Chuẩn như hình mẫu 3.1.8 / 3.1.9)
            fig4_path = generate_composite_figure(indices, area, mode='4_indices')
            print(f"    [+] Đã xuất ảnh 4 chỉ số (Chuẩn báo cáo): {fig4_path}")

            # 2. Xuất ảnh composite 5 chỉ số (Thêm BSI chuyên đề sạt lở)
            fig5_path = generate_composite_figure(indices, area, mode='5_indices')
            print(f"    [+] Đã xuất ảnh 5 chỉ số (Toàn diện):    {fig5_path}")

            # 3. Xuất GeoTIFF và PNG đơn lẻ cho từng chỉ số
            area_dir = os.path.join(OUTPUT_DIR, area['id'])
            os.makedirs(area_dir, exist_ok=True)

            index_meta = [
                ('NDVI', 'Chỉ số thực vật (NDVI)', 'RdYlGn'),
                ('NDWI', 'Chỉ số nước (NDWI)', 'Blues'),
                ('MNDWI', 'Chỉ số nước tăng cường (MNDWI)', 'Blues'),
                ('NDBI', 'Chỉ số xây dựng (NDBI)', 'coolwarm'),
                ('BSI', 'Chỉ số đất trống / vết sạt (BSI)', 'YlOrBr')
            ]

            for key, title, cmap in index_meta:
                # GeoTIFF
                tif_path = os.path.join(area_dir, f"{area['id']}_{key.lower()}.tif")
                export_geotiff(src, indices[key], tif_path)

                # PNG đơn lẻ
                png_path = os.path.join(area_dir, f"{area['id']}_{key.lower()}.png")
                export_individual_index_png(
                    indices[key], title, cmap,
                    f"{title} - {area['name']}",
                    png_path
                )

            print(f"    [+] Đã xuất 5 tệp GeoTIFF và 5 ảnh PNG đơn lẻ tại: {area_dir}")

            summary_results.append({
                'area': area,
                'fig4': fig4_path,
                'fig5': fig5_path,
                'dir': area_dir
            })

    print("\n" + "=" * 80)
    print("HOÀN TẤT XUẤT TOÀN BỘ ẢNH CHỈ SỐ PHỔ CHO TẤT CẢ KHU VỰC!")
    print(f"Thư mục lưu trữ: {OUTPUT_DIR}")
    print("=" * 80)
    return summary_results


if __name__ == '__main__':
    process_all_areas()
