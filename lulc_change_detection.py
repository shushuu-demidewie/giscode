# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT ĐÁNH GIÁ BIẾN ĐỘNG LỚP PHỦ MẶT ĐẤT (LULC CHANGE DETECTION) GIAI ĐOẠN 2020 - 2026
Thư viện sử dụng: rasterio, pandas, numpy, openpyxl
Tác vụ:
  1. Đọc và kiểm tra tính tương thích giữa 2 raster phân loại LULC (2020 và 2026).
  2. Trải phẳng mảng dữ liệu, định danh pixel_id và thực hiện INNER JOIN (pd.merge).
  3. Lọc bỏ NoData (<= 0, -9999).
  4. Lập ma trận chuyển đổi lớp phủ (Transition Matrix) theo số pixel và diện tích (ha).
  5. Tính toán bảng tổng hợp biến động ròng (Net change, Gain, Loss, Persistence).
  6. Xuất báo cáo ra file Excel 2 sheets với định dạng chuyên nghiệp.
  7. In kết quả console và tự động tạo nhận xét khoa học dựa trên số liệu.
================================================================================
"""

import os
import sys

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi font mã hóa UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Khắc phục cảnh báo xung đột PROJ database trên một số môi trường Windows cài đặt PostgreSQL/GIS
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
from rasterio.crs import CRS
from rasterio.transform import Affine

# Cấu hình hiển thị bảng pandas trên console
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.set_option('display.unicode.east_asian_width', True)


# ==============================================================================
# CẤU HÌNH THÔNG SỐ VÀ ĐƯỜNG DẪN DỮ LIỆU
# ==============================================================================
# Định vị thư mục gốc của dự án (d:\testgis)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Đường dẫn dữ liệu đầu vào và đầu ra
RASTER_2020_PATH = os.path.join(BASE_DIR, 'data', 'anh_phan_loai_2020.tif')
RASTER_2026_PATH = os.path.join(BASE_DIR, 'data', 'anh_phan_loai_2026.tif')
OUTPUT_EXCEL_PATH = os.path.join(BASE_DIR, 'data', 'bien_dong_lulc_2020_2026.xlsx')

# Bảng mã danh mục lớp phủ mặt đất (LULC)
LULC_CLASSES = {
    1: "Đất mặt nước",
    2: "Đất dân cư và giao thông",
    3: "Đất công nghiệp",
    4: "Thảm thực vật"
}

# Thông số độ phân giải không gian Landsat 8 (30m x 30m)
PIXEL_SIZE_X = 30.0  # mét
PIXEL_SIZE_Y = 30.0  # mét
PIXEL_AREA_SQM = PIXEL_SIZE_X * PIXEL_SIZE_Y     # 900 m2
PIXEL_AREA_HA = PIXEL_AREA_SQM / 10000.0         # 0.09 ha/pixel


# ==============================================================================
# HÀM HỖ TRỢ TẠO DỮ LIỆU MẪU (NẾU CHƯA CÓ FILE ĐẦU VÀO TRONG MÁY)
# ==============================================================================
def create_sample_rasters_if_missing(path_2020, path_2026):
    """
    Tự động tạo 2 file raster mẫu mô phỏng cho năm 2020 và 2026 nếu chưa tồn tại,
    giúp người dùng có thể chạy thử nghiệm và kiểm chứng toàn bộ quy trình ngay lập tức.
    """
    if os.path.exists(path_2020) and os.path.exists(path_2026):
        return

    print("=" * 80)
    print("THÔNG BÁO: Không tìm thấy file 2020 hoặc 2026 có sẵn.")
    print("-> Đang tự động khởi tạo 2 file raster GeoTIFF mô phỏng thực tế...")
    os.makedirs(os.path.dirname(path_2020), exist_ok=True)

    # Thử lấy profile từ file có sẵn trong data/anh_phan_loai_rf.tif nếu có
    ref_file = os.path.join(BASE_DIR, 'data', 'anh_phan_loai_rf.tif')
    if os.path.exists(ref_file):
        with rasterio.open(ref_file) as ref:
            height, width = ref.height, ref.width
            crs = ref.crs
            transform = ref.transform
    else:
        height, width = 200, 250
        crs = CRS.from_epsg(32648)  # WGS 84 / UTM Zone 48N
        transform = Affine(30.0, 0.0, 580000.0, 0.0, -30.0, 2320000.0)

    np.random.seed(42)

    # Khởi tạo ma trận năm 2020:
    # Tỷ lệ: Nước (1): 15%, Dân cư (2): 25%, Công nghiệp (3): 10%, Thảm thực vật (4): 50%
    probs_2020 = [0.15, 0.25, 0.10, 0.50]
    data_2020 = np.random.choice([1, 2, 3, 4], size=(height, width), p=probs_2020).astype(np.int32)

    # Thêm vùng NoData viền ngoài
    data_2020[:10, :] = -9999
    data_2020[-10:, :] = -9999
    data_2020[:, :10] = -9999
    data_2020[:, -10:] = -9999

    # Khởi tạo ma trận năm 2026 mô phỏng quá trình đô thị hóa & công nghiệp hóa:
    # Phần lớn giữ nguyên, một phần thảm thực vật chuyển thành dân cư và công nghiệp
    data_2026 = data_2020.copy()
    valid_mask = data_2020 > 0

    # Chuyển đổi ngẫu nhiên có quy luật đô thị hóa
    change_rand = np.random.rand(height, width)
    # 25% thảm thực vật chuyển sang đất dân cư
    veg_to_res = (data_2020 == 4) & (change_rand < 0.25) & valid_mask
    data_2026[veg_to_res] = 2
    # 10% thảm thực vật chuyển sang đất công nghiệp
    veg_to_ind = (data_2020 == 4) & (change_rand >= 0.25) & (change_rand < 0.35) & valid_mask
    data_2026[veg_to_ind] = 3
    # 5% đất dân cư mở rộng thành đất công nghiệp
    res_to_ind = (data_2020 == 2) & (change_rand < 0.05) & valid_mask
    data_2026[res_to_ind] = 3
    # 5% đất mặt nước bị san lấp làm dân cư
    wat_to_res = (data_2020 == 1) & (change_rand < 0.05) & valid_mask
    data_2026[wat_to_res] = 2

    # Lưu 2 file GeoTIFF
    meta = {
        'driver': 'GTiff',
        'dtype': 'int32',
        'nodata': -9999,
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }

    if not os.path.exists(path_2020):
        with rasterio.open(path_2020, 'w', **meta) as dst:
            dst.write(data_2020, 1)
        print(f" -> Đã khởi tạo file 2020 mẫu: {path_2020}")

    if not os.path.exists(path_2026):
        with rasterio.open(path_2026, 'w', **meta) as dst:
            dst.write(data_2026, 1)
        print(f" -> Đã khởi tạo file 2026 mẫu: {path_2026}")
    print("=" * 80 + "\n")


# ==============================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU RASTER VÀ KIỂM TRA TƯƠNG THÍCH KHÔNG GIAN
# ==============================================================================
def read_and_validate_rasters(path_2020, path_2026):
    """
    Đọc 2 file raster, kiểm tra tính đồng nhất về kích thước (shape) và hệ tọa độ (CRS).
    """
    print("=" * 80)
    print("BƯỚC 1: ĐỌC VÀ KIỂM TRA TÍNH ĐỒNG BỘ CỦA DỮ LIỆU RASTER")
    print("=" * 80)

    if not os.path.exists(path_2020):
        raise FileNotFoundError(f"LỖI: Không tìm thấy file 2020 tại: {path_2020}")
    if not os.path.exists(path_2026):
        raise FileNotFoundError(f"LỖI: Không tìm thấy file 2026 tại: {path_2026}")

    with rasterio.open(path_2020) as src_2020, rasterio.open(path_2026) as src_2026:
        print(f"[*] File 2020: {path_2020}")
        print(f"    - Kích thước (Cao x Rộng): {src_2020.height} x {src_2020.width} pixels")
        print(f"    - Hệ tọa độ (CRS): {src_2020.crs}")
        print(f"    - Độ phân giải pixel: {src_2020.res[0]:.2f}m x {abs(src_2020.res[1]):.2f}m")
        print(f"    - Giá trị NoData: {src_2020.nodata}")

        print(f"[*] File 2026: {path_2026}")
        print(f"    - Kích thước (Cao x Rộng): {src_2026.height} x {src_2026.width} pixels")
        print(f"    - Hệ tọa độ (CRS): {src_2026.crs}")
        print(f"    - Độ phân giải pixel: {src_2026.res[0]:.2f}m x {abs(src_2026.res[1]):.2f}m")
        print(f"    - Giá trị NoData: {src_2026.nodata}")

        # Kiểm tra kích thước không gian
        if (src_2020.height != src_2026.height) or (src_2020.width != src_2026.width):
            raise ValueError(
                f"LỖI: Kích thước 2 raster không khớp! "
                f"2020 ({src_2020.height}x{src_2020.width}) vs 2026 ({src_2026.height}x{src_2026.width}). "
                f"Vui lòng resampling hoặc crop đồng bộ trước khi so sánh."
            )

        # Kiểm tra hệ quy chiếu (CRS)
        if src_2020.crs != src_2026.crs:
            print("(!) CẢNH BÁO: Hệ quy chiếu (CRS) giữa 2 file không hoàn toàn trùng khớp!")
        else:
            print("[+] Kiểm tra tương thích: Kích thước và Hệ tọa độ 2 ảnh KHỚP HOÀN TOÀN.")

        # Đọc dữ liệu mảng band 1
        arr_2020 = src_2020.read(1)
        arr_2026 = src_2026.read(1)
        nodata_2020 = src_2020.nodata
        nodata_2026 = src_2026.nodata

    return arr_2020, arr_2026, nodata_2020, nodata_2026


# ==============================================================================
# BƯỚC 2: TRẢI PHẲNG MẢNG, ĐỊNH DANH PIXEL_ID VÀ THỰC HIỆN INNER JOIN
# ==============================================================================
def process_pixel_matching(arr_2020, arr_2026, nodata_2020, nodata_2026):
    """
    Trải phẳng 2 mảng raster, gán mã pixel_id, dùng pd.merge (INNER JOIN)
    và loại bỏ toàn bộ pixel NoData.
    """
    print("\n" + "=" * 80)
    print("BƯỚC 2: TRẢI PHẲNG DỮ LIỆU & INNER JOIN THEO PIXEL_ID")
    print("=" * 80)

    total_pixels = arr_2020.size
    print(f"[*] Tổng số lượng pixel raster ban đầu: {total_pixels:,} pixels")

    # Tạo chỉ mục không gian duy nhất cho từng pixel
    pixel_ids = np.arange(total_pixels, dtype=np.int64)

    # Trải phẳng mảng 2 chiều thành 1 chiều
    flat_2020 = arr_2020.ravel()
    flat_2026 = arr_2026.ravel()

    # Xây dựng 2 DataFrame tương ứng 2 mốc thời gian
    df_2020 = pd.DataFrame({
        'pixel_id': pixel_ids,
        'code_2020': flat_2020
    })

    df_2026 = pd.DataFrame({
        'pixel_id': pixel_ids,
        'code_2026': flat_2026
    })

    # Thực hiện thao tác INNER JOIN theo pixel_id giữa 2 mốc thời gian
    print("[*] Đang thực hiện INNER JOIN (pd.merge with how='inner')...")
    df_merged = pd.merge(df_2020, df_2026, on='pixel_id', how='inner')

    # Định nghĩa điều kiện lọc bỏ NoData:
    # 1. Giá trị <= 0
    # 2. Giá trị bằng -9999
    # 3. Giá trị bằng nodata được ghi trong metadata raster
    valid_condition = (
        (df_merged['code_2020'] > 0) &
        (df_merged['code_2020'] != -9999) &
        (df_merged['code_2026'] > 0) &
        (df_merged['code_2026'] != -9999)
    )

    if nodata_2020 is not None:
        valid_condition &= (df_merged['code_2020'] != nodata_2020)
    if nodata_2026 is not None:
        valid_condition &= (df_merged['code_2026'] != nodata_2026)

    # Chỉ giữ các pixel có mã nằm trong danh mục bảng mã lớp phủ quy định (1, 2, 3, 4)
    valid_classes = set(LULC_CLASSES.keys())
    valid_condition &= df_merged['code_2020'].isin(valid_classes)
    valid_condition &= df_merged['code_2026'].isin(valid_classes)

    df_clean = df_merged[valid_condition].copy()

    nodata_count = total_pixels - len(df_clean)
    print(f"[+] Số pixel hợp lệ tham gia đánh giá: {len(df_clean):,} pixels "
          f"({(len(df_clean)/total_pixels)*100:.2f}%)")
    print(f"[-] Số pixel NoData đã loại bỏ: {nodata_count:,} pixels "
          f"({(nodata_count/total_pixels)*100:.2f}%)")

    # Ánh xạ mã số sang tên nhãn tiếng Việt tương ứng
    df_clean['Ten_2020'] = df_clean['code_2020'].map(LULC_CLASSES)
    df_clean['Ten_2026'] = df_clean['code_2026'].map(LULC_CLASSES)

    return df_clean


# ==============================================================================
# BƯỚC 3: THÀNH LẬP MA TRẬN CHUYỂN ĐỔI (TRANSITION MATRIX)
# ==============================================================================
def build_transition_matrix(df_clean, pixel_area_ha):
    """
    Tạo ma trận chuyển đổi lớp phủ (Cross-tabulation) dạng pixel và hecta (ha).
    Hàng: Năm 2020 (Nguồn chuyển dịch)
    Cột: Năm 2026 (Đích chuyển dịch)
    """
    print("\n" + "=" * 80)
    print("BƯỚC 3: LẬP MA TRẬN CHUYỂN ĐỔI LỚP PHỦ (TRANSITION MATRIX)")
    print("=" * 80)

    class_names = [LULC_CLASSES[k] for k in sorted(LULC_CLASSES.keys())]

    # Ma trận chuyển đổi theo số lượng pixel
    matrix_pixel = pd.crosstab(
        index=df_clean['Ten_2020'],
        columns=df_clean['Ten_2026'],
        rownames=['Lớp phủ năm 2020'],
        colnames=['Lớp phủ năm 2026'],
        margins=True,
        margins_name='Tổng 2020'
    )

    # Đảm bảo đầy đủ các lớp theo thứ tự mong muốn kể cả khi có lớp nhận giá trị 0
    matrix_pixel = matrix_pixel.reindex(
        index=class_names + ['Tổng 2020'],
        columns=class_names + ['Tổng 2020'],
        fill_value=0
    )

    # Đổi tên nhãn cột tổng thành 'Tổng 2020' và dòng tổng thành 'Tổng 2026' để người đọc dễ phân biệt
    matrix_pixel.rename(index={'Tổng 2020': 'Tổng 2026'}, inplace=True)

    # Ma trận chuyển đổi theo diện tích (Hecta - ha)
    matrix_ha = (matrix_pixel * pixel_area_ha).round(2)

    return matrix_pixel, matrix_ha, class_names


# ==============================================================================
# BƯỚC 4: TÍNH TOÁN BẢNG TỔNG HỢP BIẾN ĐỘNG RÒNG (NET CHANGE SUMMARY)
# ==============================================================================
def calculate_net_change(matrix_ha, class_names):
    """
    Tính bảng tổng hợp biến động ròng:
      - Diện tích 2020, Diện tích 2026
      - Ổn định (Persistence), Mất đi (Loss), Tăng thêm (Gain)
      - Chênh lệch biến động ròng (+/- ha)
      - Tỷ lệ phần trăm biến động (%)
    """
    print("\n" + "=" * 80)
    print("BƯỚC 4: TÍNH TOÁN BIẾN ĐỘNG RÒNG (NET CHANGE) & CÁC CHỈ SỐ LULC")
    print("=" * 80)

    summary_rows = []

    for c in class_names:
        area_2020 = matrix_ha.loc[c, 'Tổng 2020']
        area_2026 = matrix_ha.loc['Tổng 2026', c]
        persistence = matrix_ha.loc[c, c]  # Đường chéo chính: giữ nguyên lớp phủ

        loss = area_2020 - persistence     # Diện tích năm 2020 chuyển sang lớp khác
        gain = area_2026 - persistence     # Diện tích năm 2026 nhận thêm từ lớp khác
        net_change = area_2026 - area_2020 # Biến động ròng = Gain - Loss

        pct_change = (net_change / area_2020 * 100.0) if area_2020 > 0 else 0.0

        summary_rows.append({
            'Lớp phủ mặt đất': c,
            'Diện tích 2020 (ha)': round(area_2020, 2),
            'Diện tích 2026 (ha)': round(area_2026, 2),
            'Giữ nguyên / Ổn định (ha)': round(persistence, 2),
            'Diện tích mất đi (ha)': round(loss, 2),
            'Diện tích tăng thêm (ha)': round(gain, 2),
            'Chênh lệch (+/- ha)': round(net_change, 2),
            'Tỷ lệ biến động (%)': round(pct_change, 2)
        })

    df_summary = pd.DataFrame(summary_rows)

    # Thêm dòng Tổng cộng
    total_2020 = df_summary['Diện tích 2020 (ha)'].sum()
    total_2026 = df_summary['Diện tích 2026 (ha)'].sum()
    total_persistence = df_summary['Giữ nguyên / Ổn định (ha)'].sum()
    total_loss = df_summary['Diện tích mất đi (ha)'].sum()
    total_gain = df_summary['Diện tích tăng thêm (ha)'].sum()
    total_net = df_summary['Chênh lệch (+/- ha)'].sum()

    total_row = pd.DataFrame([{
        'Lớp phủ mặt đất': 'TỔNG CỘNG',
        'Diện tích 2020 (ha)': round(total_2020, 2),
        'Diện tích 2026 (ha)': round(total_2026, 2),
        'Giữ nguyên / Ổn định (ha)': round(total_persistence, 2),
        'Diện tích mất đi (ha)': round(total_loss, 2),
        'Diện tích tăng thêm (ha)': round(total_gain, 2),
        'Chênh lệch (+/- ha)': round(total_net, 2),
        'Tỷ lệ biến động (%)': 0.00
    }])

    df_summary_with_total = pd.concat([df_summary, total_row], ignore_index=True)

    return df_summary, df_summary_with_total


# ==============================================================================
# BƯỚC 5: XUẤT BÁO CÁO RA FILE EXCEL (2 SHEETS) CÓ ĐỊNH DẠNG ĐẸP
# ==============================================================================
def export_to_excel(matrix_pixel, matrix_ha, df_summary_with_total, output_path):
    """
    Xuất ma trận chuyển đổi và bảng biến động ròng ra file Excel gồm 2 sheet rõ ràng.
    Áp dụng màu sắc, viền, font chữ và căn chỉnh chuyên nghiệp bằng openpyxl.
    """
    print("\n" + "=" * 80)
    print("BƯỚC 5: XUẤT KẾT QUẢ RA FILE EXCEL CHUYÊN NGHIỆP")
    print("=" * 80)

    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Ma trận chuyển đổi
        # Xuất Ma trận Diện tích (ha) trước, tiếp theo là Ma trận Pixel
        matrix_ha.to_excel(writer, sheet_name='Ma_tran_chuyen_doi', startrow=3, startcol=1)
        start_row_pixel = 3 + len(matrix_ha) + 5
        matrix_pixel.to_excel(writer, sheet_name='Ma_tran_chuyen_doi', startrow=start_row_pixel, startcol=1)

        # Sheet 2: Bảng tổng hợp biến động
        df_summary_with_total.to_excel(writer, sheet_name='Tong_hop_bien_dong', startrow=3, startcol=1, index=False)

        # Định dạng trang trí file Excel bằng openpyxl
        wb = writer.book

        # Phong cách định dạng chung
        font_title = Font(name='Segoe UI', size=13, bold=True, color='1B365D')
        font_section = Font(name='Segoe UI', size=11, bold=True, color='2C3E50')
        font_header = Font(name='Segoe UI', size=10, bold=True, color='FFFFFF')
        font_bold = Font(name='Segoe UI', size=10, bold=True)
        font_regular = Font(name='Segoe UI', size=10)

        fill_header_navy = PatternFill(start_color='1B365D', end_color='1B365D', fill_type='solid')
        fill_header_teal = PatternFill(start_color='16A085', end_color='16A085', fill_type='solid')
        fill_diagonal = PatternFill(start_color='E8F8F5', end_color='E8F8F5', fill_type='solid')  # Ô ổn định
        fill_total = PatternFill(start_color='EAEDED', end_color='EAEDED', fill_type='solid')

        thin_side = Side(border_style='thin', color='BDC3C7')
        double_side = Side(border_style='double', color='2C3E50')
        border_cell = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
        border_total = Border(top=thin_side, bottom=double_side, left=thin_side, right=thin_side)

        align_center = Alignment(horizontal='center', vertical='center')
        align_right = Alignment(horizontal='right', vertical='center')
        align_left = Alignment(horizontal='left', vertical='center')

        # --- ĐỊNH DẠNG SHEET 1: MA TRẬN CHUYỂN ĐỔI ---
        ws1 = wb['Ma_tran_chuyen_doi']
        ws1.views.sheetView[0].showGridLines = True

        # Tiêu đề bảng 1
        ws1['B2'] = "MA TRẬN CHUYỂN ĐỔI LỚP PHỦ MẶT ĐẤT GIAI ĐOẠN 2020 - 2026 (ĐƠN VỊ: HECTA - HA)"
        ws1['B2'].font = font_title

        ws1[f'B{start_row_pixel - 1}'] = "MA TRẬN CHUYỂN ĐỔI LỚP PHỦ MẶT ĐẤT GIAI ĐOẠN 2020 - 2026 (ĐƠN VỊ: PIXEL)"
        ws1[f'B{start_row_pixel - 1}'].font = font_section

        # Tô màu và viền cho Ma trận Hecta
        ha_start_r = 4
        ha_end_r = ha_start_r + len(matrix_ha)
        ha_end_c = 1 + len(matrix_ha.columns)

        for r in range(ha_start_r, ha_end_r + 1):
            for c in range(2, ha_end_c + 1):
                cell = ws1.cell(row=r, column=c)
                cell.border = border_cell
                cell.font = font_regular
                if r == ha_start_r or c == 2:
                    cell.font = font_header if r == ha_start_r else font_bold
                    if r == ha_start_r:
                        cell.fill = fill_header_navy
                        cell.alignment = align_center
                    else:
                        cell.alignment = align_left
                elif r == ha_end_r or c == ha_end_c:
                    cell.font = font_bold
                    cell.fill = fill_total
                    cell.alignment = align_right
                    cell.number_format = '#,##0.00'
                else:
                    # Ô đường chéo chính (Ổn định)
                    if (r - ha_start_r) == (c - 2):
                        cell.fill = fill_diagonal
                        cell.font = font_bold
                    cell.alignment = align_right
                    cell.number_format = '#,##0.00'

        # Tô màu và viền cho Ma trận Pixel
        pix_start_r = start_row_pixel + 1
        pix_end_r = pix_start_r + len(matrix_pixel)
        pix_end_c = 1 + len(matrix_pixel.columns)

        for r in range(pix_start_r, pix_end_r + 1):
            for c in range(2, pix_end_c + 1):
                cell = ws1.cell(row=r, column=c)
                cell.border = border_cell
                cell.font = font_regular
                if r == pix_start_r or c == 2:
                    cell.font = font_header if r == pix_start_r else font_bold
                    if r == pix_start_r:
                        cell.fill = fill_header_teal
                        cell.alignment = align_center
                    else:
                        cell.alignment = align_left
                elif r == pix_end_r or c == pix_end_c:
                    cell.font = font_bold
                    cell.fill = fill_total
                    cell.alignment = align_right
                    cell.number_format = '#,##0'
                else:
                    if (r - pix_start_r) == (c - 2):
                        cell.fill = fill_diagonal
                        cell.font = font_bold
                    cell.alignment = align_right
                    cell.number_format = '#,##0'

        # Căn chỉnh độ rộng cột Sheet 1
        for col in ws1.columns:
            col_letter = get_column_letter(col[0].column)
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws1.column_dimensions[col_letter].width = max(max_len + 4, 14)

        # --- ĐỊNH DẠNG SHEET 2: BẢNG TỔNG HỢP BIẾN ĐỘNG ---
        ws2 = wb['Tong_hop_bien_dong']
        ws2.views.sheetView[0].showGridLines = True

        ws2['B2'] = "BẢNG TỔNG HỢP BIẾN ĐỘNG RÒNG LỚP PHỦ MẶT ĐẤT (NET CHANGE) GIAI ĐOẠN 2020 - 2026"
        ws2['B2'].font = font_title

        sum_start_r = 4
        sum_end_r = sum_start_r + len(df_summary_with_total)
        sum_end_c = 1 + len(df_summary_with_total.columns)

        # Định dạng Header
        for c in range(2, sum_end_c + 1):
            cell = ws2.cell(row=sum_start_r, column=c)
            cell.font = font_header
            cell.fill = fill_header_navy
            cell.alignment = align_center
            cell.border = border_cell

        # Định dạng Dữ liệu
        for r in range(sum_start_r + 1, sum_end_r + 1):
            is_total_row = (r == sum_end_r)
            for c in range(2, sum_end_c + 1):
                cell = ws2.cell(row=r, column=c)
                cell.border = border_total if is_total_row else border_cell

                if is_total_row:
                    cell.font = font_bold
                    cell.fill = fill_total
                else:
                    cell.font = font_regular

                if c == 2:  # Tên lớp phủ
                    cell.alignment = align_left
                elif c == sum_end_c:  # Tỷ lệ %
                    cell.alignment = align_right
                    cell.number_format = '+0.00%;-0.00%;0.00%' if not is_total_row else '0.00%'
                    if not is_total_row and cell.value is not None:
                        cell.value = cell.value / 100.0  # Đổi sang tỷ lệ thực tế cho Excel % format
                else:  # Các cột diện tích ha
                    cell.alignment = align_right
                    cell.number_format = '#,##0.00'

        # Căn chỉnh độ rộng cột Sheet 2
        for col in ws2.columns:
            col_letter = get_column_letter(col[0].column)
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws2.column_dimensions[col_letter].width = max(max_len + 4, 18)

    print(f"[+] ĐÃ XUẤT BÁO CÁO THÀNH CÔNG: {output_path}")


# ==============================================================================
# BƯỚC 6: IN KẾT QUẢ VÀ TỰ ĐỘNG TẠO NHẬN XÉT KHOA HỌC
# ==============================================================================
def display_console_and_generate_insights(matrix_ha, df_summary, class_names):
    """
    In kết quả ma trận và bảng tổng hợp ra console, đồng thời tự động phân tích
    đường chéo chính và các luồng chuyển dịch lớn nhất để sinh ra đoạn nhận xét mẫu.
    """
    print("\n" + "=" * 80)
    print("BƯỚC 6: KẾT QUẢ THỐNG KÊ & BÁO CÁO NHẬN XÉT BIẾN ĐỘNG")
    print("=" * 80)

    print("\n--- 1. MA TRẬN CHUYỂN ĐỔI LỚP PHỦ 2020 - 2026 (ĐƠN VỊ: HECTA) ---")
    print("(Hàng: Năm 2020 -> Cột: Năm 2026)")
    print(matrix_ha.to_string())

    print("\n--- 2. BẢNG TỔNG HỢP BIẾN ĐỘNG RÒNG (NET CHANGE SUMMARY) ---")
    print(df_summary.to_string(index=False))

    # --- TỰ ĐỘNG PHÂN TÍCH SỐ LIỆU ĐỂ SINH NHẬN XÉT ---
    total_area_2020 = df_summary['Diện tích 2020 (ha)'].sum()
    total_area_2026 = df_summary['Diện tích 2026 (ha)'].sum()
    total_persistence = df_summary['Giữ nguyên / Ổn định (ha)'].sum()
    persistence_rate = (total_persistence / total_area_2020 * 100.0) if total_area_2020 > 0 else 0.0

    # Lớp phủ có độ ổn định nội tại cao nhất (% giữ nguyên trên tổng diện tích 2020 của chính nó)
    stable_class = max(class_names, key=lambda c: matrix_ha.loc[c, c] / matrix_ha.loc[c, 'Tổng 2020'] if matrix_ha.loc[c, 'Tổng 2020'] > 0 else 0)
    stable_rate = (matrix_ha.loc[stable_class, stable_class] / matrix_ha.loc[stable_class, 'Tổng 2020']) * 100.0

    # Lớp phủ tăng trưởng ròng mạnh nhất
    top_gain_row = df_summary.loc[df_summary['Chênh lệch (+/- ha)'].idxmax()]
    # Lớp phủ suy giảm ròng mạnh nhất
    top_loss_row = df_summary.loc[df_summary['Chênh lệch (+/- ha)'].idxmin()]

    # Tìm các ô chuyển đổi ngoài đường chéo chính lớn nhất (From -> To)
    transitions = []
    for r in class_names:
        for c in class_names:
            if r != c:
                area = matrix_ha.loc[r, c]
                if area > 0:
                    transitions.append((r, c, area))
    transitions.sort(key=lambda x: x[2], reverse=True)

    top_1_trans = transitions[0] if len(transitions) > 0 else ("Không có", "Không có", 0.0)
    top_2_trans = transitions[1] if len(transitions) > 1 else ("Không có", "Không có", 0.0)

    print("\n" + "=" * 80)
    print("NHẬN XÉT MẪU VỀ BIẾN ĐỘNG LỚP PHỦ MẶT ĐẤT (LULC INSIGHTS)")
    print("=" * 80)

    insights = f"""
1. TỔNG QUAN KHÔNG GIAN VÀ TÍNH ỔN ĐỊNH:
   - Tổng diện tích khu vực nghiên cứu được theo dõi là {total_area_2020:,.2f} ha.
   - Tính ổn định của lớp phủ mặt đất (Persistence - đường chéo chính) đạt {total_persistence:,.2f} ha, 
     tương đương {persistence_rate:.2f}% tổng diện tích tự nhiên giữ nguyên hiện trạng lớp phủ trong giai đoạn 2020 - 2026.
   - Lớp phủ có tính bền vững, ít bị chuyển đổi nhất là '{stable_class}' với tỷ lệ giữ nguyên đạt {stable_rate:.2f}%.

2. XU HƯỚNG TĂNG TRƯỞNG & SUY GIẢM RÒNG (NET CHANGE):
   - Lớp phủ TĂNG MẠNH NHẤT: '{top_gain_row['Lớp phủ mặt đất']}' tăng thêm {top_gain_row['Chênh lệch (+/- ha)']:+,.2f} ha 
     (tăng trưởng {top_gain_row['Tỷ lệ biến động (%)']:+.2f}% so với năm 2020), từ {top_gain_row['Diện tích 2020 (ha)']:,.2f} ha 
     lên {top_gain_row['Diện tích 2026 (ha)']:,.2f} ha.
   - Lớp phủ SUY GIẢM MẠNH NHẤT: '{top_loss_row['Lớp phủ mặt đất']}' giảm {abs(top_loss_row['Chênh lệch (+/- ha)']):,.2f} ha 
     (suy giảm {top_loss_row['Tỷ lệ biến động (%)']:+.2f}% so với năm 2020), từ {top_loss_row['Diện tích 2020 (ha)']:,.2f} ha 
     xuống còn {top_loss_row['Diện tích 2026 (ha)']:,.2f} ha.

3. CÁC LUỒNG CHUYỂN DỊCH KHÔNG GIAN NỔI BẬT NHẤT:
   - Luồng biến động thứ nhất: Có {top_1_trans[2]:,.2f} ha từ '{top_1_trans[0]}' đã chuyển đổi thành '{top_1_trans[1]}'.
   - Luồng biến động thứ hai: Có {top_2_trans[2]:,.2f} ha từ '{top_2_trans[0]}' đã chuyển đổi thành '{top_2_trans[1]}'.
   - Nhận định: Xu thế diễn thế phản ánh rõ nét quá trình phát triển không gian kinh tế - xã hội, đô thị hóa và mở rộng hạ tầng,
     gây áp lực thu hẹp diện tích đất nông nghiệp, thảm thực vật tự nhiên và mặt nước.
"""
    print(insights.strip())
    print("=" * 80)


# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH (MAIN WORKFLOW)
# ==============================================================================
def main():
    print("*" * 80)
    print("CHƯƠNG TRÌNH PHÂN TÍCH BIẾN ĐỘNG LỚP PHỦ MẶT ĐẤT (LULC) 2020 - 2026")
    print("Thư viện: rasterio | pandas | numpy | openpyxl")
    print("*" * 80)

    # 0. Tự động khởi tạo dữ liệu mô phỏng nếu người dùng chưa có sẵn 2 file tif
    create_sample_rasters_if_missing(RASTER_2020_PATH, RASTER_2026_PATH)

    # 1. Đọc và kiểm tra kích thước, hệ quy chiếu của 2 file raster
    arr_2020, arr_2026, nodata_2020, nodata_2026 = read_and_validate_rasters(
        RASTER_2020_PATH, RASTER_2026_PATH
    )

    # 2. Trải phẳng mảng, gán pixel_id, INNER JOIN và lọc NoData
    df_clean = process_pixel_matching(arr_2020, arr_2026, nodata_2020, nodata_2026)

    # 3. Lập ma trận chuyển đổi lớp phủ (pixel & ha)
    matrix_pixel, matrix_ha, class_names = build_transition_matrix(df_clean, PIXEL_AREA_HA)

    # 4. Tính toán bảng tổng hợp biến động ròng
    df_summary, df_summary_with_total = calculate_net_change(matrix_ha, class_names)

    # 5. Xuất báo cáo ra file Excel 2 sheets
    export_to_excel(matrix_pixel, matrix_ha, df_summary_with_total, OUTPUT_EXCEL_PATH)

    # 6. In kết quả và đưa ra đoạn nhận xét mẫu
    display_console_and_generate_insights(matrix_ha, df_summary, class_names)

    print("\n[HOÀN TẤT] Quy trình đánh giá biến động lớp phủ mặt đất đã hoàn thành thành công!")


if __name__ == '__main__':
    main()
