# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT ĐIỀU PHỐI TRUNG TÂM (MASTER RUNNER)
CHẠY TOÀN BỘ QUY TRÌNH PHÂN TÍCH GIS & HỌC MÁY LANDSAT 8
- BƯỚC 1: RIÊNG THÁNG 4/2024 (Trước bão Yagi)
- BƯỚC 2: RIÊNG THÁNG 11/2024 (Sau bão Yagi)
- BƯỚC 3: PHÂN TÍCH BIẾN ĐỘNG & KẾT HỢP CẢ 2 THÁNG
================================================================================
"""

import os
import sys
import time

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pipeline_thang_4 import run_pipeline_thang_4
from pipeline_thang_11 import run_pipeline_thang_11
from pipeline_ca_2_thang import run_pipeline_ca_2_thang


def print_banner():
    print("""
================================================================================
  HỆ THỐNG PHÂN TÍCH GIS & HỌC MÁY KHOANH VÙNG NGUY CƠ TRƯỢT LỞ ĐẤT
  KHU VỰC: XÃ MƯỜNG LAI, HUYỆN LỤC YÊN, TỈNH YÊN BÁI (ẢNH LANDSAT 8)
  THUẬT TOÁN: RANDOM FOREST (RF) & SUPPORT VECTOR MACHINE (SVM)
================================================================================
    1. Chạy quy trình RIÊNG THÁNG 4/2024 (Mùa khô - Trước bão)
    2. Chạy quy trình RIÊNG THÁNG 11/2024 (Sau bão Yagi)
    3. Chạy quy trình PHÂN TÍCH BIẾN ĐỘNG & MÔ HÌNH HÓA CẢ 2 THÁNG
    4. CHẠY TOÀN BỘ (Tháng 4 -> Tháng 11 -> Cả 2 tháng) [Mặc định]
    0. Thoát chương trình
================================================================================
    """)


def main():
    print_banner()

    # Kiểm tra tham số dòng lệnh nếu có (ví dụ: python run_all_pipelines.py 4)
    choice = "4"
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        # Nếu chạy tương tác
        try:
            user_input = input("Nhập lựa chọn của bạn (Mặc định nhấn Enter để chạy toàn bộ [4]): ").strip()
            if user_input:
                choice = user_input
        except Exception:
            choice = "4"

    t_start = time.time()

    if choice == "1":
        print("\n>>> BẮT ĐẦU CHẠY RIÊNG THÁNG 4/2024...")
        run_pipeline_thang_4()
    elif choice == "2":
        print("\n>>> BẮT ĐẦU CHẠY RIÊNG THÁNG 11/2024...")
        run_pipeline_thang_11()
    elif choice == "3":
        print("\n>>> BẮT ĐẦU CHẠY PHÂN TÍCH CẢ 2 THÁNG...")
        run_pipeline_ca_2_thang()
    elif choice == "4":
        print("\n>>> BẮT ĐẦU CHẠY TOÀN BỘ TẤT CẢ CÁC GIAI ĐOẠN...")
        print("\n--- [GIAI ĐOẠN 1/3]: RIÊNG THÁNG 4/2024 ---")
        run_pipeline_thang_4()

        print("\n--- [GIAI ĐOẠN 2/3]: RIÊNG THÁNG 11/2024 ---")
        run_pipeline_thang_11()

        print("\n--- [GIAI ĐOẠN 3/3]: PHÂN TÍCH VÀ KẾT HỢP CẢ 2 THÁNG ---")
        run_pipeline_ca_2_thang()

        total_time = time.time() - t_start
        print(f"\n[+] TỔNG THỜI GIAN THỰC HIỆN TOÀN BỘ HỆ THỐNG: {total_time:.2f} giây.")
        print("[+] Hoàn tất tất cả các bản đồ GeoTIFF, biểu đồ, bảng số liệu và ứng dụng WebGIS!")
    elif choice == "0":
        print("Đã thoát.")
        return
    else:
        print(f"Lựa chọn không hợp lệ: {choice}. Tự động chạy toàn bộ...")
        run_pipeline_thang_4()
        run_pipeline_thang_11()
        run_pipeline_ca_2_thang()


if __name__ == "__main__":
    main()
