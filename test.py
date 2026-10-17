# -*- coding: utf-8 -*-
"""
================================================================================
QUY TRÌNH PHÂN LOẠI VÀ KHOANH VÙNG NGUY CƠ TRƯỢT LỞ BẰNG SUPPORT VECTOR MACHINE (SVM)
================================================================================
File này đóng vai trò tương thích ngược (Backward Compatibility) và liên kết trực tiếp
với mô-đun chuẩn svm_classification.py.

Đề tài: Nghiên cứu ứng dụng học máy/học sâu và ảnh viễn thám Landsat 8 
thành lập bản đồ / WebGIS khoanh vùng nguy cơ trượt lở.
"""

import sys
import os

# Thêm thư mục hiện tại vào PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from svm_classification import (
    main,
    resolve_data_paths,
    extract_training_data,
    balance_and_split_data,
    train_and_evaluate_svm,
    classify_and_export_raster,
    compute_spectral_indices,
    add_spectral_indices_to_features
)

if __name__ == "__main__":
    main()