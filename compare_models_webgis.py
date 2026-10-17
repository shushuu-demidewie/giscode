# -*- coding: utf-8 -*-
"""
================================================================================
MÔ-ĐUN SO SÁNH ĐỐI ĐẦU RANDOM FOREST & SVM VÀ TỰ ĐỘNG SINH ỨNG DỤNG WEBGIS
================================================================================

Đề tài: Nghiên cứu ứng dụng học máy/học sâu và ảnh viễn thám Landsat 8 
thành lập bản đồ / WebGIS khoanh vùng nguy cơ trượt lở.

Các chức năng chính:
1. Huấn luyện đồng thời cả Random Forest (RF) và Support Vector Machine (SVM) 
   trên cùng tập mẫu viễn thám Landsat 8 (12 kênh đặc trưng).
2. Lập bảng so sánh định lượng đối đầu (Head-to-head metrics):
   Accuracy, Precision, Recall, F1-Score, Cohen's Kappa, ROC-AUC, thời gian huấn luyện.
3. Xuất đồ thị đường cong ROC kép (Dual ROC Curve) đối chiếu trực tiếp RF vs SVM.
4. Tự động sinh ứng dụng WebGIS Leaflet tương tác hoàn chỉnh (webgis_viewer.html):
   - Nền bản đồ vệ tinh ESRI World Imagery và bản đồ địa hình OpenStreetMap.
   - Bật / tắt và so sánh trực quan lớp nguy cơ sạt lở RF và SVM.
   - Thanh trượt điều chỉnh độ mờ (Opacity Slider) linh hoạt.
   - Bảng chú giải 5 cấp màu chuẩn quốc tế.
   - Bảng thống kê hiệu năng mô hình ngay trên giao diện WebGIS.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Khắc phục PROJ trên Windows
try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    if os.path.exists(proj_dir):
        os.environ['PROJ_LIB'] = proj_dir
        os.environ['PROJ_DATA'] = proj_dir
except Exception:
    pass

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Import từ 2 mô-đun RF và SVM đã hoàn thiện
from rf_classification import (
    resolve_data_paths,
    extract_training_data,
    balance_and_split_data,
    classify_and_export_raster as classify_rf,
    OUTPUT_DIR,
    OUTPUT_HAZARD_RASTER_PATH as RF_HAZARD_TIF,
    OUTPUT_WEBGIS_PNG_PATH as RF_WEBGIS_PNG,
    OUTPUT_WEBGIS_BBOX_JSON as RF_BBOX_JSON
)
from svm_classification import (
    classify_and_export_raster as classify_svm,
    OUTPUT_HAZARD_RASTER_PATH as SVM_HAZARD_TIF,
    OUTPUT_WEBGIS_PNG_PATH as SVM_WEBGIS_PNG,
    OUTPUT_WEBGIS_BBOX_JSON as SVM_BBOX_JSON
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score, roc_curve, confusion_matrix
)
import joblib

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
COMP_ROC_PNG = os.path.join(OUTPUT_DIR, 'so_sanh_roc_rf_vs_svm.png')
WEBGIS_HTML_PATH_1 = os.path.join(OUTPUT_DIR, 'webgis_viewer.html')
WEBGIS_HTML_PATH_2 = os.path.join(WORKSPACE_DIR, 'webgis_viewer.html')


def train_and_benchmark():
    print("=" * 75)
    print("  QUY TRÌNH SO SÁNH ĐỐI ĐẦU RANDOM FOREST & SUPPORT VECTOR MACHINE")
    print("  ỨNG DỤNG ẢNH VIỄN THÁM LANDSAT 8 VÀ CÔNG NGHỆ BẢN ĐỒ WEBGIS")
    print("=" * 75)

    # 1. Tự động tìm dữ liệu
    image_path, roi_path = resolve_data_paths()
    print(f"[*] Ảnh Landsat 8: {image_path}")
    print(f"[*] Shapefile mẫu: {roi_path}")

    # 2. Trích xuất 12 kênh đặc trưng
    X, y = extract_training_data(image_path=image_path, roi_path=roi_path)

    # 3. Cân bằng mẫu và chia dữ liệu (70% Train, 15% Val, 15% Test)
    data_dict = balance_and_split_data(X=X, y=y, random_state=42)

    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

    # 4. Huấn luyện Random Forest
    print("\n[*] [1/2] Đang huấn luyện Random Forest...")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=150, criterion='gini', max_features='sqrt',
        oob_score=True, n_jobs=-1, random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - t0

    # 5. Huấn luyện Support Vector Machine
    print("[*] [2/2] Đang huấn luyện Support Vector Machine (RBF Kernel)...")
    t0 = time.time()
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10.0, kernel='rbf', gamma='scale', probability=True, random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_time = time.time() - t0

    # 6. Đánh giá trên tập Test độc lập
    pred_rf = rf_model.predict(X_test)
    prob_rf = rf_model.predict_proba(X_test)[:, 1] if len(rf_model.classes_) == 2 else None

    pred_svm = svm_pipeline.predict(X_test)
    prob_svm = svm_pipeline.predict_proba(X_test)[:, 1] if len(svm_pipeline.classes_) == 2 else None

    acc_rf = accuracy_score(y_test, pred_rf)
    prec_rf = precision_score(y_test, pred_rf, average='weighted', zero_division=0)
    rec_rf = recall_score(y_test, pred_rf, average='weighted', zero_division=0)
    f1_rf = f1_score(y_test, pred_rf, average='weighted', zero_division=0)
    kap_rf = cohen_kappa_score(y_test, pred_rf)
    auc_rf = roc_auc_score(y_test, prob_rf) if prob_rf is not None else 0.0

    acc_svm = accuracy_score(y_test, pred_svm)
    prec_svm = precision_score(y_test, pred_svm, average='weighted', zero_division=0)
    rec_svm = recall_score(y_test, pred_svm, average='weighted', zero_division=0)
    f1_svm = f1_score(y_test, pred_svm, average='weighted', zero_division=0)
    kap_svm = cohen_kappa_score(y_test, pred_svm)
    auc_svm = roc_auc_score(y_test, prob_svm) if prob_svm is not None else 0.0

    # 7. In bảng so sánh đối đầu
    comp_df = pd.DataFrame({
        'Chỉ số đánh giá (Evaluation Metric)': [
            'Độ chính xác tổng thể (Overall Accuracy)',
            'Độ chính xác (Precision Weighted)',
            'Độ nhạy (Recall Weighted)',
            'Chỉ số F1-Score (F1 Weighted)',
            'Hệ số Cohen\'s Kappa',
            'Chỉ số diện tích ROC-AUC',
            'Thời gian huấn luyện (Training Time)'
        ],
        'Random Forest (RF)': [
            f"{acc_rf * 100:.2f}%",
            f"{prec_rf * 100:.2f}%",
            f"{rec_rf * 100:.2f}%",
            f"{f1_rf:.4f}",
            f"{kap_rf:.4f}",
            f"{auc_rf:.4f}",
            f"{rf_time:.3f} s"
        ],
        'Support Vector Machine (SVM)': [
            f"{acc_svm * 100:.2f}%",
            f"{prec_svm * 100:.2f}%",
            f"{rec_svm * 100:.2f}%",
            f"{f1_svm:.4f}",
            f"{kap_svm:.4f}",
            f"{auc_svm:.4f}",
            f"{svm_time:.3f} s"
        ]
    })

    print("\n" + "=" * 80)
    print("BẢNG SO SÁNH ĐỐI ĐẦU ĐỊNH LƯỢNG RANDOM FOREST VS SVM (LANDSAT 8 12 KÊNH):")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    print("=" * 80)

    # 8. Vẽ đồ thị ROC Curve kép
    plot_dual_roc_curve(y_test, prob_rf, prob_svm, auc_rf, auc_svm)

    # 9. Xuất bản đồ phân vùng và tài nguyên WebGIS cho cả RF và SVM
    print("\n[*] Đang tạo bản đồ phân vùng nguy cơ trượt lở cho cả 2 thuật toán...")
    classify_rf(image_path=image_path, model=rf_model, block_size=512)
    classify_svm(image_path=image_path, pipeline=svm_pipeline, block_size=512)

    # 10. Tự động sinh ứng dụng WebGIS Leaflet HTML
    generate_webgis_html({
        'acc_rf': acc_rf, 'f1_rf': f1_rf, 'kap_rf': kap_rf, 'auc_rf': auc_rf,
        'acc_svm': acc_svm, 'f1_svm': f1_svm, 'kap_svm': kap_svm, 'auc_svm': auc_svm
    })


def plot_dual_roc_curve(y_test, prob_rf, prob_svm, auc_rf, auc_svm):
    """Vẽ biểu đồ ROC so sánh trực tiếp RF vs SVM trên cùng một hệ trục."""
    if plt is None or prob_rf is None or prob_svm is None:
        return

    fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, prob_svm)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_rf, tpr_rf, color='#1f77b4', lw=2.5, label=f'Random Forest (AUC = {auc_rf:.4f})')
    plt.plot(fpr_svm, tpr_svm, color='#d62728', lw=2.5, linestyle='--', label=f'Support Vector Machine (AUC = {auc_svm:.4f})')
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle=':', label='Ngẫu nhiên (AUC = 0.50)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính giả (False Positive Rate)', fontsize=10, fontweight='bold')
    plt.ylabel('Tỷ lệ Dương tính thật (True Positive Rate)', fontsize=10, fontweight='bold')
    plt.title('So sánh đường cong ROC giữa Random Forest và SVM (Landsat 8)', fontsize=11, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(COMP_ROC_PNG)
    plt.close()
    print(f"[+] Đã xuất biểu đồ so sánh ROC kép tại: {COMP_ROC_PNG}")


def generate_webgis_html(metrics_dict):
    """Tự động tạo giao diện WebGIS Leaflet hiện đại, hỗ trợ so sánh bản đồ RF và SVM."""
    # Đọc thông số BBOX
    bbox_path = RF_BBOX_JSON if os.path.exists(RF_BBOX_JSON) else SVM_BBOX_JSON
    if os.path.exists(bbox_path):
        with open(bbox_path, 'r', encoding='utf-8') as f:
            bbox_info = json.load(f)
        leaflet_bounds = bbox_info['leaflet_bounds']
        center = bbox_info['center']
    else:
        leaflet_bounds = [[21.1409, 105.7705], [21.2499, 105.8867]]
        center = [21.1954, 105.8286]

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ thống WebGIS Khoanh vùng Nguy cơ Trượt lở Đất | Landsat 8 - RF vs SVM</title>
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body, html {{ height: 100%; width: 100%; font-family: 'Inter', sans-serif; overflow: hidden; }}
        #map {{ height: 100vh; width: 100vw; z-index: 1; }}

        /* Glassmorphism Control Panel */
        .glass-panel {{
            position: absolute;
            background: rgba(18, 24, 38, 0.88);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            color: #ffffff;
            z-index: 1000;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        }}

        /* Header Info Card */
        .header-card {{
            top: 20px;
            left: 20px;
            padding: 16px 20px;
            max-width: 440px;
        }}
        .header-title {{
            font-size: 15px;
            font-weight: 700;
            color: #4da6ff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .header-desc {{
            font-size: 12px;
            color: #b0c4de;
            line-height: 1.45;
        }}

        /* Layer Switcher & Opacity Card */
        .control-card {{
            top: 140px;
            left: 20px;
            padding: 16px 20px;
            width: 340px;
        }}
        .card-title {{
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #e2e8f0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 6px;
        }}
        .model-select {{
            display: flex;
            gap: 8px;
            margin-bottom: 14px;
        }}
        .model-btn {{
            flex: 1;
            padding: 8px 10px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #ffffff;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.25s ease;
            text-align: center;
        }}
        .model-btn:hover {{ background: rgba(77, 166, 255, 0.2); }}
        .model-btn.active {{
            background: #2563eb;
            border-color: #60a5fa;
            box-shadow: 0 0 12px rgba(37, 99, 235, 0.6);
        }}

        /* Slider */
        .slider-group {{
            margin-top: 10px;
        }}
        .slider-label {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #94a3b8;
            margin-bottom: 6px;
        }}
        input[type=range] {{
            width: 100%;
            height: 5px;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.2);
            outline: none;
            -webkit-appearance: none;
        }}
        input[type=range]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #60a5fa;
            cursor: pointer;
            box-shadow: 0 0 8px rgba(96, 165, 250, 0.8);
        }}

        /* Legend */
        .legend-card {{
            bottom: 25px;
            right: 25px;
            padding: 14px 18px;
            min-width: 220px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 11px;
            margin-bottom: 6px;
            color: #cbd5e1;
        }}
        .legend-color {{
            width: 18px;
            height: 12px;
            border-radius: 3px;
            margin-right: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }}

        /* Comparative Metrics Card */
        .metrics-card {{
            bottom: 25px;
            left: 20px;
            padding: 14px 18px;
            width: 340px;
        }}
        .metric-table {{
            width: 100%;
            font-size: 11px;
            border-collapse: collapse;
        }}
        .metric-table th, .metric-table td {{
            padding: 5px 6px;
            text-align: left;
        }}
        .metric-table th {{
            color: #94a3b8;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-weight: 600;
        }}
        .metric-table tr:hover {{ background: rgba(255, 255, 255, 0.05); }}
        .val-rf {{ color: #60a5fa; font-weight: 600; }}
        .val-svm {{ color: #f87171; font-weight: 600; }}
    </style>
</head>
<body>
    <div id="map"></div>

    <!-- Header Panel -->
    <div class="glass-panel header-card">
        <div class="header-title">Hệ thống Bản đồ WebGIS Khoanh vùng Nguy cơ Trượt lở</div>
        <div class="header-desc">
            Ứng dụng thuật toán Học máy <b>Random Forest (RF)</b> và <b>Support Vector Machine (SVM)</b> 
            trên ảnh viễn thám đa phổ <b>Landsat 8 OLI</b> (12 kênh đặc trưng).
        </div>
    </div>

    <!-- Control Panel -->
    <div class="glass-panel control-card">
        <div class="card-title">Lựa chọn Mô hình Học máy</div>
        <div class="model-select">
            <button class="model-btn active" id="btnRF" onclick="switchModel('rf')">Random Forest</button>
            <button class="model-btn" id="btnSVM" onclick="switchModel('svm')">SVM (RBF)</button>
        </div>

        <div class="slider-group">
            <div class="slider-label">
                <span>Độ mờ lớp phủ (Opacity)</span>
                <span id="opacityVal">85%</span>
            </div>
            <input type="range" id="opacitySlider" min="0" max="100" value="85" oninput="changeOpacity(this.value)">
        </div>
    </div>

    <!-- Metrics Comparison -->
    <div class="glass-panel metrics-card">
        <div class="card-title">So sánh Hiệu năng Mô hình (Test Set)</div>
        <table class="metric-table">
            <thead>
                <tr>
                    <th>Chỉ số</th>
                    <th>RF</th>
                    <th>SVM</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Độ chính xác (Accuracy)</td>
                    <td class="val-rf">{metrics_dict['acc_rf']*100:.1f}%</td>
                    <td class="val-svm">{metrics_dict['acc_svm']*100:.1f}%</td>
                </tr>
                <tr>
                    <td>Chỉ số F1-Score</td>
                    <td class="val-rf">{metrics_dict['f1_rf']:.4f}</td>
                    <td class="val-svm">{metrics_dict['f1_svm']:.4f}</td>
                </tr>
                <tr>
                    <td>Cohen's Kappa</td>
                    <td class="val-rf">{metrics_dict['kap_rf']:.4f}</td>
                    <td class="val-svm">{metrics_dict['kap_svm']:.4f}</td>
                </tr>
                <tr>
                    <td>ROC-AUC</td>
                    <td class="val-rf">{metrics_dict['auc_rf']:.4f}</td>
                    <td class="val-svm">{metrics_dict['auc_svm']:.4f}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- Legend Panel -->
    <div class="glass-panel legend-card">
        <div class="card-title">Phân cấp Nguy cơ Trượt lở</div>
        <div class="legend-item"><div class="legend-color" style="background:#38A800;"></div> Cấp 1: Rất thấp (0.0 - 0.2)</div>
        <div class="legend-item"><div class="legend-color" style="background:#B0E000;"></div> Cấp 2: Thấp (0.2 - 0.4)</div>
        <div class="legend-item"><div class="legend-color" style="background:#FFFF00;"></div> Cấp 3: Trung bình (0.4 - 0.6)</div>
        <div class="legend-item"><div class="legend-color" style="background:#FF8000;"></div> Cấp 4: Cao (0.6 - 0.8)</div>
        <div class="legend-item"><div class="legend-color" style="background:#FF0000;"></div> Cấp 5: Rất cao (0.8 - 1.0)</div>
    </div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Khởi tạo bản đồ
        const mapCenter = {center};
        const mapBounds = {leaflet_bounds};

        const map = L.map('map', {{
            center: mapCenter,
            zoom: 12,
            zoomControl: false
        }});
        L.control.zoom({{ position: 'topright' }}).addTo(map);

        // Các lớp bản đồ nền
        const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            maxZoom: 18
        }}).addTo(map);

        const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }});

        L.control.layers({{
            "Ảnh vệ tinh (Esri World Imagery)": esriSatellite,
            "Bản đồ OpenStreetMap": osm
        }}, null, {{ position: 'topright' }}).addTo(map);

        // Lớp ảnh phủ nguy cơ sạt lở
        const rfImageUrl = 'webgis_nguy_co_truot_lo_rf_rgba.png';
        const svmImageUrl = 'webgis_nguy_co_truot_lo_svm_rgba.png';

        let rfOverlay = L.imageOverlay(rfImageUrl, mapBounds, {{ opacity: 0.85, interactive: true }}).addTo(map);
        let svmOverlay = L.imageOverlay(svmImageUrl, mapBounds, {{ opacity: 0.85, interactive: true }});
        map.fitBounds(mapBounds);

        let currentModel = 'rf';

        function switchModel(model) {{
            currentModel = model;
            const btnRF = document.getElementById('btnRF');
            const btnSVM = document.getElementById('btnSVM');
            const curOpacity = parseFloat(document.getElementById('opacitySlider').value) / 100.0;

            if (model === 'rf') {{
                btnRF.classList.add('active');
                btnSVM.classList.remove('active');
                if (map.hasLayer(svmOverlay)) map.removeLayer(svmOverlay);
                rfOverlay.setOpacity(curOpacity);
                rfOverlay.addTo(map);
            }} else {{
                btnSVM.classList.add('active');
                btnRF.classList.remove('active');
                if (map.hasLayer(rfOverlay)) map.removeLayer(rfOverlay);
                svmOverlay.setOpacity(curOpacity);
                svmOverlay.addTo(map);
            }}
        }}

        function changeOpacity(val) {{
            const opacity = parseFloat(val) / 100.0;
            document.getElementById('opacityVal').innerText = val + '%';
            if (currentModel === 'rf') {{
                rfOverlay.setOpacity(opacity);
            }} else {{
                svmOverlay.setOpacity(opacity);
            }}
        }}
    </script>
</body>
</html>
"""

    for path in [WEBGIS_HTML_PATH_1, WEBGIS_HTML_PATH_2]:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[+] Đã tạo ứng dụng WebGIS tương tác: {path}")


if __name__ == "__main__":
    train_and_benchmark()
