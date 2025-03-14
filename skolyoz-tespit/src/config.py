# src/config.py

import os
from pathlib import Path

# Proje kök dizinini belirle
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Veri dizinleri
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PRETRAINED_DIR = os.path.join(DATA_DIR, "pretrained")

# Çıktı dizinleri
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualization")

# Model parametreleri
MODEL_PARAMS = {
    "input_size": (512, 512),    # Giriş görüntü boyutu
    "batch_size": 8,             # Batch boyutu
    "learning_rate": 0.001,      # Öğrenme hızı
    "epochs": 50,                # Eğitim devir sayısı
}

# Segmentasyon parametreleri
SEGMENTATION_PARAMS = {
    "threshold": 0.5,            # Segmentasyon eşik değeri
    "min_contour_area": 1000,    # Minimum kontur alanı
    "spine_width_range": (20, 100),  # Omurga genişlik aralığı (piksel)
}

# Cobb açısı hesaplama parametreleri
COBB_ANGLE_PARAMS = {
    "num_points": 50,            # Omurga eğrisindeki nokta sayısı
    "smoothing_factor": 0.85,    # Eğri düzleştirme faktörü
    "curve_detection_method": "polynomial",  # Eğri algılama yöntemi
}

# Görselleştirme parametreleri
VISUALIZATION_PARAMS = {
    "line_thickness": 2,         # Çizgi kalınlığı
    "point_radius": 3,           # Nokta yarıçapı
    "spine_color": (0, 255, 0),  # Omurga rengi (BGR)
    "angle_color": (0, 0, 255),  # Açı rengi (BGR)
    "text_color": (255, 255, 255),  # Metin rengi (BGR)
    "font_scale": 0.6,           # Font ölçeği
}

# Pretrained modeller
PRETRAINED_MODELS = {
    "unet_backbone": "resnet34",  # U-Net omurga mimarisi
    "imagenet_weights": True,     # ImageNet ağırlıklarını kullan
    "model_path": os.path.join(PRETRAINED_DIR, "spine_segmentation_model.pth"),
}

# Desteklenen görüntü formatları
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# Sistem dizinlerini oluştur
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PRETRAINED_DIR,
                 OUTPUT_DIR, RESULTS_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)