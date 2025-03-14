# src/utils/helpers.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


def ensure_dir(dir_path: str) -> bool:
    """
    Dizinin var olduğundan emin olur, yoksa oluşturur.

    Args:
        dir_path: Dizin yolu

    Returns:
        Başarı durumu
    """
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Dizin oluşturuldu: {dir_path}")
        return True
    except Exception as e:
        logger.error(f"Dizin oluşturulurken hata: {dir_path}, {str(e)}")
        return False


def check_gpu_availability() -> Tuple[bool, str]:
    """
    GPU kullanılabilirliğini kontrol eder.

    Returns:
        (gpu_available, device_name) çifti
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU kullanılabilir: {device_name} (Toplam: {device_count})")
        return True, device_name
    else:
        logger.info("GPU kullanılamıyor, CPU kullanılacak")
        return False, "CPU"


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Görüntüyü 0-1 aralığına normalize eder.

    Args:
        image: Görüntü verileri

    Returns:
        Normalize edilmiş görüntü
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val < 1e-6:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_val) / (max_val - min_val)


def pad_to_square(image: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """
    Görüntüyü kare haline getirmek için kenarları doldurur.

    Args:
        image: Görüntü verileri
        pad_value: Dolgu değeri

    Returns:
        Kare haline getirilmiş görüntü
    """
    h, w = image.shape[:2]

    if h == w:
        return image

    size = max(h, w)

    if len(image.shape) == 3:
        padded = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        padded = np.full((size, size), pad_value, dtype=image.dtype)

    # Görüntüyü ortala
    h_offset = (size - h) // 2
    w_offset = (size - w) // 2

    padded[h_offset:h_offset + h, w_offset:w_offset + w] = image

    return padded


def crop_to_content(image: np.ndarray, threshold: int = 10, padding: int = 10) -> np.ndarray:
    """
    Görüntüyü içeriğe göre kırpar (boş kenarları kaldırır).

    Args:
        image: Görüntü verileri
        threshold: Eşik değeri
        padding: Kırpma sonrası eklenecek kenar boşluğu

    Returns:
        Kırpılmış görüntü
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif image.dtype == np.float32:
        gray = (image * 255).astype(np.uint8)
    else:
        gray = image

    # İkili eşikleme
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # İçerik koordinatlarını bul
    coords = cv2.findNonZero(binary)

    if coords is None:
        return image

    # Sınırlayıcı kutuyu hesapla
    x, y, w, h = cv2.boundingRect(coords)

    # Kenar boşluğu ekle
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    # Görüntüyü kırp
    cropped = image[y:y + h, x:x + w]

    return cropped


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 alpha: float = 0.5) -> np.ndarray:
    """
    Maskeyi görüntünün üzerine belirli bir renkle yerleştirir.

    Args:
        image: Orijinal görüntü
        mask: İkili maske
        color: Maske rengi (BGR formatında)
        alpha: Maske opaklığı (0-1 arası)

    Returns:
        Maskeli görüntü
    """
    # Görüntü ve maskeyi doğru formata çevir
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()

    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # Görüntüyü BGR formatına çevir
    if len(image_uint8.shape) == 2:
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image_uint8

    # Maskeyi BGR formatına çevir
    mask_bgr = np.zeros_like(image_bgr, dtype=np.uint8)
    mask_bgr[mask_uint8 > 0] = color

    # Maskeyi görüntüye ekle
    return cv2.addWeighted(image_bgr, 1, mask_bgr, alpha, 0)


def draw_spine_curve(image: np.ndarray, curve_points: List[Tuple[int, int]],
                     color: Tuple[int, int, int] = (0, 0, 255),
                     thickness: int = 2) -> np.ndarray:
    """
    Omurga eğrisini görüntü üzerine çizer.

    Args:
        image: Görüntü
        curve_points: Eğri noktaları
        color: Çizgi rengi (BGR formatında)
        thickness: Çizgi kalınlığı

    Returns:
        Eğri çizilmiş görüntü
    """
    if not curve_points:
        return image

    # Görüntüyü doğru formata çevir
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()

    # Görüntüyü BGR formatına çevir
    if len(image_uint8.shape) == 2:
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image_uint8

    # Noktaları çiz
    points = np.array(curve_points, dtype=np.int32)

    # Eğriyi çiz
    for i in range(len(points) - 1):
        cv2.line(image_bgr, tuple(points[i]), tuple(points[i + 1]), color, thickness)

    return image_bgr


def save_batch_results(results: List[Dict[str, Any]],
                       output_dir: str,
                       prefix: str = "batch_results") -> str:
    """
    Toplu işlem sonuçlarını kaydeder.

    Args:
        results: Sonuç sözlükleri listesi
        output_dir: Çıktı dizini
        prefix: Dosya adı öneki

    Returns:
        Kaydedilen CSV dosyasının yolu
    """
    import csv
    import datetime

    if not results:
        logger.warning("Kaydedilecek sonuç yok!")
        return ""

    # Çıktı dizinini kontrol et
    ensure_dir(output_dir)

    # Dosya adını oluştur
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{prefix}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # CSV başlıklarını belirle
    headers = ["image_path", "cobb_angle", "success", "processing_time"]

    # CSV dosyasını oluştur
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for result in results:
            row = {
                "image_path": result.get("image_path", ""),
                "cobb_angle": f"{result.get('cobb_angle', float('nan')):.1f}" if result.get('cobb_success',
                                                                                            False) else "N/A",
                "success": "Evet" if result.get('cobb_success', False) else "Hayır",
                "processing_time": f"{result.get('processing_time', 0):.2f}"
            }
            writer.writerow(row)

    logger.info(f"Sonuçlar CSV dosyasına kaydedildi: {csv_path}")

    return csv_path


def print_system_info() -> Dict[str, str]:
    """
    Sistem bilgilerini yazdırır.

    Returns:
        Sistem bilgilerini içeren sözlük
    """
    import platform
    import torch
    import cv2
    import numpy as np

    info = {
        "Python Versiyonu": platform.python_version(),
        "İşletim Sistemi": f"{platform.system()} {platform.release()}",
        "OpenCV Versiyonu": cv2.__version__,
        "NumPy Versiyonu": np.__version__,
        "PyTorch Versiyonu": torch.__version__
    }

    # GPU bilgisi
    if torch.cuda.is_available():
        info["GPU"] = torch.cuda.get_device_name(0)
        info["CUDA Versiyonu"] = torch.version.cuda
    else:
        info["GPU"] = "Kullanılamıyor"

    # Bilgileri yazdır
    logger.info("Sistem Bilgileri:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    return info


# Test için
if __name__ == "__main__":
    # Logging konfigürasyonu
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Sistem bilgilerini göster
    print_system_info()

    # GPU kullanılabilirliğini kontrol et
    gpu_available, device_name = check_gpu_availability()

    # Örnek bir görüntü oluştur
    image = np.zeros((100, 150), dtype=np.uint8)
    image[20:80, 30:120] = 255

    # Görüntüyü kare yap
    squared = pad_to_square(image)
    logger.info(f"Orijinal boyut: {image.shape}, Kare boyut: {squared.shape}")

    # Görüntüyü içeriğe göre kırp
    cropped = crop_to_content(image, padding=5)
    logger.info(f"Kırpılmış boyut: {cropped.shape}")

    # Görüntüleri göster
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Orijinal")

    plt.subplot(1, 3, 2)
    plt.imshow(squared, cmap='gray')
    plt.title("Kare")

    plt.subplot(1, 3, 3)
    plt.imshow(cropped, cmap='gray')
    plt.title("Kırpılmış")

    plt.tight_layout()
    plt.show()