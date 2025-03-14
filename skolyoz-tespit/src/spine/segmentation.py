# src/spine/segmentation.py

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

from ..models.unet import SpineSegmentationModel

logger = logging.getLogger(__name__)


class SpineSegmenter:
    """
    X-ray görüntülerinde omurga segmentasyonu yapan sınıf.
    """

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        """
        Omurga segmentasyon sınıfını başlatır.

        Args:
            model_path: Önceden eğitilmiş model yolu
            threshold: Segmentasyon eşik değeri
        """
        self.threshold = threshold
        self.model = SpineSegmentationModel(model_path)

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Görüntüde omurga segmentasyonu yapar.

        Args:
            image: Giriş görüntüsü (normalize edilmiş, [0,1] aralığında)

        Returns:
            (olasılık_maskesi, ikili_maske) çifti
        """
        # Görüntüyü segmente et
        probability_mask = self.model.segment_spine(image)

        # İkili maskeyi oluştur
        binary_mask = self.model.get_binary_mask(probability_mask, self.threshold)

        return probability_mask, binary_mask

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Segmentasyon maskesini iyileştirir.

        Args:
            mask: İkili segmentasyon maskesi

        Returns:
            İyileştirilmiş maske
        """
        # Maskeyi uint8'e çevir
        if mask.dtype != np.uint8:
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            binary_mask = mask.copy()

        # Morfolojik işlemler uygula
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # En büyük bağlı bileşeni al
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        if num_labels > 1:
            # Arka planı atla (0 indeksi), kalanlar arasında en büyüğünü bul
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary_mask = (labels == largest_label).astype(np.uint8)

        return binary_mask

    def remove_noise(self, mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """
        Maskeden küçük gürültüleri temizler.

        Args:
            mask: İkili segmentasyon maskesi
            min_area: Minimum bileşen alanı

        Returns:
            Temizlenmiş maske
        """
        # Maskeyi uint8'e çevir
        if mask.dtype != np.uint8:
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            binary_mask = mask.copy()

        # Bağlı bileşenleri bul
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        # Küçük bileşenleri kaldır
        filtered_mask = np.zeros_like(binary_mask)

        for i in range(1, num_labels):  # 0 indeksi arka plandır
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 1

        return filtered_mask


# Test için
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..data.loader import XRayLoader
    from ..data.preprocessing import XRayPreprocessor

    # Logging konfigürasyonu
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Gerekli nesneleri oluştur
    loader = XRayLoader()
    preprocessor = XRayPreprocessor()
    segmenter = SpineSegmenter()

    # Örnek bir görüntü yükle
    image_files = loader.find_xray_files()

    if image_files:
        # İlk görüntüyü işle
        image_path = image_files[0]
        logger.info(f"Görüntü yükleniyor: {image_path}")

        # Görüntüyü yükle ve ön işle
        image = loader.load_image(image_path)
        processed = preprocessor.apply_all(image)

        # Omurga segmentasyonu
        prob_mask, binary_mask = segmenter.segment(processed['enhanced'])

        # Maskeyi iyileştir
        refined_mask = segmenter.refine_mask(binary_mask)

        # Sonuçları göster
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(processed['enhanced'], cmap='gray')
        plt.title("İşlenmiş Görüntü")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(prob_mask, cmap='jet')
        plt.title("Olasılık Maskesi")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("İkili Maske")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(refined_mask, cmap='gray')
        plt.title("İyileştirilmiş Maske")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        logger.warning("Görüntü bulunamadı!")