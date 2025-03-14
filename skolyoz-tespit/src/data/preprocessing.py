# src/data/preprocessing.py

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class XRayPreprocessor:
    """
    X-ray görüntülerini segmentasyon için hazırlayan sınıf.
    Gürültü azaltma, kontrast ayarlama, ve vücut dışı alanları kaldırma işlemleri yapar.
    """

    def __init__(self,
                 clahe_clip_limit: float = 2.0,
                 clahe_grid_size: Tuple[int, int] = (8, 8),
                 denoise_strength: int = 10,
                 body_threshold: int = 15):
        """
        Ön işleme sınıfını başlatır.

        Args:
            clahe_clip_limit: CLAHE algoritması için kırpma limiti
            clahe_grid_size: CLAHE algoritması için ızgara boyutu
            denoise_strength: Gürültü azaltma gücü
            body_threshold: Vücut tespiti için eşik değeri
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.denoise_strength = denoise_strength
        self.body_threshold = body_threshold

        # CLAHE nesnesini oluştur
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Görüntünün kontrastını artırır.

        Args:
            image: Giriş görüntüsü (0-1 aralığında normalize edilmiş veya uint8)

        Returns:
            Kontrastı artırılmış görüntü
        """
        # Görüntüyü uint8'e çevir (eğer float ise)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image

        # CLAHE uygula
        enhanced = self.clahe.apply(image_uint8)

        # Görüntüyü orijinal veri tipine geri çevir
        if image.dtype == np.float32 or image.dtype == np.float64:
            enhanced = enhanced.astype(np.float32) / 255

        return enhanced

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Görüntüdeki gürültüyü azaltır.

        Args:
            image: Giriş görüntüsü

        Returns:
            Gürültüsü azaltılmış görüntü
        """
        # Görüntüyü uint8'e çevir (eğer float ise)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image

        # Bilateral filtre uygula (kenarları koruyarak gürültüyü azaltır)
        denoised = cv2.bilateralFilter(image_uint8, 9, self.denoise_strength, self.denoise_strength)

        # Görüntüyü orijinal veri tipine geri çevir
        if image.dtype == np.float32 or image.dtype == np.float64:
            denoised = denoised.astype(np.float32) / 255

        return denoised

    def extract_body_region(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        X-ray görüntüsünden vücut bölgesini çıkarır.

        Args:
            image: Giriş görüntüsü

        Returns:
            Tuple içinde (maskelenmiş görüntü, vücut maskesi)
        """
        # Görüntüyü uint8'e çevir (eğer float ise)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image

        # Otsu eşikleme ile vücut bölgesini bul
        _, body_mask = cv2.threshold(image_uint8, self.body_threshold, 255, cv2.THRESH_BINARY)

        # Morfolojik işlemler ile maskeyi temizle
        kernel = np.ones((5, 5), np.uint8)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)

        # En büyük konturu bul
        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # En büyük konturu bul
            largest_contour = max(contours, key=cv2.contourArea)

            # Yeni bir maske oluştur
            refined_mask = np.zeros_like(body_mask)
            cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)

            # Maskeyi uygula
            masked_image = cv2.bitwise_and(image_uint8, image_uint8, mask=refined_mask)

            # Görüntüyü orijinal veri tipine geri çevir
            if image.dtype == np.float32 or image.dtype == np.float64:
                masked_image = masked_image.astype(np.float32) / 255
                refined_mask = refined_mask.astype(np.float32) / 255

            return masked_image, refined_mask
        else:
            logger.warning("Vücut bölgesi tespit edilemedi, orijinal görüntü döndürülüyor")
            return image, np.ones_like(image)

    def detect_spine_region(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        X-ray görüntüsünde omurga bölgesini tespit etmeye çalışır.

        Args:
            image: Giriş görüntüsü

        Returns:
            Tuple içinde (kırpılmış omurga bölgesi, (x, y, w, h) koordinatları)
        """
        # Görüntüyü uint8'e çevir (eğer float ise)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image

        # Kenar tespiti
        edges = cv2.Canny(image_uint8, 50, 150)

        # Dikey kenarları vurgula
        kernel_vertical = np.ones((10, 1), np.uint8)
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vertical)

        # Omurga genellikle görüntünün ortasında olur
        height, width = image.shape
        center_x = width // 2

        # Omurga bölgesi için varsayılan koordinatlar (merkez etrafında)
        spine_width = width // 3
        x = center_x - (spine_width // 2)
        y = height // 6  # Üstten biraz aşağıda başla
        w = spine_width
        h = int(height * 2 / 3)  # Görüntünün 2/3'ü kadar yükseklik

        # Varsayılan bölgeyi kırp
        spine_region = image[y:y + h, x:x + w]

        return spine_region, (x, y, w, h)

    def apply_all(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Tüm ön işleme adımlarını uygular.

        Args:
            image: Giriş görüntüsü

        Returns:
            İşlenmiş görüntü ve ara sonuçları içeren sözlük
        """
        # Görüntüyü normalize et (eğer uint8 ise)
        if image.dtype == np.uint8:
            image_norm = image.astype(np.float32) / 255
        else:
            image_norm = image.copy()

        # Kontrastı artır
        enhanced = self.enhance_contrast(image_norm)

        # Gürültüyü azalt
        denoised = self.denoise(enhanced)

        # Vücut bölgesini çıkar
        body_image, body_mask = self.extract_body_region(denoised)

        # Omurga bölgesini tespit et
        spine_region, spine_bbox = self.detect_spine_region(body_image)

        # Sonuçları sözlükte topla
        results = {
            'original': image_norm,
            'enhanced': enhanced,
            'denoised': denoised,
            'body_image': body_image,
            'body_mask': body_mask,
            'spine_region': spine_region,
            'spine_bbox': spine_bbox
        }

        return results


# Test için
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..data.loader import XRayLoader

    # Logging konfigürasyonu
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Loader oluştur
    loader = XRayLoader()

    # Görüntüleri bul
    image_files = loader.find_xray_files()

    if len(image_files) > 0:
        # İlk görüntüyü yükle
        image_path = image_files[0]
        image = loader.load_image(image_path)

        if image is not None:
            # Ön işleyici oluştur
            preprocessor = XRayPreprocessor()

            # Tüm ön işleme adımlarını uygula
            results = preprocessor.apply_all(image)

            # Sonuçları görselleştir
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.imshow(results['original'], cmap='gray')
            plt.title('Orijinal')

            plt.subplot(2, 3, 2)
            plt.imshow(results['enhanced'], cmap='gray')
            plt.title('Kontrast Artırılmış')

            plt.subplot(2, 3, 3)
            plt.imshow(results['denoised'], cmap='gray')
            plt.title('Gürültüsü Azaltılmış')

            plt.subplot(2, 3, 4)
            plt.imshow(results['body_image'], cmap='gray')
            plt.title('Vücut Bölgesi')

            plt.subplot(2, 3, 5)
            plt.imshow(results['body_mask'], cmap='gray')
            plt.title('Vücut Maskesi')

            plt.subplot(2, 3, 6)
            plt.imshow(results['spine_region'], cmap='gray')
            plt.title('Omurga Bölgesi')

            plt.tight_layout()
            plt.show()
    else:
        logger.warning("Görüntü bulunamadı!")