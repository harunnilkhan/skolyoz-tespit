# src/spine/curve_detection.py

import numpy as np
import cv2
import scipy.signal
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class SpineCurveDetector:
    """
    Omurga eğrisini tespit eden sınıf.
    """

    def __init__(self, num_points: int = 50, smoothing_factor: float = 0.85):
        """
        Omurga eğrisi dedektörünü başlatır.

        Args:
            num_points: Eğrideki nokta sayısı
            smoothing_factor: Eğri düzleştirme faktörü (0-1 arası, 1 daha düz)
        """
        self.num_points = num_points
        self.smoothing_factor = smoothing_factor

    def detect_curve(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Maske görüntüsünden omurga eğrisini tespit eder.

        Args:
            mask: İkili omurga maskesi

        Returns:
            Omurga eğri noktaları veya None
        """
        # Maskeyi normalleştir
        if mask.dtype != np.uint8:
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            binary_mask = mask.copy()

        # İnceltme işlemi uygula (skeletonization)
        skeleton = self._skeletonize(binary_mask)

        if np.sum(skeleton) == 0:
            logger.warning("Omurga iskeletleştirme başarısız oldu, alternatif yöntem deneniyor.")
            return self._detect_centerline(binary_mask)

        # İskeletten eğri noktalarını oluştur
        curve_points = self._extract_curve_from_skeleton(skeleton)

        if curve_points is None or len(curve_points) < 3:
            logger.warning("İskeletten eğri çıkarılamadı, alternatif yöntem deneniyor.")
            return self._detect_centerline(binary_mask)

        # Noktaları düzleştir
        curve_points = self._smooth_curve(curve_points)

        # Noktaları yeniden örnekle
        curve_points = self._resample_curve(curve_points, self.num_points)

        return np.array(curve_points)

    def _skeletonize(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Maskeyi iskeletleştirir.

        Args:
            binary_mask: İkili maske

        Returns:
            İskeletleştirilmiş maske
        """
        # Zhang-Suen thinning algoritması
        skeleton = np.zeros_like(binary_mask)

        try:
            # OpenCV'nin yöntemini kullan (daha hızlı)
            skeleton = cv2.ximgproc.thinning(binary_mask)
        except:
            # OpenCV yöntemi kullanılamazsa scikit-image'ı dene
            try:
                from skimage.morphology import skeletonize
                skeleton = skeletonize(binary_mask).astype(np.uint8)
            except:
                # Hiçbiri kullanılamazsa basit bir iskeletleştirme uygula
                kernel = np.ones((3, 3), np.uint8)
                eroded = binary_mask.copy()
                temp = np.zeros_like(binary_mask)

                while np.sum(eroded) > 0:
                    eroded_prev = eroded.copy()
                    eroded = cv2.erode(eroded, kernel)
                    temp = cv2.subtract(eroded_prev, eroded)
                    skeleton = cv2.bitwise_or(skeleton, temp)

        return skeleton

    def _extract_curve_from_skeleton(self, skeleton: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        İskeletten eğri noktalarını çıkarır.

        Args:
            skeleton: İskeletleştirilmiş maske

        Returns:
            Eğri noktaları veya None
        """
        # İskeletin bağlı bileşenlerini bul
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

        if num_labels <= 1:
            return None

        # En büyük bileşeni bul (arka plan hariç)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component = (labels == largest_label).astype(np.uint8)

        # İskeletin uç noktalarını bul
        endpoints = self._find_endpoints(largest_component)

        if len(endpoints) < 2:
            # Uç nokta bulunamazsa, y ekseni boyunca ortala
            height, width = skeleton.shape
            curve_points = []

            for y in range(height):
                x_coords = np.where(largest_component[y, :] > 0)[0]
                if len(x_coords) > 0:
                    x_center = int(np.mean(x_coords))
                    curve_points.append((x_center, y))

            return curve_points if len(curve_points) >= 3 else None

        # En uzak iki uç noktayı bul
        max_dist = 0
        start_point = endpoints[0]
        end_point = endpoints[0]

        for i, p1 in enumerate(endpoints):
            for p2 in endpoints[i + 1:]:
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if dist > max_dist:
                    max_dist = dist
                    start_point = p1
                    end_point = p2

        # Uç noktaları birleştiren eğriyi bul
        curve_points = self._trace_curve(largest_component, start_point, end_point)

        return curve_points

    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        İskeletin uç noktalarını bulur.

        Args:
            skeleton: İskeletleştirilmiş maske

        Returns:
            Uç noktaların listesi
        """
        # 8-komşuluk kernel
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ], dtype=np.uint8)

        # Konvolüsyon uygula
        conv = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)

        # Uç noktalar: merkezdeki piksel 1, komşuluklardan sadece 1 piksel 1 olan yerlerdir (toplam değer 11)
        endpoints = np.where((conv == 11) & (skeleton > 0))

        return list(zip(endpoints[1], endpoints[0]))  # (x, y) formatında döndür

    def _trace_curve(self, skeleton: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        İskeletten iki nokta arasındaki eğriyi takip eder.

        Args:
            skeleton: İskeletleştirilmiş maske
            start: Başlangıç noktası
            end: Bitiş noktası

        Returns:
            Eğri noktaları
        """
        # TODO: Burada tam bir eğri takip algoritması uygulanabilir
        # Basit bir yaklaşım: İskeleti yukarıdan aşağıya tara
        height, width = skeleton.shape
        curve_points = []

        for y in range(height):
            x_coords = np.where(skeleton[y, :] > 0)[0]
            if len(x_coords) > 0:
                x_center = int(np.mean(x_coords))
                curve_points.append((x_center, y))

        return curve_points

    def _detect_centerline(self, binary_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Merkez hattı yöntemiyle omurga eğrisini tespit eder.

        Args:
            binary_mask: İkili maske

        Returns:
            Eğri noktaları
        """
        height, width = binary_mask.shape

        # Her satırdaki merkez noktayı hesapla
        curve_points = []

        for y in range(height):
            # Satırdaki beyaz piksellerin x koordinatlarını bul
            white_pixels = np.where(binary_mask[y, :] > 0)[0]

            if len(white_pixels) > 0:
                # Ortalama x koordinatını hesapla
                x_center = int(np.mean(white_pixels))
                curve_points.append((x_center, y))

        if len(curve_points) < 3:
            logger.warning("Yeterli omurga noktası bulunamadı")
            return []

        # Noktaları düzleştir
        curve_points = self._smooth_curve(curve_points)

        # Eşit aralıklı noktalar seç
        curve_points = self._resample_curve(curve_points, self.num_points)

        return curve_points

    def _smooth_curve(self, curve_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Eğri noktalarını düzleştirir.

        Args:
            curve_points: Eğri noktaları

        Returns:
            Düzleştirilmiş eğri noktaları
        """
        if len(curve_points) < 3:
            return curve_points

        # Noktaları x, y dizilerine ayır
        points = np.array(curve_points)
        x_points = points[:, 0]
        y_points = points[:, 1]

        # Savitzky-Golay filtresi uygula
        window_length = min(15, len(x_points) - (1 - len(x_points) % 2))
        if window_length < 3:
            return curve_points

        # Pencere uzunluğunu tek sayı yap
        if window_length % 2 == 0:
            window_length -= 1

        try:
            x_smoothed = scipy.signal.savgol_filter(x_points, window_length, 3)

            # Noktaları yeniden oluştur
            smoothed_points = []
            for i in range(len(y_points)):
                smoothed_points.append((int(x_smoothed[i]), int(y_points[i])))

            return smoothed_points

        except Exception as e:
            logger.error(f"Eğri düzleştirme hatası: {str(e)}")
            return curve_points

    def _resample_curve(self, curve_points: List[Tuple[int, int]], num_points: int) -> List[Tuple[int, int]]:
        """
        Eğri noktalarını eşit aralıklı olarak yeniden örnekler.

        Args:
            curve_points: Eğri noktaları
            num_points: İstenen nokta sayısı

        Returns:
            Yeniden örneklenmiş eğri noktaları
        """
        if len(curve_points) <= num_points:
            return curve_points

        # Eşit aralıklı indeksler seç
        indices = np.linspace(0, len(curve_points) - 1, num_points, dtype=int)
        resampled_points = [curve_points[i] for i in indices]

        return resampled_points

    def find_inflection_points(self, curve_points: np.ndarray) -> List[int]:
        """
        Eğri üzerindeki büküm noktalarını (inflection points) bulur.

        Args:
            curve_points: Eğri noktaları

        Returns:
            Büküm noktalarının indeksleri
        """
        if len(curve_points) < 5:
            return []

        # Eğrinin x koordinatlarını al
        x = curve_points[:, 0]

        # Birinci türev
        dx = np.gradient(x)

        # İkinci türev
        ddx = np.gradient(dx)

        # İşaret değiştiren noktalarda büküm noktası vardır
        inflection_indices = []

        for i in range(1, len(ddx)):
            if ddx[i - 1] * ddx[i] < 0:  # İşaret değişimi
                inflection_indices.append(i)

        return inflection_indices


# Test için
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..data.loader import XRayLoader
    from ..data.preprocessing import XRayPreprocessor
    from ..models.unet import SpineSegmentationModel

    # Logging konfigürasyonu
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Gerekli nesneleri oluştur
    loader = XRayLoader()
    preprocessor = XRayPreprocessor()
    model = SpineSegmentationModel()
    curve_detector = SpineCurveDetector()

    # Görüntüleri bul
    image_files = loader.find_xray_files()

    if image_files:
        # İlk görüntüyü işle
        image_path = image_files[0]
        logger.info(f"Görüntü yükleniyor: {image_path}")

        # Görüntüyü yükle ve ön işle
        image = loader.load_image(image_path)
        processed = preprocessor.apply_all(image)

        # Omurga segmentasyonu
        spine_mask = model.segment_spine(processed['enhanced'])
        binary_mask = model.get_binary_mask(spine_mask)

        # Omurga eğrisini tespit et
        curve_points = curve_detector.detect_curve(binary_mask)

        # Büküm noktalarını bul
        if curve_points is not None and len(curve_points) > 0:
            inflection_indices = curve_detector.find_inflection_points(curve_points)

            # Sonuçları göster
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(processed['enhanced'], cmap='gray')
            plt.title("İşlenmiş Görüntü")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(binary_mask, cmap='gray')
            plt.title("Omurga Maskesi")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(processed['enhanced'], cmap='gray')

            if len(curve_points) > 0:
                # Eğriyi çiz
                plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2)

                # Büküm noktalarını işaretle
                for idx in inflection_indices:
                    plt.plot(curve_points[idx, 0], curve_points[idx, 1], 'ro', markersize=8)

            plt.title("Omurga Eğrisi ve Büküm Noktaları")
            plt.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            logger.warning("Omurga eğrisi tespit edilemedi!")
    else:
        logger.warning("Görüntü bulunamadı!")