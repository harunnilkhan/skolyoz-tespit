# src/spine/classic_detection.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from typing import Tuple, List, Dict, Optional, Union, Any
import logging
import math

logger = logging.getLogger(__name__)


class ClassicSpineDetector:
    """
    Geleneksel görüntü işleme teknikleri ile X-ray görüntülerinden
    omurga tespiti ve Cobb açısı hesaplama yapan sınıf.
    """

    def __init__(self,
                 clahe_clip_limit: float = 3.0,
                 smoothing_factor: int = 11,
                 poly_order: int = 3,
                 vertical_mask_ratio: float = 0.25):
        """
        Omurga dedektörünü başlatır.

        Args:
            clahe_clip_limit: CLAHE için kırpma limiti
            smoothing_factor: Eğri düzleştirme için pencere boyutu
            poly_order: Polinomial uydurmada derece
            vertical_mask_ratio: Dikey maskeleme oranı (merkezi vurgulamak için)
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.smoothing_factor = smoothing_factor
        self.poly_order = poly_order
        self.vertical_mask_ratio = vertical_mask_ratio

        # CLAHE için nesne oluştur
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))

    def detect_spine(self, image: np.ndarray) -> Dict[str, Any]:
        """
        X-ray görüntüsünden omurgayı tespit eder.

        Args:
            image: İşlenecek X-ray görüntüsü

        Returns:
            Omurga tespiti ve Cobb açısı sonuçlarını içeren sözlük
        """
        # Görüntüyü normalize et
        if image.dtype == np.float32 or image.dtype == np.float64:
            if np.max(image) <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
        else:
            image_uint8 = image.copy()

        # Görüntüyü iyileştir
        enhanced = self._enhance_image(image_uint8)

        # Omurga bölgesini tespit et
        spine_roi, spine_bbox = self._extract_spine_region(enhanced)

        # Omurga eğrisini tespit et
        centerline, curve_points = self._detect_spine_curve(spine_roi)

        # Eğer merkez hattı tespit edilemezse
        if centerline is None or len(curve_points) < 10:
            logger.warning("Omurga eğrisi tespit edilemedi")
            return {
                'success': False,
                'spine_mask': np.zeros_like(image_uint8),
                'spine_roi': spine_roi,
                'spine_bbox': spine_bbox,
                'curve_points': [],
                'cobb_angle': float('nan'),
                'upper_inflection': None,
                'lower_inflection': None,
                'message': "Omurga eğrisi tespit edilemedi"
            }

        # Eğriyi orijinal görüntü koordinatlarına dönüştür
        adjusted_curve_points = []
        for p in curve_points:
            adjusted_curve_points.append(
                (p[0] + spine_bbox[0], p[1] + spine_bbox[1])
            )

        # Omurga maskesi oluştur
        spine_mask = self._create_spine_mask(image_uint8.shape, adjusted_curve_points)

        # Cobb açısını hesapla
        cobb_result = self._calculate_cobb_angle(adjusted_curve_points)

        # Sonuç
        result = {
            'success': True,
            'spine_mask': spine_mask,
            'spine_roi': spine_roi,
            'spine_bbox': spine_bbox,
            'curve_points': adjusted_curve_points,
            'cobb_angle': cobb_result['angle'],
            'upper_inflection': cobb_result['upper_inflection'],
            'lower_inflection': cobb_result['lower_inflection'],
            'message': "Omurga tespit edildi"
        }

        return result

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        X-ray görüntüsünü iyileştirir.

        Args:
            image: Orijinal görüntü

        Returns:
            İyileştirilmiş görüntü
        """
        # CLAHE uygula (kontrastı artır)
        clahe_image = self.clahe.apply(image)

        # Gürültü azaltma
        blurred = cv2.GaussianBlur(clahe_image, (5, 5), 0)

        # Kenar tespiti
        edges = cv2.Canny(blurred, 50, 150)

        # Dikey kenarları vurgula
        kernel_vertical = np.ones((7, 1), np.uint8)
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vertical)

        # Sonucu geri döndür
        return vertical_edges

    def _extract_spine_region(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Görüntüden omurga bölgesini çıkarır.

        Args:
            image: İşlenmiş görüntü

        Returns:
            (spine_roi, (x, y, w, h)) şeklinde omurga bölgesi ve koordinatları
        """
        height, width = image.shape

        # Dikey maskeleme ile omurganın bulunduğu orta bölgeyi vurgula
        vertical_mask = np.zeros_like(image)
        mask_width = int(width * self.vertical_mask_ratio)
        start_x = (width - mask_width) // 2
        end_x = start_x + mask_width
        vertical_mask[:, start_x:end_x] = 255

        # Maskeyi uygula
        masked_image = cv2.bitwise_and(image, vertical_mask)

        # Omurga bölgesi koordinatları
        # (Görüntünün orta dikey bölgesi)
        x = max(0, start_x - 20)  # Biraz daha geniş tut
        w = min(width, mask_width + 40)
        y = 0
        h = height

        # Bölgeyi kırp
        spine_roi = masked_image[y:y + h, x:x + w]

        return spine_roi, (x, y, w, h)

    def _detect_spine_curve(self, spine_image: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]]]:
        """
        Omurga bölgesinden omurga eğrisini tespit eder.

        Args:
            spine_image: Omurga bölgesi görüntüsü

        Returns:
            (centerline, curve_points) şeklinde merkez çizgisi ve eğri noktaları
        """
        height, width = spine_image.shape

        # Her satırdaki beyaz piksellerin ortalamasını al
        centerline = np.zeros(height)
        curve_points = []

        for y in range(height):
            # Satırdaki beyaz piksellerin x koordinatlarını bul
            white_x = np.where(spine_image[y, :] > 0)[0]

            if len(white_x) > 0:
                # Ortalama x koordinatını hesapla
                center_x = np.mean(white_x)
                centerline[y] = center_x
                curve_points.append((int(center_x), y))

        # Eğer hiç nokta bulunamazsa
        if len(curve_points) < 10:
            return None, []

        # Centerline'daki boşlukları doldur
        valid_indices = np.where(centerline > 0)[0]
        if len(valid_indices) < 2:
            return None, []

        valid_values = centerline[valid_indices]
        centerline_filled = np.interp(np.arange(height), valid_indices, valid_values)

        # Centerline'ı düzleştir
        try:
            window_len = min(self.smoothing_factor, len(centerline_filled) - 2)
            if window_len % 2 == 0:
                window_len += 1  # Tek sayı olmalı

            centerline_smooth = savgol_filter(
                centerline_filled,
                window_length=window_len,
                polyorder=self.poly_order
            )

            # Yeni eğri noktaları oluştur
            curve_points = []
            for y in range(height):
                curve_points.append((int(centerline_smooth[y]), y))

            return centerline_smooth, curve_points

        except Exception as e:
            logger.error(f"Eğri düzleştirme hatası: {str(e)}")
            return centerline_filled, curve_points

    def _create_spine_mask(self, image_shape: Tuple[int, int], curve_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Eğri noktalarından omurga maskesi oluşturur.

        Args:
            image_shape: Orijinal görüntü şekli
            curve_points: Omurga eğrisi noktaları

        Returns:
            Omurga maskesi
        """
        if not curve_points:
            return np.zeros(image_shape, dtype=np.uint8)

        # Boş maske oluştur
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Eğrinin etrafında kalın bir çizgi çiz
        points = np.array(curve_points, dtype=np.int32)

        # Eğriyi çiz (kalın)
        for i in range(len(points) - 1):
            cv2.line(mask, tuple(points[i]), tuple(points[i + 1]), 255, thickness=15)

        # Morfolojik operasyonlar ile maskeyi iyileştir
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _calculate_cobb_angle(self, curve_points: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Omurga eğrisinden Cobb açısını hesaplar.

        Args:
            curve_points: Omurga eğrisi noktaları

        Returns:
            Cobb açısı bilgilerini içeren sözlük
        """
        if len(curve_points) < 10:
            return {
                'angle': float('nan'),
                'upper_inflection': None,
                'lower_inflection': None,
                'success': False
            }

        try:
            # Noktaları array'e dönüştür
            points = np.array(curve_points)
            x = points[:, 0]
            y = points[:, 1]

            # x koordinatlarının değişimini analiz et
            dx = np.gradient(x)

            # İşaret değişimlerini bul (büküm noktaları)
            sign_changes = []
            for i in range(1, len(dx)):
                if dx[i - 1] * dx[i] < 0:  # İşaret değişimi
                    sign_changes.append(i)

            # Eğer işaret değişimi yoksa, eğrinin en üst ve en alt noktalarını al
            if len(sign_changes) < 1:
                upper_idx = len(curve_points) // 4
                lower_idx = 3 * len(curve_points) // 4
            else:
                # İşaret değişimlerinin ortasını al
                midpoint = len(sign_changes) // 2
                inflection_idx = sign_changes[midpoint]

                # Üst ve alt noktaları belirle
                upper_idx = max(0, inflection_idx - len(curve_points) // 4)
                lower_idx = min(len(curve_points) - 1, inflection_idx + len(curve_points) // 4)

            # Eğim hesapla
            def calculate_slope(p1, p2):
                x1, y1 = p1
                x2, y2 = p2
                if y2 - y1 == 0:
                    return float('inf')
                return (x2 - x1) / (y2 - y1)

            # Üst ve alt segment eğimleri
            upper_segment = curve_points[max(0, upper_idx - 5):upper_idx + 5]
            lower_segment = curve_points[max(0, lower_idx - 5):lower_idx + 5]

            if len(upper_segment) < 2 or len(lower_segment) < 2:
                return {
                    'angle': float('nan'),
                    'upper_inflection': None,
                    'lower_inflection': None,
                    'success': False
                }

            upper_p1 = upper_segment[0]
            upper_p2 = upper_segment[-1]
            lower_p1 = lower_segment[0]
            lower_p2 = lower_segment[-1]

            upper_slope = calculate_slope(upper_p1, upper_p2)
            lower_slope = calculate_slope(lower_p1, lower_p2)

            # Cobb açısını hesapla
            if np.isinf(upper_slope) and np.isinf(lower_slope):
                angle = 0.0
            elif np.isinf(upper_slope):
                angle = 90.0 - abs(np.degrees(np.arctan(lower_slope)))
            elif np.isinf(lower_slope):
                angle = 90.0 - abs(np.degrees(np.arctan(upper_slope)))
            else:
                # İki eğim arasındaki açıyı hesapla
                tangent = abs((upper_slope - lower_slope) / (1 + upper_slope * lower_slope))
                angle = np.degrees(np.arctan(tangent))

            return {
                'angle': angle,
                'upper_inflection': upper_segment[len(upper_segment) // 2],
                'lower_inflection': lower_segment[len(lower_segment) // 2],
                'upper_slope': upper_slope,
                'lower_slope': lower_slope,
                'success': True
            }

        except Exception as e:
            logger.error(f"Cobb açısı hesaplama hatası: {str(e)}")
            return {
                'angle': float('nan'),
                'upper_inflection': None,
                'lower_inflection': None,
                'success': False
            }

    def visualize_results(self, image: np.ndarray,
                          result: Dict[str, Any],
                          output_path: Optional[str] = None,
                          show_plot: bool = True) -> np.ndarray:
        """
        Tespit sonuçlarını görselleştirir.

        Args:
            image: Orijinal görüntü
            result: Tespit sonuçları
            output_path: Çıktı dosya yolu
            show_plot: Grafiği göster/gösterme

        Returns:
            Görselleştirilmiş sonuç görüntüsü
        """
        # Görüntüyü BGR formatına dönüştür
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_image = image.copy()

        # Eğer tespit başarılıysa
        if result['success']:
            # Omurga eğrisini çiz
            curve_points = result['curve_points']
            for i in range(len(curve_points) - 1):
                cv2.line(display_image, curve_points[i], curve_points[i + 1], (0, 255, 0), 2)

            # Kritik noktaları çiz
            if result['upper_inflection'] is not None:
                cv2.circle(display_image, result['upper_inflection'], 5, (0, 0, 255), -1)

            if result['lower_inflection'] is not None:
                cv2.circle(display_image, result['lower_inflection'], 5, (0, 0, 255), -1)

            # Cobb açısını yaz
            angle = result['cobb_angle']
            if not np.isnan(angle):
                text = f"Cobb Açısı: {angle:.1f}°"
                cv2.putText(display_image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)
        else:
            # Hata mesajını yaz
            cv2.putText(display_image, "Omurga tespit edilemedi", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Sonucu kaydet
        if output_path is not None:
            cv2.imwrite(output_path, display_image)

        # Göster
        if show_plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return display_image


# Test için
if __name__ == "__main__":
    import os

    # Logging konfigürasyonu
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Test görüntüsü yükle (örnek)
    test_image_path = "../../data/raw/sample.jpg"  # Bu yolu kendi test görüntünüze göre değiştirin

    if os.path.exists(test_image_path):
        # Görüntüyü yükle
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

        # Spine dedektörü oluştur
        detector = ClassicSpineDetector()

        # Omurga tespiti yap
        result = detector.detect_spine(image)

        # Sonuçları görselleştir
        detector.visualize_results(image, result)

        # Sonuçları yazdır
        if result['success']:
            print(f"Cobb Açısı: {result['cobb_angle']:.1f}°")
        else:
            print("Omurga tespit edilemedi.")
    else:
        print(f"Test görüntüsü bulunamadı: {test_image_path}")