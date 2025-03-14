# src/spine/cobb_angle.py

import numpy as np
import cv2
import scipy.signal
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict, Optional, Union
import logging
import matplotlib.pyplot as plt
from ..config import COBB_ANGLE_PARAMS

logger = logging.getLogger(__name__)


class SpineCurveDetector:
    """
    Segmentasyon maskesinden omurga eğrisini tespit eden sınıf.
    """

    def __init__(self,
                 num_points: int = COBB_ANGLE_PARAMS['num_points'],
                 smoothing_factor: float = COBB_ANGLE_PARAMS['smoothing_factor'],
                 curve_detection_method: str = COBB_ANGLE_PARAMS['curve_detection_method']):
        """
        Omurga eğrisi tespit ediciyi başlatır.

        Args:
            num_points: Omurga eğrisindeki nokta sayısı
            smoothing_factor: Eğri düzleştirme faktörü (0-1 arası, 1 daha düz)
            curve_detection_method: Eğri tespit yöntemi ('centerline', 'polynomial', 'spline')
        """
        self.num_points = num_points
        self.smoothing_factor = smoothing_factor
        self.curve_detection_method = curve_detection_method

    def detect_curve_points(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Omurga maskesinden eğri noktalarını tespit eder.

        Args:
            mask: İkili omurga maskesi

        Returns:
            Omurga eğri noktaları [(x1,y1), (x2,y2), ...] veya None
        """
        # Maskeyi ikili formata çevir
        if mask.dtype != np.uint8:
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            binary_mask = mask.copy()

        # Merkez hattı bul
        if self.curve_detection_method == 'centerline':
            return self._detect_centerline(binary_mask)
        elif self.curve_detection_method == 'polynomial':
            return self._detect_polynomial_curve(binary_mask)
        elif self.curve_detection_method == 'spline':
            return self._detect_spline_curve(binary_mask)
        else:
            logger.error(f"Bilinmeyen eğri tespit yöntemi: {self.curve_detection_method}")
            return self._detect_centerline(binary_mask)  # Varsayılan yöntem

    def _detect_centerline(self, binary_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Merkez hattı yöntemiyle omurga eğrisini tespit eder.
        Maskenin her satırındaki beyaz piksellerin ortalamasını alır.

        Args:
            binary_mask: İkili omurga maskesi

        Returns:
            Omurga eğri noktaları veya None
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
            return None

        # Noktaları düzleştir
        curve_points = self._smooth_curve(curve_points)

        # Eşit aralıklı noktaları seç
        curve_points = self._resample_curve(curve_points, self.num_points)

        return np.array(curve_points)

    def _detect_polynomial_curve(self, binary_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Polinom uydurma yöntemiyle omurga eğrisini tespit eder.

        Args:
            binary_mask: İkili omurga maskesi

        Returns:
            Omurga eğri noktaları veya None
        """
        # Önce merkez hattı yöntemiyle noktaları bul
        centerline_points = self._detect_centerline(binary_mask)

        if centerline_points is None or len(centerline_points) < 3:
            return None

        try:
            # Noktaları x, y dizilerine ayır
            x_points = centerline_points[:, 0]
            y_points = centerline_points[:, 1]

            # 3. dereceden polinom uydur (skolyoz S şeklinde olabilir)
            polynomial = np.polyfit(y_points, x_points, 3)

            # Eşit aralıklı y değerleri oluştur
            y_values = np.linspace(np.min(y_points), np.max(y_points), self.num_points)

            # Polinomu kullanarak x değerlerini hesapla
            x_values = np.polyval(polynomial, y_values)

            # Noktaları birleştir
            curve_points = np.column_stack((x_values, y_values)).astype(int)

            return curve_points

        except Exception as e:
            logger.error(f"Polinom uydurma hatası: {str(e)}")
            return centerline_points  # Hata durumunda merkez hattı noktalarını döndür

    def _detect_spline_curve(self, binary_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Spline uydurma yöntemiyle omurga eğrisini tespit eder.

        Args:
            binary_mask: İkili omurga maskesi

        Returns:
            Omurga eğri noktaları veya None
        """
        # Önce merkez hattı yöntemiyle noktaları bul
        centerline_points = self._detect_centerline(binary_mask)

        if centerline_points is None or len(centerline_points) < 3:
            return None

        try:
            # Noktaları x, y dizilerine ayır
            x_points = centerline_points[:, 0]
            y_points = centerline_points[:, 1]

            # Eşit aralıklı parametrelendirme
            t = np.linspace(0, 1, len(x_points))

            # Spline uydur
            x_spline = interp1d(t, x_points, kind='cubic')
            y_spline = interp1d(t, y_points, kind='cubic')

            # Yeni noktaları hesapla
            t_new = np.linspace(0, 1, self.num_points)
            x_values = x_spline(t_new)
            y_values = y_spline(t_new)

            # Noktaları birleştir
            curve_points = np.column_stack((x_values, y_values)).astype(int)

            return curve_points

        except Exception as e:
            logger.error(f"Spline uydurma hatası: {str(e)}")
            return centerline_points  # Hata durumunda merkez hattı noktalarını döndür

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


class CobbAngleCalculator:
    """
    Omurga eğrisinden Cobb açısını hesaplayan sınıf.
    """

    def __init__(self):
        """
        Cobb açısı hesaplayıcıyı başlatır.
        """
        self.curve_detector = SpineCurveDetector()

    def calculate_angle(self, mask: np.ndarray) -> Dict[str, Union[float, List[Tuple[int, int]]]]:
        """
        Omurga maskesinden Cobb açısını hesaplar.

        Args:
            mask: İkili omurga maskesi

        Returns:
            Cobb açısı bilgilerini içeren sözlük
        """
        # Omurga eğrisini tespit et
        curve_points = self.curve_detector.detect_curve_points(mask)

        if curve_points is None or len(curve_points) < 3:
            logger.warning("Cobb açısı hesaplanamadı, yeterli omurga noktası bulunamadı")
            return {
                'angle': float('nan'),
                'curve_points': [],
                'inflection_point': None,
                'upper_end_vertebra': None,
                'lower_end_vertebra': None,
                'success': False
            }

        # Omurga eğrisini analiz et
        try:
            # Inflection point ve end vertebraları tespit et
            inflection_idx, upper_idx, lower_idx = self._find_critical_points(curve_points)

            inflection_point = curve_points[inflection_idx]
            upper_end_vertebra = curve_points[upper_idx]
            lower_end_vertebra = curve_points[lower_idx]

            # Üst ve alt eğim çizgilerini hesapla
            upper_slope = self._calculate_slope(curve_points[upper_idx - 1], curve_points[upper_idx + 1])
            lower_slope = self._calculate_slope(curve_points[lower_idx - 1], curve_points[lower_idx + 1])

            # Açıyı hesapla
            angle = self._calculate_cobb_angle(upper_slope, lower_slope)

            return {
                'angle': angle,
                'curve_points': curve_points.tolist(),
                'inflection_point': inflection_point.tolist(),
                'upper_end_vertebra': upper_end_vertebra.tolist(),
                'lower_end_vertebra': lower_end_vertebra.tolist(),
                'upper_slope': upper_slope,
                'lower_slope': lower_slope,
                'success': True
            }

        except Exception as e:
            logger.error(f"Cobb açısı hesaplama hatası: {str(e)}")
            return {
                'angle': float('nan'),
                'curve_points': curve_points.tolist() if curve_points is not None else [],
                'inflection_point': None,
                'upper_end_vertebra': None,
                'lower_end_vertebra': None,
                'success': False
            }

    def _find_critical_points(self, curve_points: np.ndarray) -> Tuple[int, int, int]:
        """
        Omurga eğrisindeki kritik noktaları (inflection point ve end vertebraları) bulur.

        Args:
            curve_points: Omurga eğri noktaları

        Returns:
            (inflection_idx, upper_idx, lower_idx) indeksleri
        """
        # Eğrinin türevini hesapla
        x = curve_points[:, 0]

        # Birinci türev (eğim)
        dx = np.gradient(x)

        # İkinci türev (eğriliğin değişimi)
        ddx = np.gradient(dx)

        # Inflection point (eğriliğin değiştiği nokta)
        inflection_idx = np.argmax(np.abs(ddx))

        # Üst ve alt end vertebraları bul
        upper_segment = curve_points[:inflection_idx + 1]
        lower_segment = curve_points[inflection_idx:]

        if len(upper_segment) < 3 or len(lower_segment) < 3:
            # Yeterli nokta yoksa, basit bir yaklaşım kullan
            upper_idx = max(0, len(curve_points) // 4)
            lower_idx = min(len(curve_points) - 1, len(curve_points) * 3 // 4)
        else:
            # En büyük eğim değişimine sahip noktaları bul
            upper_dx = np.gradient(upper_segment[:, 0])
            lower_dx = np.gradient(lower_segment[:, 0])

            upper_idx = np.argmax(np.abs(upper_dx))
            lower_idx = inflection_idx + np.argmax(np.abs(lower_dx))

        return inflection_idx, upper_idx, lower_idx

    def _calculate_slope(self, point1: Union[Tuple[int, int], np.ndarray],
                         point2: Union[Tuple[int, int], np.ndarray]) -> float:
        """
        İki nokta arasındaki eğimi hesaplar.

        Args:
            point1: Birinci nokta (x1, y1)
            point2: İkinci nokta (x2, y2)

        Returns:
            Eğim değeri
        """
        x1, y1 = point1
        x2, y2 = point2

        # y ekseninde değişim yoksa, dikey çizgi (sonsuz eğim)
        if y2 - y1 == 0:
            return float('inf')

        return (x2 - x1) / (y2 - y1)

    def _calculate_cobb_angle(self, slope1: float, slope2: float) -> float:
        """
        İki eğim arasındaki açıyı hesaplar.

        Args:
            slope1: Birinci eğim
            slope2: İkinci eğim

        Returns:
            Açı değeri (derece cinsinden)
        """
        # Sonsuz eğim durumunu kontrol et
        if np.isinf(slope1) and np.isinf(slope2):
            return 0.0

        if np.isinf(slope1):
            angle = 90 - np.degrees(np.arctan(slope2))
        elif np.isinf(slope2):
            angle = 90 - np.degrees(np.arctan(slope1))
        else:
            # İki eğim arasındaki açıyı hesapla
            angle_rad = np.arctan(np.abs((slope2 - slope1) / (1 + slope1 * slope2)))
            angle = np.degrees(angle_rad)

        return angle


# Test için
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..data.loader import XRayLoader
    from ..data.preprocessing import XRayPreprocessor
    from ..models.unet import SpineSegmentationModel

    # Logging konfigürasyonu
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Loader, önişleyici ve model oluştur
    loader = XRayLoader()
    preprocessor = XRayPreprocessor()
    model = SpineSegmentationModel()

    # Görüntüleri bul
    image_files = loader.find_xray_files()

    if len(image_files) > 0:
        # İlk görüntüyü yükle ve işle
        image_path = image_files[0]
        image = loader.load_image(image_path)

        if image is not None:
            # Görüntüyü önişlemeden geçir
            processed = preprocessor.apply_all(image)

            # Omurga segmentasyonu
            spine_mask = model.segment_spine(processed['enhanced'])
            binary_mask = model.get_binary_mask(spine_mask)

            # Cobb açısını hesapla
            cobb_calculator = CobbAngleCalculator()
            cobb_result = cobb_calculator.calculate_angle(binary_mask)

            # Sonuçları görselleştir
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(processed['enhanced'], cmap='gray')
            plt.title('İşlenmiş Görüntü')

            plt.subplot(2, 2, 2)
            plt.imshow(binary_mask, cmap='gray')
            plt.title('Omurga Maskesi')

            plt.subplot(2, 2, 3)
            plt.imshow(processed['enhanced'], cmap='gray')

            if cobb_result['success']:
                # Eğri noktalarını göster
                curve_points = np.array(cobb_result['curve_points'])
                plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2)

                # Kritik noktaları göster
                inflection = cobb_result['inflection_point']
                upper = cobb_result['upper_end_vertebra']
                lower = cobb_result['lower_end_vertebra']

                plt.plot(inflection[0], inflection[1], 'go', markersize=8)
                plt.plot(upper[0], upper[1], 'ro', markersize=8)
                plt.plot(lower[0], lower[1], 'ro', markersize=8)

                plt.title(f"Omurga Eğrisi ve Kritik Noktalar")
            else:
                plt.title("Omurga Eğrisi Tespit Edilemedi")

            plt.subplot(2, 2, 4)
            plt.imshow(processed['enhanced'], cmap='gray')

            if cobb_result['success']:
                # Cobb açısını gösteren çizgileri çiz
                upper = cobb_result['upper_end_vertebra']
                lower = cobb_result['lower_end_vertebra']

                upper_slope = cobb_result['upper_slope']
                lower_slope = cobb_result['lower_slope']

                # Çizgileri göster
                y_len = 50  # Çizgi uzunluğu

                # Üst çizgi
                if np.isinf(upper_slope):
                    x1, y1 = upper[0], upper[1] - y_len
                    x2, y2 = upper[0], upper[1] + y_len
                else:
                    x1 = upper[0] - y_len * upper_slope
                    y1 = upper[1] - y_len
                    x2 = upper[0] + y_len * upper_slope
                    y2 = upper[1] + y_len

                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

                # Alt çizgi
                if np.isinf(lower_slope):
                    x1, y1 = lower[0], lower[1] - y_len
                    x2, y2 = lower[0], lower[1] + y_len
                else:
                    x1 = lower[0] - y_len * lower_slope
                    y1 = lower[1] - y_len
                    x2 = lower[0] + y_len * lower_slope
                    y2 = lower[1] + y_len

                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

                plt.title(f"Cobb Açısı: {cobb_result['angle']:.1f}°")
            else:
                plt.title("Cobb Açısı Hesaplanamadı")

            plt.tight_layout()
            plt.show()
    else:
        logger.warning("Görüntü bulunamadı!")