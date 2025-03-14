# src/visualization/visualize.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging
from ..config import VISUALIZATION_PARAMS, VISUALIZATION_DIR

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """
    Skolyoz tespiti sonuçlarını görselleştiren sınıf.
    """

    def __init__(self,
                 output_dir: str = VISUALIZATION_DIR,
                 line_thickness: int = VISUALIZATION_PARAMS['line_thickness'],
                 point_radius: int = VISUALIZATION_PARAMS['point_radius'],
                 spine_color: Tuple[int, int, int] = VISUALIZATION_PARAMS['spine_color'],
                 angle_color: Tuple[int, int, int] = VISUALIZATION_PARAMS['angle_color'],
                 text_color: Tuple[int, int, int] = VISUALIZATION_PARAMS['text_color'],
                 font_scale: float = VISUALIZATION_PARAMS['font_scale']):
        """
        Görselleştiriciyi başlatır.

        Args:
            output_dir: Çıktı dizini
            line_thickness: Çizgi kalınlığı
            point_radius: Nokta yarıçapı
            spine_color: Omurga rengi (BGR formatında)
            angle_color: Açı rengi (BGR formatında)
            text_color: Metin rengi (BGR formatında)
            font_scale: Font ölçeği
        """
        self.output_dir = output_dir
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.spine_color = spine_color
        self.angle_color = angle_color
        self.text_color = text_color
        self.font_scale = font_scale

        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_results(self, original_image: np.ndarray,
                          spine_mask: np.ndarray,
                          cobb_result: Dict[str, Any],
                          output_path: Optional[str] = None,
                          show_plot: bool = True) -> np.ndarray:
        """
        Skolyoz tespiti sonuçlarını görselleştirir.

        Args:
            original_image: Orijinal X-ray görüntüsü
            spine_mask: Omurga segmentasyon maskesi
            cobb_result: CobbAngleCalculator'dan elde edilen sonuç sözlüğü
            output_path: Sonuçların kaydedileceği dosya yolu
            show_plot: Grafiği göster/gösterme

        Returns:
            Görselleştirilmiş sonuç görüntüsü
        """
        # Görüntüyü RGB formatına çevir
        if len(original_image.shape) == 2:
            display_image = cv2.cvtColor(
                (original_image * 255).astype(np.uint8) if original_image.dtype == np.float32
                else original_image,
                cv2.COLOR_GRAY2BGR
            )
        else:
            display_image = original_image.copy()

        # Omurga maskesini RGB kanalına ekle
        if spine_mask is not None:
            mask_overlay = self._create_mask_overlay(original_image, spine_mask)
            # Maskeyi görüntüyle birleştir
            display_image = cv2.addWeighted(display_image, 0.7, mask_overlay, 0.3, 0)

        # Cobb açısı sonuçlarını çiz
        if cobb_result.get('success', False):
            # Omurga eğrisini çiz
            if 'curve_points' in cobb_result and len(cobb_result['curve_points']) > 1:
                curve_points = np.array(cobb_result['curve_points'])
                for i in range(len(curve_points) - 1):
                    pt1 = tuple(curve_points[i].astype(int))
                    pt2 = tuple(curve_points[i + 1].astype(int))
                    cv2.line(display_image, pt1, pt2, self.spine_color, self.line_thickness)

            # Kritik noktaları çiz
            for point_name in ['inflection_point', 'upper_end_vertebra', 'lower_end_vertebra']:
                if point_name in cobb_result and cobb_result[point_name] is not None:
                    point = np.array(cobb_result[point_name]).astype(int)
                    cv2.circle(display_image, tuple(point), self.point_radius,
                               (0, 0, 255) if point_name != 'inflection_point' else (0, 255, 0),
                               -1)

            # Cobb açısını gösteren çizgileri çiz
            self._draw_cobb_angle_lines(display_image, cobb_result)

            # Cobb açısı değerini yaz
            angle = cobb_result.get('angle', float('nan'))
            if not np.isnan(angle):
                text = f"Cobb Açısı: {angle:.1f}°"
                cv2.putText(display_image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale, self.text_color, 2)

        # Sonuçları kaydet
        if output_path is not None:
            cv2.imwrite(output_path, display_image)
            logger.info(f"Sonuç görüntüsü kaydedildi: {output_path}")

        # Görselleştirmeyi göster
        if show_plot:
            plt.figure(figsize=(12, 10))
            plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            plt.title("Skolyoz Tespiti Sonucu")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return display_image

    def _create_mask_overlay(self, original_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Maske için renkli katman oluşturur.

        Args:
            original_image: Orijinal görüntü
            mask: İkili maske

        Returns:
            Renklendirilmiş maske katmanı
        """
        # Görüntü boyutlarını al
        if len(original_image.shape) == 2:
            height, width = original_image.shape
        else:
            height, width = original_image.shape[:2]

        # Boş bir RGB görüntü oluştur
        overlay = np.zeros((height, width, 3), dtype=np.uint8)

        # Maskeyi normalize et
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            binary_mask = mask.copy()

        # Mask boyutlarını kontrol et ve gerekirse yeniden boyutlandır
        if binary_mask.shape[:2] != (height, width):
            binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Maskeyi yeşil kanala ekle
        overlay[binary_mask > 0] = (0, 255, 0)  # BGR formatında yeşil

        return overlay

    def _draw_cobb_angle_lines(self, image: np.ndarray, cobb_result: Dict[str, Any]) -> None:
        """
        Cobb açısını gösteren çizgileri çizer.

        Args:
            image: Çizim yapılacak görüntü
            cobb_result: Cobb açısı sonuçları
        """
        if not cobb_result.get('success', False):
            return

        upper = np.array(cobb_result.get('upper_end_vertebra')).astype(int)
        lower = np.array(cobb_result.get('lower_end_vertebra')).astype(int)

        upper_slope = cobb_result.get('upper_slope', 0)
        lower_slope = cobb_result.get('lower_slope', 0)

        # Çizgi uzunluğu
        y_len = 50

        # Üst çizgi
        if np.isinf(upper_slope):
            x1, y1 = upper[0], upper[1] - y_len
            x2, y2 = upper[0], upper[1] + y_len
        else:
            x1 = int(upper[0] - y_len * upper_slope)
            y1 = upper[1] - y_len
            x2 = int(upper[0] + y_len * upper_slope)
            y2 = upper[1] + y_len

        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                 self.angle_color, self.line_thickness)

        # Alt çizgi
        if np.isinf(lower_slope):
            x1, y1 = lower[0], lower[1] - y_len
            x2, y2 = lower[0], lower[1] + y_len
        else:
            x1 = int(lower[0] - y_len * lower_slope)
            y1 = lower[1] - y_len
            x2 = int(lower[0] + y_len * lower_slope)
            y2 = lower[1] + y_len

        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                 self.angle_color, self.line_thickness)

    def create_comparison_grid(self, images: List[Dict[str, np.ndarray]],
                               output_path: Optional[str] = None) -> np.ndarray:
        """
        Birden fazla görüntü için karşılaştırma ızgarası oluşturur.

        Args:
            images: Görüntü bilgilerini içeren sözlük listesi
                   [{'image': np.ndarray, 'title': str, 'cobb_angle': float}, ...]
            output_path: Izgaranın kaydedileceği dosya yolu

        Returns:
            Karşılaştırma ızgarası görüntüsü
        """
        if not images:
            return None

        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols

        plt.figure(figsize=(4 * cols, 4 * rows))

        for i, img_data in enumerate(images):
            plt.subplot(rows, cols, i + 1)

            # Görüntüyü göster
            if 'image' in img_data:
                if len(img_data['image'].shape) == 3:
                    plt.imshow(cv2.cvtColor(img_data['image'], cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(img_data['image'], cmap='gray')

            # Başlık ekle
            title = img_data.get('title', f"Görüntü {i + 1}")
            if 'cobb_angle' in img_data and not np.isnan(img_data['cobb_angle']):
                title += f" (Cobb: {img_data['cobb_angle']:.1f}°)"

            plt.title(title)
            plt.axis('off')

        plt.tight_layout()

        # Kaydet
        if output_path is not None:
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Karşılaştırma ızgarası kaydedildi: {output_path}")

        # Görüntüyü NumPy dizisine dönüştür
        plt.savefig('temp.png')
        grid = cv2.imread('temp.png')
        os.remove('temp.png')

        return grid

    def save_report(self, image_path: str, processed_image: np.ndarray,
                    spine_mask: np.ndarray, cobb_result: Dict[str, Any],
                    output_filename: Optional[str] = None) -> str:
        """
        Özet rapor oluşturur ve kaydeder.

        Args:
            image_path: Orijinal görüntü dosyasının yolu
            processed_image: İşlenmiş görüntü
            spine_mask: Omurga segmentasyon maskesi
            cobb_result: Cobb açısı sonuçları
            output_filename: Çıktı dosyasının adı

        Returns:
            Kaydedilen rapor dosyasının yolu
        """
        # Dosya adını oluştur
        if output_filename is None:
            basename = os.path.basename(image_path)
            name, _ = os.path.splitext(basename)
            output_filename = f"{name}_report.png"

        output_path = os.path.join(self.output_dir, output_filename)

        # Görüntü dizisini oluştur
        fig = plt.figure(figsize=(12, 8))

        # Orijinal görüntü
        plt.subplot(2, 2, 1)
        plt.imshow(processed_image, cmap='gray')
        plt.title("İşlenmiş Görüntü")
        plt.axis('off')

        # Omurga maskesi
        plt.subplot(2, 2, 2)
        plt.imshow(spine_mask, cmap='gray')
        plt.title("Omurga Segmentasyonu")
        plt.axis('off')

        # Omurga eğrisi
        plt.subplot(2, 2, 3)
        plt.imshow(processed_image, cmap='gray')

        if cobb_result.get('success', False) and 'curve_points' in cobb_result:
            curve_points = np.array(cobb_result['curve_points'])
            plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2)

            # Kritik noktaları göster
            if 'inflection_point' in cobb_result:
                inflection = np.array(cobb_result['inflection_point'])
                plt.plot(inflection[0], inflection[1], 'go', markersize=8)

            if 'upper_end_vertebra' in cobb_result:
                upper = np.array(cobb_result['upper_end_vertebra'])
                plt.plot(upper[0], upper[1], 'ro', markersize=8)

            if 'lower_end_vertebra' in cobb_result:
                lower = np.array(cobb_result['lower_end_vertebra'])
                plt.plot(lower[0], lower[1], 'ro', markersize=8)

        plt.title("Omurga Eğrisi")
        plt.axis('off')

        # Cobb açısı görselleştirme
        plt.subplot(2, 2, 4)

        if cobb_result.get('success', False):
            # Cobb açısı hesaplanabildi
            angle = cobb_result.get('angle', float('nan'))

            plt.imshow(processed_image, cmap='gray')

            # Açı çizgilerini çiz
            if 'upper_end_vertebra' in cobb_result and 'upper_slope' in cobb_result:
                upper = np.array(cobb_result['upper_end_vertebra'])
                upper_slope = cobb_result['upper_slope']

                y_len = 50
                if np.isinf(upper_slope):
                    x1, y1 = upper[0], upper[1] - y_len
                    x2, y2 = upper[0], upper[1] + y_len
                else:
                    x1 = upper[0] - y_len * upper_slope
                    y1 = upper[1] - y_len
                    x2 = upper[0] + y_len * upper_slope
                    y2 = upper[1] + y_len

                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

            if 'lower_end_vertebra' in cobb_result and 'lower_slope' in cobb_result:
                lower = np.array(cobb_result['lower_end_vertebra'])
                lower_slope = cobb_result['lower_slope']

                y_len = 50
                if np.isinf(lower_slope):
                    x1, y1 = lower[0], lower[1] - y_len
                    x2, y2 = lower[0], lower[1] + y_len
                else:
                    x1 = lower[0] - y_len * lower_slope
                    y1 = lower[1] - y_len
                    x2 = lower[0] + y_len * lower_slope
                    y2 = lower[1] + y_len

                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

            plt.title(f"Cobb Açısı: {angle:.1f}°")
        else:
            # Cobb açısı hesaplanamadı
            plt.text(0.5, 0.5, "Cobb açısı hesaplanamadı",
                     horizontalalignment='center',
                     verticalalignment='center')
            plt.title("Cobb Açısı")

        plt.axis('off')

        # Dosyaya kaydet
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Rapor kaydedildi: {output_path}")

        return output_path


# Test için
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..data.loader import XRayLoader
    from ..data.preprocessing import XRayPreprocessor
    from ..models.unet import SpineSegmentationModel
    from ..spine.cobb_angle import CobbAngleCalculator

    # Logging konfigürasyonu
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Gerekli nesneleri oluştur
    loader = XRayLoader()
    preprocessor = XRayPreprocessor()
    model = SpineSegmentationModel()
    cobb_calculator = CobbAngleCalculator()
    visualizer = ResultVisualizer()

    # Görüntüleri bul
    image_files = loader.find_xray_files()

    if len(image_files) > 0:
        # İlk birkaç görüntüyü işle
        comparison_images = []

        for i, image_path in enumerate(image_files[:min(3, len(image_files))]):
            logger.info(f"Görüntü işleniyor: {image_path}")

            # Görüntüyü yükle ve ön işle
            image = loader.load_image(image_path)

            if image is not None:
                # Ön işleme
                processed = preprocessor.apply_all(image)

                # Omurga segmentasyonu
                spine_mask = model.segment_spine(processed['enhanced'])
                binary_mask = model.get_binary_mask(spine_mask)

                # Cobb açısı hesaplama
                cobb_result = cobb_calculator.calculate_angle(binary_mask)

                # Sonuçları görselleştir
                basename = os.path.basename(image_path)
                result_image = visualizer.visualize_results(
                    processed['enhanced'],
                    binary_mask,
                    cobb_result,
                    output_path=os.path.join(visualizer.output_dir, f"{basename}_result.png"),
                    show_plot=False
                )

                # Rapor oluştur
                report_path = visualizer.save_report(
                    image_path,
                    processed['enhanced'],
                    binary_mask,
                    cobb_result
                )

                # Karşılaştırma için ekle
                comparison_images.append({
                    'image': result_image,
                    'title': f"#{i + 1}: {basename}",
                    'cobb_angle': cobb_result.get('angle', float('nan'))
                })

        # Karşılaştırma ızgarası oluştur
        if comparison_images:
            visualizer.create_comparison_grid(
                comparison_images,
                output_path=os.path.join(visualizer.output_dir, "comparison.png")
            )

            plt.show()
    else:
        logger.warning("Görüntü bulunamadı!")