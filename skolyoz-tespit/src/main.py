# src/main.py

import os
import argparse
import logging
import sys
import time
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from .spine.classic_detection import ClassicSpineDetector

import numpy as np
import cv2

from .config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, PRETRAINED_DIR,
    RESULTS_DIR, VISUALIZATION_DIR, SUPPORTED_FORMATS
)
from .data.loader import XRayLoader
from .data.preprocessing import XRayPreprocessor
from .models.unet import SpineSegmentationModel
from .spine.cobb_angle import CobbAngleCalculator
from .visualization.visualize import ResultVisualizer

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('skolyoz_detection.log')
    ]
)

logger = logging.getLogger(__name__)


class SkolyozDetector:
    """
    Skolyoz tespiti için ana uygulama sınıfı.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 threshold: float = 0.5,
                 input_size: tuple = (512, 512)):
        """
        Skolyoz dedektörünü başlatır.

        Args:
            model_path: Segmentasyon modeli dosya yolu
            threshold: Segmentasyon eşik değeri
            input_size: Giriş görüntüsü boyutu
        """
        logger.info("Skolyoz Dedektörü başlatılıyor...")

        self.threshold = threshold
        self.input_size = input_size

        # Bileşenleri oluştur
        self.loader = XRayLoader(target_size=input_size)
        self.preprocessor = XRayPreprocessor()

        # Uygun bir model dosyası bulunmazsa, model yolunu None olarak bırak
        if model_path is None:
            # Önceden eğitilmiş model dosyaları ara
            model_files = glob.glob(os.path.join(PRETRAINED_DIR, "spine_*.pth"))
            model_files += glob.glob(os.path.join(PRETRAINED_DIR, "spine_*.h5"))

            if model_files:
                model_path = model_files[0]
                logger.info(f"Bulunan önceden eğitilmiş model: {model_path}")

        self.model = SpineSegmentationModel(model_path)
        self.cobb_calculator = CobbAngleCalculator()
        self.visualizer = ResultVisualizer()

        logger.info("Skolyoz Dedektörü başlatıldı.")

    def process_image(self, image_path: str,
                      output_dir: Optional[str] = None,
                      save_results: bool = True,
                      show_results: bool = False) -> Dict[str, Any]:
        """
        Bir X-ray görüntüsünü işleyerek skolyoz tespiti yapar.

        Args:
            image_path: X-ray görüntüsünün dosya yolu
            output_dir: Sonuçların kaydedileceği dizin
            save_results: Sonuçları kaydet
            show_results: Sonuçları ekranda göster

        Returns:
            Sonuçları içeren sözlük
        """
        logger.info(f"Görüntü işleniyor: {image_path}")
        start_time = time.time()

        # Çıktı dizinini ayarla
        if output_dir is None:
            output_dir = RESULTS_DIR

        os.makedirs(output_dir, exist_ok=True)

        # Görüntüyü yükle
        logger.info("Görüntü yükleniyor...")
        image = self.loader.load_image(image_path)

        if image is None:
            logger.error("Görüntü yüklenemedi!")
            return {'success': False, 'error': 'Görüntü yüklenemedi'}

        # Ön işleme
        logger.info("Görüntü ön işleniyor...")
        processed = self.preprocessor.apply_all(image)

        try:
            # ---------- DEĞİŞİKLİK BAŞLANGICI ----------
            # Klasik tespit metodu kullan (U-Net yerine)
            classic_detector = ClassicSpineDetector()

            # Omurga tespiti yap
            logger.info("Omurga tespiti yapılıyor...")
            detection_result = classic_detector.detect_spine(processed['enhanced'])

            # Eğer tespit başarısızsa U-Net'i dene
            if not detection_result['success']:
                try:
                    # Omurga segmentasyonu
                    logger.info("Omurga segmentasyonu yapılıyor...")
                    spine_mask = self.model.segment_spine(processed['enhanced'])
                    binary_mask = self.model.get_binary_mask(spine_mask)

                    # Cobb açısı hesaplama
                    logger.info("Cobb açısı hesaplanıyor...")
                    cobb_result = self.cobb_calculator.calculate_angle(binary_mask)
                except Exception as e:
                    logger.warning(f"U-Net metodu başarısız: {str(e)}")
                    # U-Net hata verirse boş sonuç döndür
                    spine_mask = np.zeros_like(processed['enhanced'])
                    binary_mask = np.zeros_like(processed['enhanced'])
                    cobb_result = {
                        'success': False,
                        'angle': float('nan'),
                        'curve_points': []
                    }
            else:
                # Klasik tespit başarılıysa sonuçları kullan
                spine_mask = detection_result['spine_mask'] / 255.0 if detection_result['spine_mask'].max() > 1 else \
                detection_result['spine_mask']
                binary_mask = (spine_mask > 0.5).astype(np.float32)
                cobb_result = {
                    'success': detection_result['success'],
                    'angle': detection_result['cobb_angle'],
                    'curve_points': detection_result['curve_points'],
                    'inflection_point': detection_result.get('upper_inflection'),
                    'upper_end_vertebra': detection_result.get('upper_inflection'),
                    'lower_end_vertebra': detection_result.get('lower_inflection')
                }
            # ---------- DEĞİŞİKLİK SONU ----------

            # Sonuçları görselleştir
            basename = os.path.basename(image_path)
            name, _ = os.path.splitext(basename)

            result_image = None
            result_path = None
            report_path = None

            if save_results or show_results:
                logger.info("Sonuçlar görselleştiriliyor...")

                result_path = os.path.join(output_dir, f"{name}_result.png") if save_results else None

                result_image = self.visualizer.visualize_results(
                    processed['enhanced'],
                    binary_mask,
                    cobb_result,
                    output_path=result_path,
                    show_plot=show_results
                )

                if save_results:
                    report_path = self.visualizer.save_report(
                        image_path,
                        processed['enhanced'],
                        binary_mask,
                        cobb_result,
                        output_filename=f"{name}_report.png"
                    )

            # İşlem süresini hesapla
            processing_time = time.time() - start_time
            logger.info(f"İşlem tamamlandı. Süre: {processing_time:.2f} saniye")

            # Sonuçları hazırla
            result = {
                'success': True,
                'image_path': image_path,
                'result_path': result_path,
                'report_path': report_path,
                'processing_time': processing_time,
                'cobb_angle': cobb_result.get('angle', float('nan')),
                'cobb_success': cobb_result.get('success', False),
                'result_image': result_image,
                'spine_mask': binary_mask,
                'curve_points': cobb_result.get('curve_points', []),
                'inflection_point': cobb_result.get('inflection_point', None),
                'upper_end_vertebra': cobb_result.get('upper_end_vertebra', None),
                'lower_end_vertebra': cobb_result.get('lower_end_vertebra', None)
            }

            return result

        except Exception as e:
            # Hata durumunda
            logger.error(f"Görüntü işlenirken hata: {str(e)}")
            processing_time = time.time() - start_time

            return {
                'success': False,
                'image_path': image_path,
                'result_path': None,
                'report_path': None,
                'processing_time': processing_time,
                'error': str(e)
            }

    def process_directory(self, input_dir: str,
                          output_dir: Optional[str] = None,
                          save_results: bool = True,
                          show_results: bool = False,
                          compare_results: bool = True) -> List[Dict[str, Any]]:
        """
        Bir dizindeki tüm X-ray görüntülerini işler.

        Args:
            input_dir: Girdi dizini
            output_dir: Çıktı dizini
            save_results: Sonuçları kaydet
            show_results: Sonuçları ekranda göster
            compare_results: Sonuçları karşılaştırma ızgarasında göster

        Returns:
            Tüm sonuçların listesi
        """
        logger.info(f"Dizin işleniyor: {input_dir}")

        # Çıktı dizinini ayarla
        if output_dir is None:
            output_dir = RESULTS_DIR

        os.makedirs(output_dir, exist_ok=True)

        # Tüm görüntü dosyalarını bul
        image_files = []
        for ext in SUPPORTED_FORMATS:
            image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

        if not image_files:
            logger.warning(f"Dizinde ({input_dir}) desteklenen görüntü dosyası bulunamadı!")
            return []

        logger.info(f"Toplam {len(image_files)} görüntü işlenecek")

        # Tüm görüntüleri işle
        results = []

        for image_path in image_files:
            result = self.process_image(
                image_path,
                output_dir=output_dir,
                save_results=save_results,
                show_results=show_results
            )

            results.append(result)

            # İlerleme göster
            logger.info(f"İşlenen: {len(results)}/{len(image_files)}")

        # Sonuçları karşılaştır
        if compare_results and results:
            logger.info("Sonuçlar karşılaştırılıyor...")

            comparison_images = []

            for result in results:
                if result.get('success', False) and result.get('result_image') is not None:
                    comparison_images.append({
                        'image': result['result_image'],
                        'title': os.path.basename(result['image_path']),
                        'cobb_angle': result.get('cobb_angle', float('nan'))
                    })

            if comparison_images:
                self.visualizer.create_comparison_grid(
                    comparison_images,
                    output_path=os.path.join(output_dir, "comparison.png")
                )

        # Özet göster
        successful = sum(1 for r in results if r.get('success', False))
        cobb_detected = sum(1 for r in results if r.get('cobb_success', False))

        logger.info(f"Tüm işlemler tamamlandı:")
        logger.info(f"  Toplam görüntü: {len(image_files)}")
        logger.info(f"  Başarıyla işlenen: {successful}")
        logger.info(f"  Cobb açısı tespit edilen: {cobb_detected}")

        if cobb_detected > 0:
            avg_angle = np.mean([r['cobb_angle'] for r in results if
                                 r.get('cobb_success', False) and not np.isnan(r.get('cobb_angle', float('nan')))])
            logger.info(f"  Ortalama Cobb açısı: {avg_angle:.2f}°")

        return results

    def interactive_process(self, image_path: str) -> Dict[str, Any]:
        """
        Bir görüntüyü interaktif olarak işler ve sonuçları gösterir.

        Args:
            image_path: Görüntü dosyasının yolu

        Returns:
            İşlem sonuçları
        """
        import matplotlib.pyplot as plt

        # Görüntüyü işle
        result = self.process_image(
            image_path,
            save_results=False,
            show_results=False
        )

        if not result.get('success', False):
            logger.error("Görüntü işlenirken hata oluştu!")
            return result

        # Sonuçları görselleştir
        plt.figure(figsize=(12, 10))

        # Orijinal görüntü
        plt.subplot(2, 2, 1)
        image = self.loader.load_image(image_path)
        plt.imshow(image, cmap='gray')
        plt.title("Orijinal X-ray")
        plt.axis('off')

        # İşlenmiş görüntü
        plt.subplot(2, 2, 2)
        processed = self.preprocessor.apply_all(image)
        plt.imshow(processed['enhanced'], cmap='gray')
        plt.title("İşlenmiş Görüntü")
        plt.axis('off')

        # Omurga segmentasyonu
        plt.subplot(2, 2, 3)
        plt.imshow(result['spine_mask'], cmap='gray')
        plt.title("Omurga Segmentasyonu")
        plt.axis('off')

        # Sonuç
        plt.subplot(2, 2, 4)
        plt.imshow(result['result_image'])
        if result.get('cobb_success', False):
            plt.title(f"Skolyoz Tespiti - Cobb Açısı: {result['cobb_angle']:.1f}°")
        else:
            plt.title("Skolyoz Tespiti - Cobb Açısı Hesaplanamadı")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return result


def main():
    """
    Ana uygulama fonksiyonu.
    """
    parser = argparse.ArgumentParser(description="Skolyoz Tespiti Uygulaması")

    # Temel parametreler
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Girdi görüntüsü veya dizini")
    parser.add_argument("--output", "-o", type=str, default=RESULTS_DIR,
                        help=f"Çıktı dizini (varsayılan: {RESULTS_DIR})")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Omurga segmentasyonu model dosyası yolu")

    # İşlem parametreleri
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Segmentasyon eşik değeri (varsayılan: 0.5)")
    parser.add_argument("--size", "-s", type=int, default=512,
                        help="Görüntü işleme boyutu (varsayılan: 512)")

    # Çıktı seçenekleri
    parser.add_argument("--save", action="store_true", default=True,
                        help="Sonuçları kaydet (varsayılan: True)")
    parser.add_argument("--no-save", action="store_false", dest="save",
                        help="Sonuçları kaydetme")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Sonuçları göster (varsayılan: False)")
    parser.add_argument("--compare", action="store_true", default=True,
                        help="Dizin işlenirken sonuçları karşılaştır (varsayılan: True)")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="İnteraktif mod")

    # Diğer seçenekler
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Detaylı çıktı göster")

    args = parser.parse_args()

    # Detaylı çıktı modunu ayarla
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Skolyoz dedektörünü oluştur
    detector = SkolyozDetector(
        model_path=args.model,
        threshold=args.threshold,
        input_size=(args.size, args.size)
    )

    # Girdi türünü belirle (dosya veya dizin)
    input_path = os.path.abspath(args.input)

    if os.path.isfile(input_path):
        # Tek bir görüntüyü işle
        if args.interactive:
            detector.interactive_process(input_path)
        else:
            detector.process_image(
                input_path,
                output_dir=args.output,
                save_results=args.save,
                show_results=args.show
            )
    elif os.path.isdir(input_path):
        # Tüm dizini işle
        detector.process_directory(
            input_path,
            output_dir=args.output,
            save_results=args.save,
            show_results=args.show,
            compare_results=args.compare
        )
    else:
        logger.error(f"Geçersiz girdi: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()