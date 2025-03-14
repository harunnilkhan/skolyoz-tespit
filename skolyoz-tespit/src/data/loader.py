# src/data/loader.py

import os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging

from ..config import SUPPORTED_FORMATS, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class XRayLoader:
    """
    X-ray görüntülerini yüklemek ve ön işlemek için sınıf.
    """

    def __init__(self,
                 data_dir: str = RAW_DATA_DIR,
                 output_dir: str = PROCESSED_DATA_DIR,
                 target_size: Tuple[int, int] = (512, 512)):
        """
        X-ray yükleyici sınıfını başlatır.

        Args:
            data_dir: X-ray görüntülerinin bulunduğu dizin
            output_dir: İşlenmiş görüntülerin kaydedileceği dizin
            target_size: Görüntülerin boyutlandırılacağı hedef boyut
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_size = target_size

        # Dizinleri oluştur
        os.makedirs(self.output_dir, exist_ok=True)

        # İstatistikler
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'skipped_images': 0,
            'error_images': 0,
        }

    def find_xray_files(self) -> List[str]:
        """
        Veri dizininde desteklenen formatlardaki tüm X-ray dosyalarını bulur.

        Returns:
            Dosya yollarının listesi
        """
        image_files = []

        for ext in SUPPORTED_FORMATS:
            # os.walk kullanarak alt dizinleri de tara
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.lower().endswith(ext):
                        image_files.append(os.path.join(root, file))

        self.stats['total_images'] = len(image_files)
        logger.info(f"Toplam {len(image_files)} X-ray görüntüsü bulundu")

        return image_files

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Belirtilen yoldaki görüntüyü yükler.

        Args:
            image_path: Görüntü dosyasının yolu

        Returns:
            Yüklenen görüntü veya yükleme başarısız olduysa None
        """
        try:
            # Görüntüyü grayscale olarak yükle
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                logger.warning(f"Görüntü yüklenemedi: {image_path}")
                self.stats['error_images'] += 1
                return None

            return image

        except Exception as e:
            logger.error(f"Görüntü yüklenirken hata: {image_path}, {str(e)}")
            self.stats['error_images'] += 1
            return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Görüntüyü ön işlemeden geçirir: yeniden boyutlandırma, normalizasyon ve kontrast artırma.

        Args:
            image: Görüntü verileri

        Returns:
            İşlenmiş görüntü
        """
        # Görüntüyü hedef boyuta yeniden boyutlandır
        image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Histogram eşitleme ile kontrast artırma
        image_eq = cv2.equalizeHist(image_resized)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) uygula
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image_resized)

        # Normalize et (0-1 aralığına)
        image_norm = image_clahe.astype(np.float32) / 255.0

        return image_norm

    def process_images(self, max_images: Optional[int] = None, save_processed: bool = True) -> Dict[str, np.ndarray]:
        """
        Veri setindeki tüm görüntüleri işler.

        Args:
            max_images: İşlenecek maksimum görüntü sayısı (None ise tümünü işle)
            save_processed: İşlenmiş görüntüleri kaydet

        Returns:
            Görüntü adı -> işlenmiş görüntü eşlemesi
        """
        # Görüntü dosyalarını bul
        image_files = self.find_xray_files()

        # Eğer maksimum sayı belirtildiyse, o kadar görüntü al
        if max_images is not None:
            image_files = image_files[:max_images]

        # İşlenmiş görüntüleri sakla
        processed_images = {}

        for image_path in image_files:
            try:
                # Görüntü adını al
                image_name = os.path.basename(image_path)
                image_name_no_ext = os.path.splitext(image_name)[0]

                # Görüntüyü yükle
                image = self.load_image(image_path)

                if image is None:
                    continue

                # Görüntüyü ön işlemeden geçir
                processed_image = self.preprocess_image(image)

                # İşlenmiş görüntüyü kaydet
                if save_processed:
                    output_path = os.path.join(self.output_dir, f"{image_name_no_ext}_processed.png")
                    cv2.imwrite(output_path, (processed_image * 255).astype(np.uint8))

                # Görüntüyü sözlüğe ekle
                processed_images[image_name] = processed_image

                self.stats['processed_images'] += 1

            except Exception as e:
                logger.error(f"Görüntü işlenirken hata: {image_path}, {str(e)}")
                self.stats['error_images'] += 1

        # İstatistikleri göster
        logger.info(f"İşleme tamamlandı: {self.stats['processed_images']} başarılı, "
                    f"{self.stats['error_images']} hatalı")

        return processed_images

    def load_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Tek bir X-ray görüntüsünü yükler ve işler.

        Args:
            image_path: Görüntü dosyasının yolu

        Returns:
            İşlenmiş görüntü veya işleme başarısız olduysa None
        """
        try:
            # Görüntüyü yükle
            image = self.load_image(image_path)

            if image is None:
                return None

            # Görüntüyü ön işlemeden geçir
            processed_image = self.preprocess_image(image)

            return processed_image

        except Exception as e:
            logger.error(f"Görüntü işlenirken hata: {image_path}, {str(e)}")
            return None


# Test için
if __name__ == "__main__":
    # Logging konfigürasyonu
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Loader oluştur
    loader = XRayLoader()

    # Görüntüleri bul
    image_files = loader.find_xray_files()

    if len(image_files) > 0:
        # İlk görüntüyü yükle ve işle
        image_path = image_files[0]
        logger.info(f"İlk görüntü işleniyor: {image_path}")

        processed_image = loader.load_single_image(image_path)

        if processed_image is not None:
            # Görüntüyü göster
            cv2.imshow("Processed X-Ray", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        logger.warning(f"'{loader.data_dir}' dizininde görüntü bulunamadı!")