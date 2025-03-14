# tests/test_preprocessing.py

import unittest
import numpy as np
import os
import sys

# Projenin kök dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import XRayPreprocessor


class TestPreprocessing(unittest.TestCase):
    """
    Ön işleme fonksiyonlarını test eden birim testleri.
    """

    def setUp(self):
        """
        Test için gerekli nesneleri oluşturur.
        """
        self.preprocessor = XRayPreprocessor()

        # Test görüntüsü oluştur
        self.test_image = np.zeros((100, 100), dtype=np.float32)
        self.test_image[20:80, 30:70] = 1.0

    def test_enhance_contrast(self):
        """
        Kontrast artırma fonksiyonunu test eder.
        """
        enhanced = self.preprocessor.enhance_contrast(self.test_image)

        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertGreaterEqual(np.max(enhanced), np.max(self.test_image))

    def test_denoise(self):
        """
        Gürültü azaltma fonksiyonunu test eder.
        """
        # Gürültülü görüntü oluştur
        noisy_image = self.test_image.copy()
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, self.test_image.shape)
        noisy_image += noise
        noisy_image = np.clip(noisy_image, 0, 1)

        denoised = self.preprocessor.denoise(noisy_image)

        self.assertEqual(denoised.shape, noisy_image.shape)

        # Gürültü azalmalı
        self.assertLess(np.std(denoised), np.std(noisy_image))

    def test_extract_body_region(self):
        """
        Vücut bölgesi çıkarma fonksiyonunu test eder.
        """
        masked_image, body_mask = self.preprocessor.extract_body_region(self.test_image)

        self.assertEqual(masked_image.shape, self.test_image.shape)
        self.assertEqual(body_mask.shape, self.test_image.shape)

        # Maske görüntüden daha az beyaz alan içermeli
        self.assertLessEqual(np.sum(body_mask), np.sum(self.test_image > 0))

    def test_detect_spine_region(self):
        """
        Omurga bölgesi tespit fonksiyonunu test eder.
        """
        spine_region, spine_bbox = self.preprocessor.detect_spine_region(self.test_image)

        self.assertIsNotNone(spine_region)
        self.assertIsNotNone(spine_bbox)

        # Kırpılmış bölge daha küçük olmalı
        self.assertLess(spine_region.shape[0] * spine_region.shape[1],
                        self.test_image.shape[0] * self.test_image.shape[1])

    def test_apply_all(self):
        """
        Tüm ön işleme adımlarını test eder.
        """
        results = self.preprocessor.apply_all(self.test_image)

        # Tüm beklenen anahtarlar olmalı
        expected_keys = ['original', 'enhanced', 'denoised', 'body_image',
                         'body_mask', 'spine_region', 'spine_bbox']

        for key in expected_keys:
            self.assertIn(key, results)

        # Sonuçlar geçerli olmalı
        self.assertEqual(results['original'].shape, self.test_image.shape)
        self.assertEqual(results['enhanced'].shape, self.test_image.shape)
        self.assertEqual(results['denoised'].shape, self.test_image.shape)


if __name__ == "__main__":
    unittest.main()