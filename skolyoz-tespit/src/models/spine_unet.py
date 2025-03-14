# src/models/spine_unet.py

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import cv2

from .improved_unet import ImprovedUNet

logger = logging.getLogger(__name__)


class SpineSegmentationUNet:
    def __init__(self, model_path=None, device=None, resize_dim=512):
        # Cihazı ayarla
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        logger.info(f"Model cihazı: {self.device}")

        # Görüntü boyutlandırma parametresi
        self.resize_dim = resize_dim

        # UNet modelini oluştur
        self.model = ImprovedUNet(n_channels=1, n_classes=1, bilinear=True)
        self.model.to(self.device)

        # Model ağırlıklarını yükle
        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
        else:
            logger.warning("Önceden eğitilmiş model bulunamadı, varsayılan ağırlıklar kullanılacak.")

    def _load_weights(self, model_path):
        """Model ağırlıklarını yükler."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Farklı checkpoint formatlarıyla başa çıkma
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            logger.info(f"Model ağırlıkları yüklendi: {model_path}")

        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")

    def preprocess_image(self, image):
        """Görüntüyü ön işlemeden geçirir."""
        # Görüntü tipini kontrol et
        if isinstance(image, np.ndarray):
            # Normalize et
            if image.dtype != np.float32:
                image = image.astype(np.float32)

            if image.max() > 1.0:
                image = image / 255.0

            # Kanalları kontrol et
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)

            # Görüntüyü yeniden boyutlandır
            resized_img = cv2.resize(image, (self.resize_dim, self.resize_dim))

            # Tensor formatına dönüştür: (H,W,C) -> (1,C,H,W)
            image_tensor = torch.from_numpy(resized_img.transpose(2, 0, 1)).unsqueeze(0)

            return image_tensor.to(self.device)

        else:
            raise ValueError("Geçersiz görüntü formatı. NumPy array bekleniyor.")

    def postprocess_mask(self, mask_tensor, original_shape):
        """Tahmin edilen maskeyi orijinal boyutlara döndürür."""
        # Tensörü numpy array'e dönüştür
        mask = mask_tensor.cpu().numpy().squeeze()

        # Orijinal boyuta yeniden boyutlandır
        if original_shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

        return mask

    def segment_spine(self, image):
        """Omurga segmentasyonu yapar."""
        # Orijinal boyutları sakla
        original_shape = image.shape

        # Ön işleme
        image_tensor = self.preprocess_image(image)

        # Modeli değerlendirme moduna al
        self.model.eval()

        # Segmentasyon
        with torch.no_grad():
            try:
                mask_tensor = self.model(image_tensor)

                # Son işleme
                mask = self.postprocess_mask(mask_tensor, original_shape)

                return mask

            except Exception as e:
                logger.error(f"Segmentasyon hatası: {str(e)}")
                # Hata durumunda boş maske döndür
                return np.zeros(original_shape[:2], dtype=np.float32)

    def get_binary_mask(self, probability_mask, threshold=0.5):
        """Olasılık maskesinden ikili (binary) maske oluşturur."""
        return (probability_mask > threshold).astype(np.float32)