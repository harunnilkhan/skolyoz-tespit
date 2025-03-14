# src/models/unet.py

import torch
import torch.nn as nn
import torchvision
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    U-Net için çift konvolüsyon bloğu.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """
    U-Net için aşağı blok (konvolüsyon + max pooling).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    """
    U-Net için yukarı blok (yukarı örnekleme + konvolüsyon).
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(UpBlock, self).__init__()

        # Yukarı örnekleme yöntemi
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Bilinear yükseltme kullanıyorsak, kanal sayısını kendimiz azaltmalıyız
            self.channel_reducer = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)

        # Birleştirmeden sonraki kanal sayısı: skip_channels + in_channels // 2
        self.conv = ConvBlock(out_channels + in_channels // 2, out_channels)

    def forward(self, x, skip):
        # Yukarı örnekleme
        if hasattr(self, 'channel_reducer'):
            x = self.channel_reducer(x)
        x = self.up(x)

        # Skip bağlantısı ile boyutları eşleştir
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        # Padding ekle
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])

        # Skip bağlantı ile birleştir
        x = torch.cat([skip, x], dim=1)

        # Konvolüsyon uygula
        return self.conv(x)


class SimplifiedUNet(nn.Module):
    """
    Daha güçlü ve daha basit bir U-Net modeli.
    Röntgen görüntülerinden omurga segmentasyonu için özelleştirilmiştir.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, bilinear: bool = True):
        super(SimplifiedUNet, self).__init__()

        # Sabit özellik sayıları
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Daha basit bir mimari yapı
        # Derinlik sınırlandırılmış ve özellik sayıları azaltılmış
        features = [32, 64, 128, 256]

        # Encoder
        self.input_conv = ConvBlock(in_channels, features[0])
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])

        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        # Decoder
        self.up1 = UpBlock(features[3] * 2, features[2], bilinear)
        self.up2 = UpBlock(features[2], features[1], bilinear)
        self.up3 = UpBlock(features[1], features[0], bilinear)

        # Çıkış katmanı
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Sigmoid aktivasyonu
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.input_conv(x)
        x2, skip1 = self.down1(x1)
        x3, skip2 = self.down2(x2)
        x4, skip3 = self.down3(x3)

        # Bottleneck
        x5 = self.bottleneck(x4)

        # Decoder - Skip bağlantılar ile
        x = self.up1(x5, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)

        # Çıkış
        logits = self.output_conv(x)
        return self.sigmoid(logits)


class SpineSegmentationModel:
    """
    Omurga segmentasyonu için model sınıfı.
    """

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Omurga segmentasyon modelini başlatır.

        Args:
            model_path: Önceden eğitilmiş model yolu
            device: İşlem yapılacak cihaz (GPU veya CPU)
        """
        self.device = device
        logger.info(f"Model cihazı: {self.device}")

        # Daha basit model oluştur
        self.model = SimplifiedUNet(in_channels=1, out_channels=1)
        self.model.to(self.device)

        # Önceden eğitilmiş model varsa yükle
        if model_path is not None:
            self._load_model(model_path)
        else:
            logger.warning("Önceden eğitilmiş model yüklenmedi.")

    def _load_model(self, model_path: str) -> None:
        """
        Önceden eğitilmiş modeli yükler.

        Args:
            model_path: Model dosyasının yolu
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Checkpoint içeriğini kontrol et
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            logger.info(f"Model başarıyla yüklendi: {model_path}")

        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")
            logger.info("Model rastgele ağırlıklarla başlatıldı.")

    def _preprocess_tensor(self, tensor):
        """
        Tensörü boyut kontrolü yaparak işler
        """
        # Maksimum boyut sınırlaması (1024x1024'den büyük olmasın)
        max_size = 1024

        # Görüntü çok büyükse yeniden boyutlandır
        if tensor.shape[2] > max_size or tensor.shape[3] > max_size:
            # Orijinal yükseklik ve genişlik oranını koru
            h, w = tensor.shape[2], tensor.shape[3]
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            # Tensörü yeniden boyutlandır
            tensor = nn.functional.interpolate(
                tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
            )

            logger.info(f"Büyük görüntü yeniden boyutlandırıldı: {h}x{w} -> {new_h}x{new_w}")

        return tensor

    def segment_spine(self, image: np.ndarray) -> np.ndarray:
        """
        Verilen görüntüde omurgayı segmente eder.

        Args:
            image: Görüntü (1 kanallı, normalize edilmiş)

        Returns:
            Omurga segmentasyon maskesi
        """
        # Model değerlendirme moduna al
        self.model.eval()

        # Görüntüyü tensor'a çevir
        if len(image.shape) == 2:
            # (H, W) -> (1, 1, H, W)
            tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            # (H, W, 1) -> (1, 1, H, W)
            tensor = torch.from_numpy(image.squeeze()).float().unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Beklenmeyen görüntü şekli: {image.shape}")

        tensor = tensor.to(self.device)

        # Büyük görüntüler için ön işleme
        tensor = self._preprocess_tensor(tensor)

        # Gradyan hesaplamayı devre dışı bırak
        with torch.no_grad():
            try:
                # Model çıktısını al
                output = self.model(tensor)

                # CPU'ya geri taşı ve NumPy'a çevir
                mask = output.cpu().squeeze().numpy()

                # Görüntü yeniden boyutlandırıldıysa, orijinal boyuta geri getir
                if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)

                return mask

            except Exception as e:
                logger.error(f"Segmentasyon işlemi sırasında hata: {str(e)}")
                # Hata durumunda boş bir maske döndür
                return np.zeros(image.shape[:2], dtype=np.float32)

    def get_binary_mask(self, probability_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Olasılık maskesini ikili maskeye çevirir.

        Args:
            probability_mask: Olasılık maskesi (0-1 aralığında)
            threshold: Eşik değeri

        Returns:
            İkili maske
        """
        return (probability_mask > threshold).astype(np.float32)


# Test için
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    import os

    # Logging konfigürasyonu
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Test görüntüsü yükle
    test_image_path = "../../data/raw/sample.jpg"  # Bu yolu kendi test görüntünüze göre değiştirin

    if os.path.exists(test_image_path):
        # Görüntüyü yükle
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0  # 0-1 aralığına normalize et

        # Segmentasyon modeli oluştur
        model = SpineSegmentationModel()

        # Omurga segmentasyonu yap
        spine_mask = model.segment_spine(image)
        binary_mask = model.get_binary_mask(spine_mask)

        # Sonuçları göster
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Orijinal X-ray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(spine_mask, cmap='jet')
        plt.title('Omurga Olasılık Maskesi')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Omurga İkili Maskesi')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        logger.warning(f"Test görüntüsü bulunamadı: {test_image_path}")