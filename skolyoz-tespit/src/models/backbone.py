# src/models/backbone.py

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional


class ResNetBackbone(nn.Module):
    """
    ResNet tabanlı omurga ağı.
    """

    def __init__(self, model_name: str = "resnet34", pretrained: bool = True):
        """
        ResNet omurga ağını başlatır.

        Args:
            model_name: ResNet modeli adı ('resnet18', 'resnet34', 'resnet50', vb.)
            pretrained: ImageNet ile önceden eğitilmiş ağırlıkları kullan
        """
        super(ResNetBackbone, self).__init__()

        # Modeli yükle
        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            self.model = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Desteklenmeyen model adı: {model_name}")

        # Giriş katmanını 1 kanallı olarak değiştir (grayscale için)
        original_layer = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            # Ağırlıkları kopyala (grayscale için RGB kanallarının ortalamasını al)
            with torch.no_grad():
                self.model.conv1.weight = nn.Parameter(
                    original_layer.weight.mean(dim=1, keepdim=True)
                )

        # Sınıflandırma kafasını kaldır
        self.encoder = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x):
        """
        İleri geçiş.

        Args:
            x: Giriş görüntüsü

        Returns:
            Özellik haritaları
        """
        return self.encoder(x)


def get_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """
    İstenilen omurga modelini döndürür.

    Args:
        name: Omurga modeli adı
        pretrained: ImageNet ile önceden eğitilmiş ağırlıkları kullan

    Returns:
        Omurga modeli
    """
    if name.startswith('resnet'):
        return ResNetBackbone(model_name=name, pretrained=pretrained)
    else:
        raise ValueError(f"Desteklenmeyen omurga adı: {name}")