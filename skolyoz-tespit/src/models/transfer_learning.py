# src/models/transfer_learning.py

import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


def initialize_unet_with_backbone(unet_model, backbone_name='resnet34', pretrained=True):
    """
    UNet modelini önceden eğitilmiş bir omurga ağıyla başlatır.

    Args:
        unet_model: UNet modeli
        backbone_name: Omurga modeli adı ('resnet18', 'resnet34', 'resnet50')
        pretrained: ImageNet ile önceden eğitilmiş ağırlıkları kullan

    Returns:
        Transfer öğrenme ile başlatılmış UNet modeli
    """
    try:
        # Omurga modelini yükle
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        else:
            logger.warning(f"Desteklenmeyen omurga: {backbone_name}, varsayılan ResNet34 kullanılıyor")
            backbone = models.resnet34(pretrained=pretrained)

        # Encoder ağırlıklarını transfer et
        # Not: Bu basit bir örnek, gerçek uygulamada daha karmaşık olabilir

        # İlk katman (1 kanallı giriş için)
        unet_model.inc.double_conv[0].weight.data = backbone.conv1.weight.data.mean(dim=1, keepdim=True)

        # Diğer katmanlar için ağırlık transferi
        # (Bu kısım karmaşık olabilir ve modelin tam yapısına bağlıdır)

        logger.info(f"UNet modeli {backbone_name} omurgası ile başlatıldı")
        return unet_model

    except Exception as e:
        logger.error(f"Transfer öğrenme hatası: {str(e)}")
        return unet_model  # Orijinal modeli döndür