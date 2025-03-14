# Skolyoz Tespiti Projesi

Bu proje, X-ray görüntülerinden otomatik olarak omurga segmentasyonu yaparak skolyoz tespiti ve Cobb açısı hesaplaması yapar. MHA formatı yerine standart görüntü formatlarını (JPG, PNG vb.) kullanır ve önceden hazırlanmış maskelere ihtiyaç duymadan çalışır.

## Özellikler

- X-ray görüntülerini otomatik işleme ve ön hazırlık
- U-Net tabanlı derin öğrenme ile omurga segmentasyonu
- Omurga eğrisini otomatik tespit etme
- Cobb açısı otomatik hesaplama
- Görsel sonuçları kaydetme ve görselleştirme
- Toplu görüntü işleme ve karşılaştırma

## Kurulum

### Gereksinimler

- Python 3.7+
- PyTorch 1.10+
- OpenCV 4.5+
- NumPy, Matplotlib, SciPy ve diğer bağımlılıklar

### Paket Kurulumu

Projeyi klonlayın ve gerekli paketleri yükleyin:

```bash
git clone https://github.com/username/skolyoz-tespit.git
cd skolyoz-tespit
pip install -r requirements.txt
```

Veya doğrudan kurulum yapın:

```bash
pip install -e .
```

## Kullanım

### Komut Satırından Kullanım

Tek bir X-ray görüntüsünü işlemek için:

```bash
python -m src.main --input path/to/xray.jpg --output results/ --show
```

Bir dizindeki tüm X-ray görüntülerini işlemek için:

```bash
python -m src.main --input path/to/directory/ --output results/
```

İnteraktif mod (sonuçları canlı gösterir):

```bash
python -m src.main --input path/to/xray.jpg --interactive
```

### Program İçinden Kullanım

```python
from src.main import SkolyozDetector

# Dedektörü başlat
detector = SkolyozDetector()

# Tek bir görüntüyü işle
result = detector.process_image("path/to/xray.jpg")

# Cobb açısını al
cobb_angle = result['cobb_angle']
print(f"Cobb Açısı: {cobb_angle:.1f}°")
```

## Komut Satırı Parametreleri

```
--input, -i       : Girdi görüntüsü veya dizini
--output, -o      : Çıktı dizini (varsayılan: results/)
--model, -m       : Segmentasyon modeli dosya yolu
--threshold, -t   : Segmentasyon eşik değeri (varsayılan: 0.5)
--size, -s        : Görüntü işleme boyutu (varsayılan: 512)
--save            : Sonuçları kaydet (varsayılan: True)
--no-save         : Sonuçları kaydetme
--show            : Sonuçları göster (varsayılan: False)
--compare         : Dizin işlenirken sonuçları karşılaştır (varsayılan: True)
--interactive     : İnteraktif mod
--verbose, -v     : Detaylı çıktı göster
```

## Proje Yapısı

```
skolyoz-tespit/
│
├── data/
│   ├── raw/             # Ham röntgen görüntüleri
│   ├── processed/       # İşlenmiş görüntüler
│   └── pretrained/      # Önceden eğitilmiş modeller
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Veri yükleme işlemleri
│   │   └── preprocessing.py     # Görüntü ön işleme
│   │
│   ├── models/
│   │   ├── unet.py              # U-Net model tanımı
│   │   └── backbone.py          # Omurga modelleri
│   │
│   ├── spine/
│   │   ├── segmentation.py      # Omurga segmentasyonu
│   │   ├── cobb_angle.py        # Cobb açısı hesaplama
│   │   └── curve_detection.py   # Omurga eğrisi tespiti
│   │
│   ├── visualization/
│   │   └── visualize.py         # Sonuç görselleştirme
│   │
│   ├── utils/
│   │   └── helpers.py           # Yardımcı işlevler
│   │
│   ├── config.py                # Konfigürasyon ayarları
│   └── main.py                  # Ana program
│
├── notebooks/
│   └── demo.ipynb               # Örnek kullanım
│
├── requirements.txt             # Gerekli paketler
├── setup.py                     # Kurulum dosyası
└── README.md                    # Proje açıklaması
```

## Skolyoz ve Cobb Açısı Hakkında

Skolyoz, omurganın yana doğru anormal eğriliğidir. Cobb açısı, röntgen görüntülerinde skolyozun şiddetini ölçmek için kullanılan standart bir yöntemdir. Bu projede, otomatik olarak:

1. X-ray görüntüsünden omurga segmente edilir
2. Omurga eğrisi tespit edilir
3. En eğri bölgeler tespit edilir
4. Cobb açısı hesaplanır

Cobb açısı değerleri şu şekilde yorumlanabilir:
- 0-10°: Normal veya minimal eğrilik
- 10-25°: Hafif skolyoz
- 25-40°: Orta şiddetli skolyoz
- 40°+: Şiddetli skolyoz

## Lisans

Bu proje MIT lisansı altında dağıtılmaktadır. Daha fazla bilgi için `LICENSE` dosyasına bakınız.

## İletişim

Sorularınız veya geri bildirimleriniz için lütfen bizimle iletişime geçin.