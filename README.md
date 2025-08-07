# Derin Öğrenme ile Akciğer Kanseri Tespiti 🫁

Bu proje, Convolutional Neural Network (CNN) kullanarak göğüs röntgeni görüntülerinden pnömoni tespiti yapmaktadır. TensorFlow/Keras framework'ü kullanılarak geliştirilmiştir.

## 📋 Proje Hakkında

Bu sistem, tıbbi görüntü analizi alanında yapay zeka uygulaması olarak geliştirilmiştir. Göğüs röntgeni görüntülerini analiz ederek normal akciğer ile pnömoni olan akciğer arasında ayrım yapabilmektedir.

### 🎯 Proje Hedefleri
- Göğüs röntgeni görüntülerinden otomatik pnömoni tespiti
- Yüksek doğruluk oranı ile tıbbi tanı desteği
- Modüler ve yeniden kullanılabilir kod yapısı
- Görselleştirme ile model performansının takibi

## 🏗️ Proje Yapısı

```
LungCancerDetection/
├── src/
│   ├── data_loader.py      # Veri yükleme ve ön işleme
│   ├── model.py           # CNN model mimarisi
│   ├── train.py          # Ana eğitim scripti
│   └── visualize.py      # Sonuç görselleştirme
├── data/
│   └── chest_xray/       # Veri seti (ayrıca indirilmeli)
│       ├── train/
│       ├── test/
│       └── val/
├── requirements.txt       # Gerekli paketler
└── .gitignore            # Git yapılandırması
```

## 🔧 Kurulum

### 1. Gerekli Bağımlılıkları Yükleyin

```bash
# Sanal ortam oluşturun (önerilen)
python -m venv tfenv
tfenv\Scripts\activate  # Windows
# source tfenv/bin/activate  # Linux/Mac

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

### 2. Veri Setini İndirin

Bu proje [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) veri setini kullanmaktadır.

1. Kaggle'dan veri setini indirin
2. `data/chest_xray/` klasörüne çıkarın
3. Klasör yapısının şu şekilde olduğundan emin olun:
   ```
   data/chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── test/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── val/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

## 🚀 Kullanım

### Model Eğitimi

```bash
cd src
python train.py
```

Eğitim süreci şunları içerir:
- Veri yükleme ve ön işleme
- Model mimarisi oluşturma
- Veri artırma (Data Augmentation)
- Model eğitimi
- Performans değerlendirmesi
- Sonuçların görselleştirilmesi

### Eğitim Parametreleri

`train.py` dosyasında aşağıdaki parametreleri değiştirebilirsiniz:

```python
IMG_SIZE = 150        # Görüntü boyutu
EPOCHS = 3           # Eğitim döngü sayısı
BATCH_SIZE = 32      # Batch boyutu
```

## 🧠 Model Mimarisi

CNN modeli aşağıdaki katmanlardan oluşmaktadır:

1. **Konvolüsyon Katmanları:**
   - Conv2D (128 filtre, 7x7 kernel)
   - Conv2D (64 filtre, 5x5 kernel) 
   - Conv2D (32 filtre, 3x3 kernel)

2. **Düzenleme Katmanları:**
   - BatchNormalization
   - MaxPooling2D
   - Dropout (0.1, 0.2)

3. **Tam Bağlantılı Katmanlar:**
   - Dense (128 unit, ReLU)
   - Dense (1 unit, Sigmoid) - Binary classification

## 📊 Performans İyileştirmeleri

### Veri Artırma (Data Augmentation)
```python
- Rotasyon: ±30°
- Yakınlaştırma: %20
- Yatay/Dikey kaydırma: %10
- Yatay çevirme
```

### Öğrenme Oranı Planlaması
- ReduceLROnPlateau callback'i kullanılmaktadır
- Doğrulama accuracy'si gelişmediğinde öğrenme oranı azaltılır

## 📈 Sonuçlar ve Görselleştirme

Eğitim tamamlandıktan sonra:
- Model `pneumonia_classifier_model.h5` olarak kaydedilir
- Eğitim grafiği `training_history.png` olarak oluşturulur
- Konsola test accuracy ve loss değerleri yazdırılır

## 📦 Çıktılar

- **Eğitilmiş Model:** `pneumonia_classifier_model.h5`
- **Eğitim Grafiği:** `training_history.png`
- **Konsol Logları:** Eğitim süreci ve test sonuçları

## 🔍 Kod Modülleri

### `data_loader.py`
- Görüntü yükleme ve ön işleme
- Normalizasyon (0-1 arasına ölçekleme)
- Train/Test/Validation setlerini ayırma

### `model.py`
- CNN mimarisi tanımlama
- Model derleme ve yapılandırma

### `train.py`
- Ana eğitim döngüsü
- Veri artırma ayarları
- Callback'ler ve model kaydetme

### `visualize.py`
- Eğitim geçmişi görselleştirme
- Accuracy ve Loss grafikleri

## ⚙️ Sistem Gereksinimleri

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy, Matplotlib, Pandas
- En az 4GB RAM (8GB önerilen)
- GPU desteği için CUDA (opsiyonel)

## 📝 Notlar

- Bu proje eğitim/araştırma amaçlıdır
- Tıbbi tanı için profesyonel doktor görüşü alınmalıdır
- Veri seti Kaggle'dan indirilmelidir (büyük dosya boyutu nedeniyle repo'da bulunmaz)

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 📞 İletişim

Proje hakkında sorularınız için issue açabilirsiniz.