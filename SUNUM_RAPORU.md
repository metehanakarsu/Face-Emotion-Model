# Yapay Zeka Tabanlı Duygu Analizi Projesi - Sunum Raporu

## 📋 Proje Özeti

Bu proje, **FER2013** veri seti kullanılarak eğitilmiş derin öğrenme modeli ile gerçek zamanlı duygu analizi yapan bir sistemdir. Proje, yüz tespiti ve duygu sınıflandırması özelliklerini birleştirerek kullanıcı dostu bir arayüz sunar.

## 🎯 Proje Hedefleri

- **Gerçek zamanlı duygu analizi**: Webcam üzerinden canlı duygu tespiti
- **Yüz tanıma**: Tespit edilen yüzlerin tanınması
- **Sosyal mesafe kontrolü**: Çoklu yüz tespitinde mesafe analizi
- **İstatistik toplama**: Duygu geçmişi ve analiz raporları
- **Kullanıcı dostu arayüz**: GUI ve komut satırı seçenekleri

## 🏗️ Proje Mimarisi

### Ana Bileşenler

```
src/
├── models/
│   ├── emotion_detection.py    # Duygu analizi sınıfı
│   └── face_detection.py       # Yüz tespiti sınıfı
├── utils/
│   └── data_preparation.py     # Veri hazırlama araçları
└── web/                        # Web arayüzü (gelecek)
```

### Kullanılan Teknolojiler

- **TensorFlow/Keras**: Derin öğrenme modeli
- **OpenCV**: Görüntü işleme ve yüz tespiti
- **MediaPipe**: Gelişmiş yüz tespiti
- **Tkinter**: GUI arayüzü
- **NumPy/Pandas**: Veri işleme ve analiz

## 📊 Model Performansı

### Eğitim Veri Seti
- **FER2013**: 35,887 görüntü
- **7 duygu sınıfı**: Kızgın, İğrenmiş, Korkmuş, Mutlu, Üzgün, Şaşkın, Nötr
- **Görüntü boyutu**: 48x48 piksel (gri tonlama)

### Model Mimarisi
```python
# CNN tabanlı model
- Giriş katmanı: 48x48x1
- Konvolüsyon katmanları
- MaxPooling katmanları
- Dropout katmanları
- Dense katmanları
- Çıkış: 7 sınıf (softmax)
```

## 🚀 Proje Çalıştırma

### 1. Kurulum
```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# Kurulum scriptini çalıştır
python setup.py
```

### 2. Çalıştırma Seçenekleri

#### Ana Menü (Önerilen)
```bash
python run_project.py
```

#### GUI Versiyonu
```bash
python gui_emotion_detection.py
```

#### Gerçek Zamanlı Analiz
```bash
python test_model.py
```

#### Toplu Test
```bash
python batch_test.py
```

## 📈 Özellikler ve Fonksiyonlar

### 1. Gerçek Zamanlı Analiz (`test_model.py`)
- **FPS göstergesi**: Performans takibi
- **İstatistik toplama**: 30 saniyelik pencere
- **Otomatik kayıt**: JSON ve CSV formatında
- **Duygu geçiş animasyonu**: Yumuşak geçişler

### 2. GUI Arayüzü (`gui_emotion_detection.py`)
- **Canlı video akışı**: 640x480 çözünürlük
- **Duygu çubukları**: Anlık duygu oranları
- **Kayıt özelliği**: Oturum kaydetme
- **FPS ve süre göstergesi**: Performans metrikleri

### 3. Toplu Test (`batch_test.py`)
- **Klasör analizi**: Toplu görüntü işleme
- **İstatistik raporları**: Detaylı analiz
- **Görselleştirme**: Pasta ve çubuk grafikleri
- **CSV/JSON çıktı**: Veri analizi için

### 4. Ana Program (`main.py`)
- **Çoklu özellik**: Yüz tanıma + duygu analizi
- **Sosyal mesafe**: Çoklu yüz tespitinde
- **Gerçek zamanlı**: Kamera akışı

## 📁 Proje Yapısı

```
bitirme1/
├── data/
│   ├── models/                 # Eğitilmiş modeller
│   ├── raw/                    # Ham veri seti
│   └── test/                   # Test görüntüleri
├── src/                        # Kaynak kodlar
├── sonuclar/                   # Analiz sonuçları
├── notebooks/                  # Jupyter notebook'lar
├── requirements.txt            # Bağımlılıklar
├── run_project.py             # Ana çalıştırma dosyası
├── setup.py                   # Kurulum scripti
└── README.md                  # Proje dokümantasyonu
```

## 🎨 Kullanıcı Arayüzü

### GUI Özellikleri
- **Modern tasarım**: Renkli butonlar ve göstergeler
- **Gerçek zamanlı güncelleme**: Anlık duygu değişimleri
- **Kullanıcı dostu**: Kolay navigasyon
- **Çoklu mod**: Webcam ve dosya yükleme

### Komut Satırı Özellikleri
- **Hızlı başlatma**: Minimal kurulum
- **Batch işleme**: Toplu analiz
- **Detaylı çıktı**: İstatistik ve raporlar

## 📊 Sonuçlar ve Analiz

### Performans Metrikleri
- **FPS**: 15-25 (sistem performansına bağlı)
- **Doğruluk**: Model eğitim sonuçlarına göre
- **Gecikme**: Gerçek zamanlı işleme

### Çıktı Formatları
- **JSON**: Yapılandırılmış veri
- **CSV**: Tablo formatında veri
- **Görseller**: Grafik ve analizler

## 🔧 Teknik Detaylar

### Yüz Tespiti
```python
# OpenCV Cascade Classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```

### Duygu Analizi
```python
# Model tahmini
predictions = model.predict(processed_image)
emotion = EMOTIONS[np.argmax(predictions)]
```

### Veri Ön İşleme
```python
# Görüntü hazırlama
image = cv2.resize(image, (48, 48))
image = image.astype("float") / 255.0
```

## 🎓 Akademik Katkı

### Kullanılan Yöntemler
- **Derin Öğrenme**: CNN mimarisi
- **Görüntü İşleme**: OpenCV kütüphanesi
- **Makine Öğrenmesi**: TensorFlow framework
- **Yazılım Mühendisliği**: Modüler tasarım

### İnovatif Özellikler
- **Gerçek zamanlı işleme**: Düşük gecikme
- **Çoklu yüz analizi**: Paralel işleme
- **İstatistik toplama**: Veri analizi
- **Kullanıcı dostu arayüz**: Erişilebilirlik

## 🚀 Gelecek Geliştirmeler

### Planlanan Özellikler
- **Web arayüzü**: Flask tabanlı
- **Mobil uygulama**: Android/iOS
- **API servisi**: RESTful endpoint'ler
- **Veritabanı entegrasyonu**: Kullanıcı verileri

### Teknik İyileştirmeler
- **Model optimizasyonu**: Daha hızlı işleme
- **Doğruluk artırma**: Daha iyi tahmin
- **Çoklu dil desteği**: Uluslararasılaştırma

## 📝 Sonuç

Bu proje, yapay zeka ve görüntü işleme tekniklerini kullanarak pratik bir duygu analizi sistemi geliştirmeyi başarmıştır. Gerçek zamanlı işleme, kullanıcı dostu arayüz ve kapsamlı analiz özellikleri ile akademik ve ticari uygulamalar için uygun bir temel oluşturmaktadır.

---

**Proje Sahibi**: [Öğrenci Adı]  
**Tarih**: 2024  
**Ders**: Bitirme Projesi  
**Danışman**: [Danışman Adı] 