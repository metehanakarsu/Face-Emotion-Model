# Yapay Zeka Tabanlı Duygu Analizi Projesi

Bu proje, **FER2013** veri seti kullanılarak eğitilmiş derin öğrenme modeli ile gerçek zamanlı duygu analizi yapan bir sistemdir.

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# Kurulum scriptini çalıştır
python setup.py
```

### 2. Projeyi Çalıştır
```bash
# Ana menü ile başlat (Önerilen)
python run_project.py
```

## 📋 Özellikler

- ✅ **Gerçek zamanlı duygu analizi**: Webcam üzerinden canlı tespit
- ✅ **Yüz tanıma**: Tespit edilen yüzlerin tanınması  
- ✅ **Sosyal mesafe kontrolü**: Çoklu yüz tespitinde mesafe analizi
- ✅ **İstatistik toplama**: Duygu geçmişi ve analiz raporları
- ✅ **Kullanıcı dostu arayüz**: GUI ve komut satırı seçenekleri
- ✅ **Toplu test**: Klasördeki görüntüleri analiz etme

## 🎯 Çalıştırma Seçenekleri

### Ana Menü (Önerilen)
```bash
python run_project.py
```
- Tüm özellikleri içeren ana menü
- Kolay navigasyon
- Hata yönetimi

### GUI Versiyonu
```bash
python gui_emotion_detection.py
```
- Modern grafik arayüz
- Canlı video akışı
- Duygu çubukları
- Kayıt özelliği

### Gerçek Zamanlı Analiz
```bash
python test_model.py
```
- Komut satırı versiyonu
- FPS göstergesi
- İstatistik toplama
- Otomatik kayıt

### Toplu Test
```bash
python batch_test.py
```
- Klasör analizi
- İstatistik raporları
- Görselleştirme
- CSV/JSON çıktı

### Ana Program
```bash
python main.py
```
- Çoklu özellik
- Yüz tanıma + duygu analizi
- Sosyal mesafe kontrolü

## 📊 Model Bilgileri

- **Veri Seti**: FER2013 (35,887 görüntü)
- **Duygu Sınıfları**: 7 (Kızgın, İğrenmiş, Korkmuş, Mutlu, Üzgün, Şaşkın, Nötr)
- **Görüntü Boyutu**: 48x48 piksel (gri tonlama)
- **Model**: CNN tabanlı derin öğrenme

## 📁 Proje Yapısı

```
bitirme1/
├── data/
│   ├── models/                 # Eğitilmiş modeller
│   ├── raw/                    # Ham veri seti
│   └── test/                   # Test görüntüleri
├── src/                        # Kaynak kodlar
│   ├── models/                 # Model sınıfları
│   └── utils/                  # Yardımcı araçlar
├── sonuclar/                   # Analiz sonuçları
├── notebooks/                  # Jupyter notebook'lar
├── requirements.txt            # Bağımlılıklar
├── run_project.py             # Ana çalıştırma dosyası
├── setup.py                   # Kurulum scripti
└── README.md                  # Bu dosya
```

## 🔧 Gereksinimler

- Python 3.8+
- Webcam (gerçek zamanlı analiz için)
- 4GB+ RAM (önerilen)
- GPU (opsiyonel, hızlandırma için)

## 📈 Sonuçlar

Analiz sonuçları `sonuclar/` klasöründe saklanır:
- **JSON dosyaları**: Yapılandırılmış veri
- **CSV dosyaları**: Tablo formatında veri
- **Görseller**: Grafik ve analizler

## 🎓 Sunum Raporu

Detaylı sunum raporu için `SUNUM_RAPORU.md` dosyasını inceleyin.

## 🆘 Sorun Giderme

### Model Dosyası Bulunamadı
```bash
# Model dosyasını kontrol edin
ls data/models/
```

### Kamera Erişim Hatası
- Kamera izinlerini kontrol edin
- Başka uygulamaların kamerayı kullanmadığından emin olun

### Paket Yükleme Hatası
```bash
# Pip'i güncelleyin
python -m pip install --upgrade pip

# Sanal ortam kullanın
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

---

**Proje Sahibi**: [Öğrenci Adı]  
**Tarih**: 2024  
**Ders**: Bitirme Projesi 