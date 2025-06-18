# Demo Talimatları - Duygu Analizi Projesi

## 🎬 Sunum Demo Rehberi

### Demo Öncesi Hazırlık

1. **Sistem Kontrolü**
   ```bash
   # Proje klasörüne git
   cd bitirme1
   
   # Kurulumu kontrol et
   python setup.py
   ```

2. **Model Dosyası Kontrolü**
   ```bash
   # Model dosyasının varlığını kontrol et
   ls data/models/
   # emotion_model.h5 dosyası olmalı
   ```

3. **Kamera Testi**
   ```bash
   # Kamera erişimini test et
   python -c "import cv2; cap = cv2.VideoCapture(0); print('Kamera OK' if cap.isOpened() else 'Kamera Hatası'); cap.release()"
   ```

### Demo Senaryoları

## 📋 Senaryo 1: Ana Menü Demo (5-7 dakika)

### Başlangıç
```bash
python run_project.py
```

### Demo Adımları:
1. **Ana menüyü göster** - Tüm seçenekleri açıkla
2. **GUI Versiyonu** - En görsel demo
3. **Özellikleri göster**:
   - Canlı video akışı
   - Duygu çubukları
   - FPS göstergesi
   - Kayıt özelliği

### Sunum Noktaları:
- "Bu ana menü ile tüm özelliklere kolayca erişebiliyoruz"
- "GUI versiyonu en kullanıcı dostu seçenek"
- "Gerçek zamanlı duygu analizi yapıyor"

## 📋 Senaryo 2: Teknik Demo (3-5 dakika)

### Komut Satırı Versiyonu
```bash
python test_model.py
```

### Demo Adımları:
1. **FPS göstergesini** vurgula
2. **İstatistik çıktılarını** göster
3. **'q' tuşu ile çıkış** yap
4. **Oluşan dosyaları** göster:
   ```bash
   ls sonuclar/
   ```

### Sunum Noktaları:
- "Komut satırı versiyonu daha teknik kullanıcılar için"
- "FPS değeri performansı gösteriyor"
- "Sonuçlar otomatik olarak kaydediliyor"

## 📋 Senaryo 3: Toplu Test Demo (2-3 dakika)

### Test Klasörü Hazırlama
```bash
# Test görüntülerini kopyala
cp data/raw/test/happy/*.jpg data/test/ 2>/dev/null || echo "Test görüntüleri hazır"
```

### Demo Çalıştırma
```bash
python batch_test.py
```

### Demo Adımları:
1. **İlerleme çubuğunu** göster
2. **Sonuçları** açıkla
3. **Oluşan grafikleri** göster:
   ```bash
   ls sonuclar/batch_test/
   ```

### Sunum Noktaları:
- "Toplu test ile çok sayıda görüntüyü analiz edebiliyoruz"
- "İstatistiksel analizler otomatik oluşturuluyor"
- "Görselleştirmeler sunum için hazır"

## 📋 Senaryo 4: Kod İnceleme (2-3 dakika)

### Önemli Dosyaları Göster

```bash
# Ana model sınıfını göster
cat src/models/emotion_detection.py | head -30

# GUI ana dosyasını göster  
cat gui_emotion_detection.py | head -30
```

### Sunum Noktaları:
- "Modüler tasarım kullandık"
- "Kod okunabilir ve genişletilebilir"
- "Sınıf tabanlı yapı"

## 🎯 Demo İpuçları

### Başarılı Demo İçin:
1. **Önceden test et** - Tüm komutları çalıştır
2. **Yedek plan hazırla** - Sorun çıkarsa alternatif demo
3. **Zamanı kontrol et** - Her senaryo için süre belirle
4. **Etkileşimli ol** - Sorular sor, geri bildirim al

### Teknik Sorunlar İçin:
- **Kamera çalışmıyorsa**: Önceden kaydedilmiş video kullan
- **Model yüklenmiyorsa**: Demo görüntüleri göster
- **Performans düşükse**: Daha düşük çözünürlük kullan

### Sunum Sırası:
1. **Proje tanıtımı** (1-2 dakika)
2. **Ana menü demo** (5-7 dakika)
3. **Teknik özellikler** (3-5 dakika)
4. **Kod inceleme** (2-3 dakika)
5. **Sorular** (2-3 dakika)

## 📊 Demo Sonrası

### Dosyaları Göster:
```bash
# Proje yapısı
tree -L 2

# Sonuçlar
ls -la sonuclar/

# Model dosyaları
ls -la data/models/
```

### Sunum Noktaları:
- "Proje tamamen çalışır durumda"
- "Tüm özellikler test edildi"
- "Dokümantasyon hazır"

## 🎓 Akademik Sunum İçin

### Vurgulanacak Noktalar:
1. **Yapay zeka uygulaması**: Gerçek dünya problemi
2. **Teknik beceriler**: Python, TensorFlow, OpenCV
3. **Yazılım mühendisliği**: Modüler tasarım, dokümantasyon
4. **İnovasyon**: Gerçek zamanlı işleme, kullanıcı dostu arayüz
5. **Gelecek planları**: Web arayüzü, mobil uygulama

### Sorulara Hazırlık:
- "Model doğruluğu nedir?" → Eğitim sonuçlarını göster
- "Performans nasıl?" → FPS değerlerini açıkla
- "Gelecek planları?" → SUNUM_RAPORU.md'den oku
- "Teknik zorluklar?" → Yüz tespiti, gerçek zamanlı işleme

---

**Demo Süresi**: 15-20 dakika  
**Hazırlık Süresi**: 30 dakika  
**Gerekli Ekipman**: Laptop, Kamera, Projeksiyon 