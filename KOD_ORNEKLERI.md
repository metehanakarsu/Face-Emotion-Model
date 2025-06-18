# Kod Örnekleri - Duygu Analizi Projesi

## 🎯 Sunum Raporu İçin Ana Kodlar

### 1. Ana Model Sınıfı - Duygu Analizi

```python
# src/models/emotion_detection.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class EmotionDetector:
    def __init__(self, model_path='data/models/emotion_model.h5'):
        self.emotions = ['Kızgın', 'İğrenmiş', 'Korkmuş', 'Mutlu', 'Üzgün', 'Şaşkın', 'Nötr']
        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        emotions_detected = []
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = self.model.predict(roi)[0]
            emotion_idx = preds.argmax()
            emotion = self.emotions[emotion_idx]
            emotions_detected.append(emotion)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame, emotions_detected
```

### 2. GUI Ana Sınıfı

```python
# gui_emotion_detection.py
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading

class EmotionGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Model yükle
        self.model = load_model("data/models/emotion_model.h5")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Video yakalama
        self.cap = None
        self.is_webcam_active = False
        
        # GUI bileşenleri
        self.create_gui()
        
    def create_gui(self):
        # Ana çerçeve
        main_frame = ttk.Frame(self.window)
        main_frame.pack(padx=10, pady=10)
        
        # Video görüntüleme alanı
        self.video_frame = ttk.Frame(main_frame)
        self.video_frame.grid(row=0, column=0, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()
        
        # Kontrol paneli
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, padx=5, pady=5)
        
        self.webcam_btn = ttk.Button(control_frame, text="Webcam Başlat", 
                                    command=self.toggle_webcam)
        self.webcam_btn.pack(side=tk.LEFT, padx=5)
        
        # Duygu göstergeleri
        self.emotion_frame = ttk.Frame(main_frame)
        self.emotion_frame.grid(row=0, column=1, padx=5, pady=5, sticky="n")
        
        self.emotion_bars = {}
        for emotion in ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Uzgun', 'Saskin', 'Notr']:
            frame = ttk.Frame(self.emotion_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=emotion).pack(side=tk.LEFT)
            progress = ttk.Progressbar(frame, length=100, mode='determinate')
            progress.pack(side=tk.LEFT, padx=5)
            
            self.emotion_bars[emotion] = progress
```

### 3. Gerçek Zamanlı Analiz

```python
# test_model.py
def realtime_emotion_detection():
    # Model yükle
    model = load_model("data/models/emotion_model.h5")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Webcam başlat
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüzleri tespit et
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        # Her yüz için işlem yap
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Duygu tahmini
            predictions = model.predict(roi, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion = EMOTIONS[emotion_idx]
            probability = predictions[emotion_idx]
            
            # Sonuçları çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion}: {probability:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Duygu Analizi", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

### 4. Toplu Test İşlemi

```python
# batch_test.py
def batch_test(test_dir, output_dir='sonuclar/batch_test'):
    # Model yükle
    model = load_model("data/models/emotion_model.h5")
    
    # Görüntü dosyalarını bul
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend([f for f in os.listdir(test_dir) 
                           if f.lower().endswith(ext[1:])])
    
    results = []
    
    # Her görüntü için tahmin yap
    for img_file in tqdm(image_files):
        img_path = os.path.join(test_dir, img_file)
        
        # Görüntüyü yükle ve ön işle
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        # Duygu tahmini
        predictions = model.predict(img, verbose=0)
        emotion_probs = {EMOTIONS[i]: float(prob) for i, prob in enumerate(predictions[0])}
        emotion = max(emotion_probs.items(), key=lambda x: x[1])
        
        # Sonuçları kaydet
        result = {
            'dosya_adi': img_file,
            'tahmin_edilen_duygu': emotion[0],
            'guven_skoru': emotion[1],
            **emotion_probs
        }
        results.append(result)
    
    # Sonuçları DataFrame'e dönüştür ve kaydet
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'batch_test_sonuclari.csv'), index=False)
    
    return results_df
```

### 5. Ana Program

```python
# main.py
def main():
    # Kamera başlat
    cap = cv2.VideoCapture(0)
    
    # Dedektörleri başlat
    face_detector = FaceDetector()
    emotion_detector = EmotionDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Yüz tespiti
        frame, faces = face_detector.detect_faces(frame)
        
        # Duygu analizi
        frame, emotions = emotion_detector.detect_emotion(frame)
        
        # Sosyal mesafe kontrolü
        if len(faces) > 1:
            violations = face_detector.calculate_social_distance(faces)
            for i, j in violations:
                pt1 = (faces[i][0] + faces[i][2]//2, faces[i][1] + faces[i][3]//2)
                pt2 = (faces[j][0] + faces[j][2]//2, faces[j][1] + faces[j][3]//2)
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
        
        # Ekranda göster
        cv2.imshow('Gerçek Zamanlı Analiz', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

### 6. Veri Ön İşleme

```python
# src/utils/data_preparation.py
def preprocess_image(image_path, target_size=(48, 48)):
    """Görüntüyü model için hazırlar"""
    # Görüntüyü yükle
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Yeniden boyutlandır
    img = cv2.resize(img, target_size)
    
    # Normalize et
    img = img.astype("float") / 255.0
    
    # Boyutları ayarla
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    return img

def load_and_preprocess_batch(image_paths, target_size=(48, 48)):
    """Toplu görüntü yükleme ve ön işleme"""
    processed_images = []
    
    for path in image_paths:
        try:
            img = preprocess_image(path, target_size)
            processed_images.append(img)
        except Exception as e:
            print(f"Hata ({path}): {str(e)}")
    
    return np.vstack(processed_images)
```

### 7. İstatistik Toplama

```python
# test_model.py - EmotionStats sınıfı
class EmotionStats:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.emotion_history = deque(maxlen=window_size)
        self.session_stats = {emotion: 0 for emotion in EMOTIONS}
        self.session_start = datetime.now()
        
    def update(self, emotion, probability):
        current_time = datetime.now()
        self.emotion_history.append((emotion, probability))
        self.session_stats[emotion] += 1
        
    def get_current_stats(self):
        if not self.emotion_history:
            return {}
        
        # Son 30 saniyedeki duygu dağılımı
        recent_emotions = [e[0] for e in self.emotion_history]
        emotion_counts = {emotion: recent_emotions.count(emotion) 
                         for emotion in EMOTIONS}
        total = len(recent_emotions)
        
        return {
            'son_30_saniye': {
                emotion: count/total for emotion, count in emotion_counts.items()
            },
            'genel_istatistik': {
                emotion: count/sum(self.session_stats.values()) 
                for emotion, count in self.session_stats.items()
            }
        }
```

### 8. Model Eğitimi (Jupyter Notebook)

```python
# notebooks/model_development.ipynb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

def create_emotion_model():
    """Duygu analizi için CNN modeli oluşturur"""
    model = Sequential([
        # İlk konvolüsyon bloğu
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # İkinci konvolüsyon bloğu
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Üçüncü konvolüsyon bloğu
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Düzleştirme ve tam bağlantılı katmanlar
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 duygu sınıfı
    ])
    
    return model

# Model oluştur ve derle
model = create_emotion_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model eğitimi
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# Modeli kaydet
model.save('data/models/emotion_model.h5')
```

### 9. Ana Çalıştırma Menüsü

```python
# run_project.py
def main():
    """Ana menü ve program seçenekleri"""
    
    def run_gui():
        """GUI versiyonunu çalıştır"""
        try:
            from gui_emotion_detection import EmotionGUI
            root = tk.Tk()
            app = EmotionGUI(root, "Duygu Analizi - GUI")
            root.mainloop()
        except Exception as e:
            messagebox.showerror("Hata", f"GUI başlatılamadı: {str(e)}")
    
    def run_realtime():
        """Gerçek zamanlı analizi çalıştır"""
        try:
            from test_model import realtime_emotion_detection
            realtime_emotion_detection()
        except Exception as e:
            print(f"Hata: {str(e)}")
    
    # Ana pencere oluştur
    root = tk.Tk()
    root.title("Duygu Analizi Projesi - Ana Menü")
    root.geometry("400x300")
    
    # Başlık
    title_label = tk.Label(root, text="Yapay Zeka Tabanlı\nDuygu Analizi Projesi", 
                          font=("Arial", 16, "bold"))
    title_label.pack(pady=20)
    
    # Butonlar
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    tk.Button(button_frame, text="GUI Versiyonu Çalıştır", 
              command=run_gui, width=25, height=2).pack(pady=5)
    
    tk.Button(button_frame, text="Gerçek Zamanlı Analiz", 
              command=run_realtime, width=25, height=2).pack(pady=5)
    
    root.mainloop()
```

## 📊 Kod İstatistikleri

- **Toplam Satır**: ~800 satır
- **Python Dosyaları**: 8 ana dosya
- **Sınıf Sayısı**: 4 ana sınıf
- **Fonksiyon Sayısı**: 15+ fonksiyon
- **Kullanılan Kütüphaneler**: 10+ kütüphane

## 🎯 Sunum İçin Önemli Kod Noktaları

1. **Modüler Tasarım**: Her özellik ayrı sınıflarda
2. **Hata Yönetimi**: Try-catch blokları ile güvenli çalışma
3. **Performans Optimizasyonu**: Gerçek zamanlı işleme
4. **Kullanıcı Dostu**: GUI ve komut satırı seçenekleri
5. **Genişletilebilir**: Yeni özellikler kolayca eklenebilir

---

**Bu kodlar sunum raporuna eklenebilir ve projenin teknik detaylarını göstermek için kullanılabilir.** 