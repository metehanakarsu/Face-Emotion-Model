import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
from datetime import datetime
import json
from collections import deque
import pandas as pd
import os

# Duygu sınıfları
EMOTIONS = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Uzgun', 'Saskin', 'Notr']

class EmotionStats:
    def __init__(self, window_size=30):  # 30 saniyelik pencere
        self.window_size = window_size
        self.emotion_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.session_stats = {emotion: 0 for emotion in EMOTIONS}
        self.session_start = datetime.now()
        
    def update(self, emotion, probability):
        current_time = datetime.now()
        self.emotion_history.append((emotion, probability))
        self.timestamps.append(current_time)
        self.session_stats[emotion] += 1
        
    def get_current_stats(self):
        if not self.emotion_history:
            return {}
        
        # Son 30 saniyedeki duygu dağılımı
        recent_emotions = [e[0] for e in self.emotion_history]
        emotion_counts = {emotion: recent_emotions.count(emotion) for emotion in EMOTIONS}
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
    
    def save_session_stats(self, output_dir='sonuclar'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        session_data = {
            'baslangic_zamani': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'bitis_zamani': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'toplam_tespit': sum(self.session_stats.values()),
            'duygu_dagilimi': self.session_stats
        }
        
        # JSON olarak kaydet
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(output_dir, f'duygu_analizi_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)
        
        # CSV olarak kaydet
        if self.emotion_history:
            df = pd.DataFrame({
                'zaman': list(self.timestamps),
                'duygu': [e[0] for e in self.emotion_history],
                'olasilik': [e[1] for e in self.emotion_history]
            })
            csv_path = os.path.join(output_dir, f'duygu_gecmisi_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
        
        return json_path

def load_and_preprocess_image(image, target_size=(48, 48)):
    # Görüntüyü gri tonlamaya çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Yeniden boyutlandır
    roi = cv2.resize(gray, target_size)
    
    # Normalize et ve boyutları ayarla
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    return roi

def predict_emotion(model, image):
    # Tahmin yap
    predictions = model.predict(image, verbose=0)
    
    # En yüksek olasılıklı duyguyu bul
    emotion_idx = np.argmax(predictions[0])
    emotion = EMOTIONS[emotion_idx]
    probability = predictions[0][emotion_idx]
    
    return emotion, probability

def realtime_emotion_detection():
    # Model yolu
    model_path = "data/models/emotion_model.h5"
    
    try:
        # Modeli yükle
        model = load_model(model_path)
        print("Model başarıyla yüklendi!")
        
        # İstatistik toplayıcıyı başlat
        stats = EmotionStats()
        
        # Yüz tespiti için cascade sınıflandırıcı
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Webcam'i başlat
        cap = cv2.VideoCapture(0)
        
        # FPS hesaplama için değişkenler
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            # Kare yakala
            ret, frame = cap.read()
            if not ret:
                break
                
            # FPS hesapla
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
                
                # Her saniye istatistikleri güncelle ve göster
                current_stats = stats.get_current_stats()
                if current_stats:
                    print("\nGüncel Duygu Dağılımı (Son 30 saniye):")
                    for emotion, ratio in current_stats['son_30_saniye'].items():
                        print(f"{emotion}: {ratio:.2%}")
            
            # Görüntüyü kopyala
            frame_copy = frame.copy()
            
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Yüzleri tespit et
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Her yüz için işlem yap
            for (x, y, w, h) in faces:
                # Yüz bölgesini al
                roi = gray[y:y + h, x:x + w]
                
                # Görüntüyü ön işle
                processed_roi = load_and_preprocess_image(roi)
                
                # Duygu tahmini yap
                emotion, probability = predict_emotion(model, processed_roi)
                
                # İstatistikleri güncelle
                stats.update(emotion, probability)
                
                # Sonuçları görüntü üzerine çiz
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"{emotion}: {probability:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # FPS'i göster
            cv2.putText(frame_copy, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Sonucu göster
            cv2.imshow("Duygu Analizi", frame_copy)
            
            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Oturumu kaydet
        saved_path = stats.save_session_stats()
        print(f"\nOturum istatistikleri kaydedildi: {saved_path}")
        
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Hata: {str(e)}")

if __name__ == "__main__":
    realtime_emotion_detection() 