import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Duygu sınıfları
EMOTIONS = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Uzgun', 'Saskin', 'Notr']

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
    model_path = "models/emotion_model.h5"
    
    try:
        # Modeli yükle
        model = load_model(model_path)
        print("Model başarıyla yüklendi!")
        
        # Yüz tespiti için cascade sınıflandırıcı
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Webcam'i başlat
        cap = cv2.VideoCapture(0)
        
        while True:
            # Kare yakala
            ret, frame = cap.read()
            if not ret:
                break
                
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
                
                # Sonuçları görüntü üzerine çiz
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"{emotion}: {probability:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Sonucu göster
            cv2.imshow("Duygu Analizi", frame_copy)
            
            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Hata: {str(e)}")

if __name__ == "__main__":
    realtime_emotion_detection() 