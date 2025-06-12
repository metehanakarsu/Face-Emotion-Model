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
            gray = cv2.cvtColor(frame, cv2.COLOR