import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from typing import Tuple, List

class EmotionDetector:
    def __init__(self, model_path: str = 'data/models/emotion_model.h5'):
        """
        Duygu analizi sınıfı
        """
        self.emotions = ['Kızgın', 'İğrenmiş', 'Korkmuş', 'Mutlu', 'Üzgün', 'Şaşkın', 'Nötr']
        try:
            self.model = load_model(model_path)
        except:
            print(f"Model dosyası bulunamadı: {model_path}")
            self.model = None
        
        # Yüz tespiti için cascade sınıflandırıcı
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Yüz görüntüsünü model için hazırlar
        """
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype("float") / 255.0
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def detect_emotion(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Görüntüdeki yüzlerin duygularını tespit eder
        """
        if self.model is None:
            return frame, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        emotions_detected = []
        
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            roi = self.preprocess_face(roi)
            
            preds = self.model.predict(roi)[0]
            emotion_idx = preds.argmax()
            emotion = self.emotions[emotion_idx]
            emotions_detected.append(emotion)
            
            # Yüz ve duygu etiketini çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame, emotions_detected

    def get_emotion_stats(self, emotions: List[str]) -> dict:
        """
        Tespit edilen duyguların istatistiklerini hesaplar
        """
        stats = {emotion: emotions.count(emotion) for emotion in self.emotions}
        return stats 