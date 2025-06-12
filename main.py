import cv2
import numpy as np
from src.models.face_detection import FaceDetector
from src.models.emotion_detection import EmotionDetector

def main():
    # Kamera başlat
    cap = cv2.VideoCapture(0)
    
    # Dedektörleri başlat
    face_detector = FaceDetector()
    emotion_detector = EmotionDetector()
    
    # Bilinen yüzler için boş listeler (daha sonra doldurulacak)
    known_face_encodings = []
    known_face_names = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Yüz tespiti
        frame, faces = face_detector.detect_faces(frame)
        
        # Yüz tanıma
        if known_face_encodings:
            frame, names = face_detector.recognize_faces(frame, known_face_encodings, known_face_names)
        
        # Duygu analizi
        frame, emotions = emotion_detector.detect_emotion(frame)
        
        # Sosyal mesafe kontrolü
        if len(faces) > 1:
            violations = face_detector.calculate_social_distance(faces)
            for i, j in violations:
                # Sosyal mesafe ihlallerini kırmızı çizgi ile göster
                pt1 = (faces[i][0] + faces[i][2]//2, faces[i][1] + faces[i][3]//2)
                pt2 = (faces[j][0] + faces[j][2]//2, faces[j][1] + faces[j][3]//2)
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
        
        # Ekranda göster
        cv2.imshow('Gerçek Zamanlı Analiz', frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 