import cv2
import numpy as np
import mediapipe as mp
import face_recognition
from typing import List, Tuple

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
    def detect_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Görüntüdeki yüzleri tespit eder ve konumlarını döndürür
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        faces = []
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                )
                faces.append(bbox)
                # Yüz çerçevesini çiz
                cv2.rectangle(frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                            (0, 255, 0), 2)
                
        return frame, faces

    def recognize_faces(self, frame: np.ndarray, known_face_encodings: List, known_face_names: List) -> Tuple[np.ndarray, List]:
        """
        Tespit edilen yüzleri tanır ve isimlendirir
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Bilinmeyen"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            face_names.append(name)
            
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
        return frame, face_names

    def calculate_social_distance(self, faces: List, min_distance: float = 100) -> List[Tuple]:
        """
        Yüzler arasındaki sosyal mesafeyi hesaplar
        """
        violations = []
        for i, face1 in enumerate(faces):
            for j, face2 in enumerate(faces[i+1:], i+1):
                x1 = face1[0] + face1[2]//2
                y1 = face1[1] + face1[3]//2
                x2 = face2[0] + face2[2]//2
                y2 = face2[1] + face2[3]//2
                
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance < min_distance:
                    violations.append((i, j))
                    
        return violations 