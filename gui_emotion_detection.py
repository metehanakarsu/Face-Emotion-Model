import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import os
from collections import deque

# Duygu sınıfları
EMOTIONS = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Uzgun', 'Saskin', 'Notr']

class EmotionGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Model yükle
        self.model = load_model("data/models/emotion_model.h5")
        print("Model başarıyla yüklendi!")
        
        # Yüz tespiti için cascade sınıflandırıcı
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Video yakalama nesnesi
        self.cap = None
        self.is_webcam_active = False
        
        # Duygu geçiş animasyonu için değişkenler
        self.current_emotions = {}  # Her yüz için mevcut duygu
        self.emotion_transitions = {}  # Her yüz için geçiş animasyonu
        self.transition_frames = 10  # Geçiş için kare sayısı
        
        # FPS hesaplama için değişkenler
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
        # Kayıt değişkenleri
        self.is_recording = False
        self.record_start_time = None
        self.emotion_history = deque(maxlen=300)  # Son 5 dakika
        
        # GUI bileşenleri
        self.create_gui()
        
        # Pencere kapatıldığında temizlik
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
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
        
        # Webcam başlat/durdur butonu
        self.webcam_btn = ttk.Button(control_frame, text="Webcam Başlat", command=self.toggle_webcam)
        self.webcam_btn.pack(side=tk.LEFT, padx=5)
        
        # Kayıt başlat/durdur butonu
        self.record_btn = ttk.Button(control_frame, text="Kayıt Başlat", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        # FPS ve süre göstergesi
        self.info_label = ttk.Label(control_frame, text="FPS: 0 | Süre: 00:00")
        self.info_label.pack(side=tk.LEFT, padx=5)
        
        # Duygu göstergeleri
        self.emotion_frame = ttk.Frame(main_frame)
        self.emotion_frame.grid(row=0, column=1, padx=5, pady=5, sticky="n")
        
        self.emotion_bars = {}
        for emotion in EMOTIONS:
            frame = ttk.Frame(self.emotion_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=emotion).pack(side=tk.LEFT)
            
            progress = ttk.Progressbar(frame, length=100, mode='determinate')
            progress.pack(side=tk.LEFT, padx=5)
            
            value_label = ttk.Label(frame, text="0%")
            value_label.pack(side=tk.LEFT)
            
            self.emotion_bars[emotion] = (progress, value_label)
    
    def update_emotion_bars(self, emotions):
        if not emotions:
            # Yüz tespit edilmediyse tüm çubukları sıfırla
            for emotion in EMOTIONS:
                progress, label = self.emotion_bars[emotion]
                progress['value'] = 0
                label['text'] = "0%"
            return
        
        # Tüm yüzlerin ortalama duygularını hesapla
        avg_emotions = {emotion: 0 for emotion in EMOTIONS}
        for face_emotions in emotions.values():
            for emotion, value in face_emotions.items():
                avg_emotions[emotion] += value
        
        # Ortalamaları al
        num_faces = len(emotions)
        for emotion in EMOTIONS:
            avg_emotions[emotion] /= num_faces
            
            # Progress bar'ı güncelle
            progress, label = self.emotion_bars[emotion]
            value = avg_emotions[emotion] * 100
            progress['value'] = value
            label['text'] = f"{value:.1f}%"
    
    def process_emotions(self, face_id, new_emotions):
        """Duygu geçişlerini yumuşat"""
        if face_id not in self.current_emotions:
            self.current_emotions[face_id] = new_emotions
            return new_emotions
        
        if face_id not in self.emotion_transitions:
            self.emotion_transitions[face_id] = {
                'start': self.current_emotions[face_id],
                'end': new_emotions,
                'step': 0
            }
        
        transition = self.emotion_transitions[face_id]
        
        # Geçiş tamamlandıysa yeni geçiş başlat
        if transition['step'] >= self.transition_frames:
            self.current_emotions[face_id] = transition['end']
            transition['start'] = transition['end']
            transition['end'] = new_emotions
            transition['step'] = 0
        
        # Ara değerleri hesapla
        smooth_emotions = {}
        for emotion in EMOTIONS:
            start_val = transition['start'][emotion]
            end_val = transition['end'][emotion]
            progress = transition['step'] / self.transition_frames
            smooth_emotions[emotion] = start_val + (end_val - start_val) * progress
        
        transition['step'] += 1
        return smooth_emotions
    
    def update_frame(self):
        if not self.is_webcam_active:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # FPS hesapla
            self.fps_counter += 1
            if (time.time() - self.fps_start_time) > 1:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Görüntüyü kopyala ve gri tonlamaya çevir
            frame_copy = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Yüzleri tespit et
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Tespit edilen tüm duyguları sakla
            current_face_emotions = {}
            
            # Her yüz için işlem yap
            for i, (x, y, w, h) in enumerate(faces):
                # Yüz bölgesini al
                roi = gray[y:y + h, x:x + w]
                
                # Görüntüyü ön işle
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                # Duygu tahmini yap
                predictions = self.model.predict(roi, verbose=0)[0]
                emotions = {EMOTIONS[i]: float(prob) for i, prob in enumerate(predictions)}
                
                # Duygu geçişlerini yumuşat
                smooth_emotions = self.process_emotions(i, emotions)
                current_face_emotions[i] = smooth_emotions
                
                # En yüksek olasılıklı duyguyu bul
                emotion = max(smooth_emotions.items(), key=lambda x: x[1])[0]
                probability = smooth_emotions[emotion]
                
                # Sonuçları görüntü üzerine çiz
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"{emotion}: {probability:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Duygu çubuklarını güncelle
            self.update_emotion_bars(current_face_emotions)
            
            # FPS ve süre bilgisini güncelle
            duration = "00:00"
            if self.is_recording and self.record_start_time:
                seconds = int(time.time() - self.record_start_time)
                duration = f"{seconds//60:02d}:{seconds%60:02d}"
            self.info_label['text'] = f"FPS: {self.fps} | Süre: {duration}"
            
            # Kayıt modundaysa duyguları kaydet
            if self.is_recording and current_face_emotions:
                timestamp = datetime.now()
                avg_emotions = {emotion: 0 for emotion in EMOTIONS}
                for face_emotions in current_face_emotions.values():
                    for emotion, value in face_emotions.items():
                        avg_emotions[emotion] += value
                num_faces = len(current_face_emotions)
                for emotion in avg_emotions:
                    avg_emotions[emotion] /= num_faces
                self.emotion_history.append((timestamp, avg_emotions))
            
            # Görüntüyü GUI'de göster
            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(image=img)
            
            self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
            self.canvas.img = img
        
        self.window.after(10, self.update_frame)
    
    def toggle_webcam(self):
        if not self.is_webcam_active:
            self.cap = cv2.VideoCapture(0)
            self.is_webcam_active = True
            self.webcam_btn['text'] = "Webcam Durdur"
            self.update_frame()
        else:
            self.is_webcam_active = False
            self.webcam_btn['text'] = "Webcam Başlat"
            if self.cap:
                self.cap.release()
    
    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_btn['text'] = "Kayıt Durdur"
            self.record_start_time = time.time()
            self.emotion_history.clear()
        else:
            self.is_recording = False
            self.record_btn['text'] = "Kayıt Başlat"
            self.save_recording()
    
    def save_recording(self):
        if not self.emotion_history:
            return
            
        # Kayıt klasörünü oluştur
        output_dir = "sonuclar/kayitlar"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Zaman damgası
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV dosyasına kaydet
        import pandas as pd
        data = []
        for ts, emotions in self.emotion_history:
            row = {'timestamp': ts}
            row.update(emotions)
            data.append(row)
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, f'duygu_kaydi_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Kayıt kaydedildi: {csv_path}")
    
    def on_closing(self):
        self.is_webcam_active = False
        if self.cap:
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionGUI(root, "Duygu Analizi")
    root.mainloop() 