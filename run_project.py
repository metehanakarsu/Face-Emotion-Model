#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yapay Zeka Tabanlı Duygu Analizi Projesi - Ana Çalıştırma Dosyası
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

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
    
    def run_batch_test():
        """Toplu test işlemini çalıştır"""
        try:
            from batch_test import batch_test
            test_dir = "data/test"
            if not os.path.exists(test_dir):
                print(f"Test klasörü bulunamadı: {test_dir}")
                return
            batch_test(test_dir)
        except Exception as e:
            print(f"Hata: {str(e)}")
    
    def run_main():
        """Ana programı çalıştır"""
        try:
            from main import main as main_program
            main_program()
        except Exception as e:
            print(f"Hata: {str(e)}")
    
    # Ana pencere oluştur
    root = tk.Tk()
    root.title("Duygu Analizi Projesi - Ana Menü")
    root.geometry("400x300")
    root.configure(bg='#f0f0f0')
    
    # Başlık
    title_label = tk.Label(root, text="Yapay Zeka Tabanlı\nDuygu Analizi Projesi", 
                          font=("Arial", 16, "bold"), bg='#f0f0f0')
    title_label.pack(pady=20)
    
    # Butonlar
    button_frame = tk.Frame(root, bg='#f0f0f0')
    button_frame.pack(pady=20)
    
    tk.Button(button_frame, text="GUI Versiyonu Çalıştır", 
              command=run_gui, width=25, height=2, bg='#4CAF50', fg='white').pack(pady=5)
    
    tk.Button(button_frame, text="Gerçek Zamanlı Analiz", 
              command=run_realtime, width=25, height=2, bg='#2196F3', fg='white').pack(pady=5)
    
    tk.Button(button_frame, text="Toplu Test İşlemi", 
              command=run_batch_test, width=25, height=2, bg='#FF9800', fg='white').pack(pady=5)
    
    tk.Button(button_frame, text="Ana Program", 
              command=run_main, width=25, height=2, bg='#9C27B0', fg='white').pack(pady=5)
    
    # Çıkış butonu
    tk.Button(button_frame, text="Çıkış", 
              command=root.quit, width=25, height=2, bg='#f44336', fg='white').pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    main() 