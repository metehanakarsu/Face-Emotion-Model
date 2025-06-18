#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proje Kurulum Dosyası
"""

import os
import sys
import subprocess
import platform

def install_requirements():
    """Gerekli paketleri yükle"""
    print("Gerekli paketler yükleniyor...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Paketler başarıyla yüklendi!")
    except subprocess.CalledProcessError:
        print("❌ Paket yükleme hatası!")
        return False
    return True

def create_directories():
    """Gerekli klasörleri oluştur"""
    directories = [
        "data/test",
        "sonuclar",
        "sonuclar/batch_test",
        "sonuclar/kayitlar"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Klasör oluşturuldu: {directory}")

def check_model_files():
    """Model dosyalarını kontrol et"""
    model_path = "data/models/emotion_model.h5"
    if not os.path.exists(model_path):
        print("⚠️  Model dosyası bulunamadı!")
        print("Model dosyasını 'data/models/' klasörüne yerleştirin.")
        return False
    print("✅ Model dosyası bulundu!")
    return True

def main():
    """Ana kurulum fonksiyonu"""
    print("=" * 50)
    print("Yapay Zeka Tabanlı Duygu Analizi Projesi")
    print("Kurulum Başlatılıyor...")
    print("=" * 50)
    
    # Klasörleri oluştur
    create_directories()
    
    # Paketleri yükle
    if not install_requirements():
        return
    
    # Model dosyalarını kontrol et
    if not check_model_files():
        print("\n❌ Kurulum tamamlanamadı!")
        return
    
    print("\n" + "=" * 50)
    print("✅ Kurulum tamamlandı!")
    print("Projeyi çalıştırmak için:")
    print("python run_project.py")
    print("=" * 50)

if __name__ == "__main__":
    main() 