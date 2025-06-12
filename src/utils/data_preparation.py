import os
import pandas as pd
from pathlib import Path

def prepare_dataset():
    """
    Veri setini kontrol et ve hazırla
    """
    try:
        # CSV dosyasını oku
        data = pd.read_csv('data/raw/fer2013.csv')
        
        # Duygu etiketlerini kontrol et
        emotion_counts = data['emotion'].value_counts()
        print("\nDuygu dağılımı:")
        emotions = ['Kızgın', 'İğrenmiş', 'Korkmuş', 'Mutlu', 'Üzgün', 'Şaşkın', 'Nötr']
        for idx, count in emotion_counts.items():
            print(f"{emotions[idx]}: {count}")
            
        print(f"\nToplam görüntü sayısı: {len(data)}")
        return True
        
    except Exception as e:
        print(f"Veri seti hazırlanırken hata oluştu: {str(e)}")
        return False

if __name__ == "__main__":
    prepare_dataset() 