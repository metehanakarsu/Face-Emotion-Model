import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Duygu sınıfları
EMOTIONS = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Uzgun', 'Saskin', 'Notr']

def load_and_preprocess_image(image_path, target_size=(48, 48)):
    # Görüntüyü yükle
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    
    # Yeniden boyutlandır
    img = cv2.resize(img, target_size)
    
    # Normalize et ve boyutları ayarla
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_emotion(model, image):
    # Tahmin yap
    predictions = model.predict(image, verbose=0)
    
    # Tüm olasılıkları al
    emotion_probs = {EMOTIONS[i]: float(prob) for i, prob in enumerate(predictions[0])}
    
    # En yüksek olasılıklı duyguyu bul
    emotion = max(emotion_probs.items(), key=lambda x: x[1])
    
    return emotion[0], emotion_probs

def create_results_visualizations(results_df, output_dir):
    # 1. Duygu Dağılımı Pasta Grafiği
    plt.figure(figsize=(10, 6))
    emotion_counts = results_df['tahmin_edilen_duygu'].value_counts()
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%')
    plt.title('Tahmin Edilen Duyguların Dağılımı')
    plt.savefig(os.path.join(output_dir, 'duygu_dagilimi_pasta.png'))
    plt.close()
    
    # 2. Ortalama Güven Skorları Çubuk Grafiği
    plt.figure(figsize=(12, 6))
    avg_confidence = results_df.groupby('tahmin_edilen_duygu')['guven_skoru'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_confidence.index, y=avg_confidence.values)
    plt.title('Duygulara Göre Ortalama Güven Skorları')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ortalama_guven_skorlari.png'))
    plt.close()
    
    # 3. Güven Skoru Dağılımı (Violin Plot)
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=results_df, x='tahmin_edilen_duygu', y='guven_skoru')
    plt.title('Duygulara Göre Güven Skoru Dağılımı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'guven_skoru_dagilimi.png'))
    plt.close()

def batch_test(test_dir, output_dir='sonuclar/batch_test'):
    """
    Belirtilen klasördeki tüm görüntüler üzerinde duygu analizi yapar.
    
    Args:
        test_dir: Test görüntülerinin bulunduğu klasör
        output_dir: Sonuçların kaydedileceği klasör
    """
    # Çıktı klasörünü oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Model yolu
    model_path = "data/models/emotion_model.h5"
    
    try:
        # Modeli yükle
        model = load_model(model_path)
        print("Model başarıyla yüklendi!")
        
        # Görüntü dosyalarını bul
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend([f for f in os.listdir(test_dir) if f.lower().endswith(ext[1:])])
        
        if not image_files:
            raise ValueError(f"Test klasöründe görüntü bulunamadı: {test_dir}")
        
        # Sonuçları saklamak için liste
        results = []
        
        # Her görüntü için tahmin yap
        print("\nGörüntüler analiz ediliyor...")
        for img_file in tqdm(image_files):
            img_path = os.path.join(test_dir, img_file)
            try:
                # Görüntüyü yükle ve ön işle
                processed_img = load_and_preprocess_image(img_path)
                
                # Duygu tahmini yap
                emotion, all_probs = predict_emotion(model, processed_img)
                
                # Sonuçları kaydet
                result = {
                    'dosya_adi': img_file,
                    'tahmin_edilen_duygu': emotion,
                    'guven_skoru': all_probs[emotion],
                    **all_probs
                }
                results.append(result)
                
            except Exception as e:
                print(f"\nHata ({img_file}): {str(e)}")
        
        # Sonuçları DataFrame'e dönüştür
        results_df = pd.DataFrame(results)
        
        # Zaman damgası
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sonuçları kaydet
        csv_path = os.path.join(output_dir, f'batch_test_sonuclari_{timestamp}.csv')
        results_df.to_csv(csv_path, index=False)
        
        # Özet istatistikler
        summary = {
            'test_zamani': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'toplam_goruntu': len(image_files),
            'basarili_tahmin': len(results),
            'duygu_dagilimi': results_df['tahmin_edilen_duygu'].value_counts().to_dict(),
            'ortalama_guven_skoru': float(results_df['guven_skoru'].mean()),
            'minimum_guven_skoru': float(results_df['guven_skoru'].min()),
            'maksimum_guven_skoru': float(results_df['guven_skoru'].max())
        }
        
        # Özeti JSON olarak kaydet
        json_path = os.path.join(output_dir, f'batch_test_ozeti_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        
        # Görselleştirmeleri oluştur
        create_results_visualizations(results_df, output_dir)
        
        print(f"\nTest tamamlandı!")
        print(f"Toplam görüntü sayısı: {len(image_files)}")
        print(f"Başarılı tahmin sayısı: {len(results)}")
        print(f"\nSonuçlar kaydedildi:")
        print(f"- CSV: {csv_path}")
        print(f"- JSON: {json_path}")
        print(f"- Görseller: {output_dir}")
        
        return results_df, summary
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Test klasörü
    test_dir = "data/test"
    
    # Batch test'i çalıştır
    results_df, summary = batch_test(test_dir) 