import pandas as pd
import requests
import io
import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def download_football_data():
    print("--- Football-Data.co.uk Kaynağına Bağlanılıyor ---")
    
    # URL Mantığı: 
    # mmz4281 = Klasör
    # 2526 = 2025-2026 Sezonu
    # SP1 = Spain 1 (La Liga)
    # Eğer bu kod seneye çalışırsa '2627' yapmak gerekir.
    url = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        
        # Eğer dosya henüz yoksa (Sezon başı vs.) 404 dönebilir
        if response.status_code == 404:
            print("HATA: Bu sezonun dosyası henüz sitede oluşturulmamış veya URL yanlış.")
            sys.exit(1)
            
        response.raise_for_status()
        
        # Gelen veri doğrudan CSV formatında
        print("Veri indirildi, işleniyor...")
        csv_data = io.StringIO(response.content.decode('utf-8', errors='ignore'))
        
        df = pd.read_csv(csv_data)
        
        # --- VERİ TEMİZLEME ---
        # Boş satırları temizle
        df = df.dropna(how='all')
        
        # Tarih formatını düzeltelim (Site genelde dd/mm/yyyy verir)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.sort_values(by='Date')

        print(f"Başarılı! Toplam {len(df)} maç verisi çekildi.")
        
        # Örnek veri göster (Loglarda görmek için)
        print("Son 2 Maç:")
        print(df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].tail(2))
        
        return df

    except Exception as e:
        print(f"İndirme Hatası: {e}")
        sys.exit(1)

def save_and_upload(df):
    if df is None or len(df) == 0:
        print("Veri boş, Kaggle işlemi iptal.")
        return

    print("--- CSV Kaydediliyor ---")
    filename = "laliga_2025_2026_stats.csv"
    df.to_csv(filename, index=False)
    print(f"Dosya oluşturuldu: {filename}")

    print("--- Kaggle Yüklemesi Başlıyor ---")
    
    if not os.path.exists("dataset-metadata.json"):
        print("HATA: dataset-metadata.json dosyası eksik!")
        sys.exit(1)

    try:
        api = KaggleApi()
        api.authenticate()
        
        api.dataset_create_version(
            folder=".",
            version_notes=f"Auto Update (Football-Data): {pd.Timestamp.now().strftime('%Y-%m-%d')}",
            dir_mode='zip'
        )
        print("--- BAŞARILI: Kaggle Dataset Güncellendi! ---")
    except Exception as e:
        print(f"Kaggle Yükleme Hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    df = download_football_data()
    save_and_upload(df)
