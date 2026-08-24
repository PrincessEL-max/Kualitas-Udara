# Prediksi & Analisis Kualitas Udara Jakarta

Proyek portofolio data science: analisis pola dan prediksi AQI (PM2.5) Jakarta
menggunakan data real dari [OpenAQ](https://openaq.org), lalu disajikan lewat
dashboard interaktif dengan rekomendasi actionable.

## Kenapa proyek ini

Jakarta rutin masuk kategori kualitas udara "Sedang" sampai "Tidak Sehat" —
masalah kesehatan publik yang nyata dan berlangsung terus, bukan cuma latihan
data science di atas kertas. Proyek ini juga jadi jembatan dari background
Fisika Instrumentasi ke data science: intinya sama-sama kerja dengan data
sensor lingkungan, cuma di sini fokusnya ke analisis & prediksi datanya.

## Setup

1. Daftar API key gratis di https://explore.openaq.org (OpenAQ v3 mewajibkan
   API key di semua request).
2. Buat virtual environment (opsional tapi disarankan):
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set API key sebagai environment variable:
   ```
   export OPENAQ_API_KEY="apikey_kamu"
   ```
5. Jalankan pengambilan data:
   ```
   python fetch_data.py
   ```
   Hasilnya: `data/locations.csv` (daftar stasiun di Jakarta) dan
   `data/pm25_jakarta_daily.csv` (data harian PM2.5 per stasiun).

> **Catatan:** `fetch_data.py` disusun berdasarkan dokumentasi resmi OpenAQ
> API v3 (docs.openaq.org) per Agustus 2026, tapi belum pernah dites live —
> sandbox yang dipakai untuk nulis ini gak bisa akses `api.openaq.org`
> langsung. Kalau pas dijalankan ada error skema respons (field yang gak
> ketemu, dsb), kirim aja pesan errornya, gampang disesuaikan.

## Struktur proyek

```
aqi-jakarta/
├── fetch_data.py       # ambil data dari OpenAQ API
├── requirements.txt
├── data/                # hasil fetch (locations.csv, pm25_jakarta_daily.csv)
└── README.md
```

## Roadmap

- [x] Setup project & script pengambilan data
- [ ] Jalankan `fetch_data.py`, validasi data yang masuk (jumlah stasiun,
      rentang tanggal, missing values)
- [ ] EDA: pola harian/mingguan/musiman, perbandingan antar stasiun,
      korelasi dengan variabel cuaca
- [ ] Feature engineering (lag features, hari libur, musim kemarau/hujan)
- [ ] Model prediksi AQI beberapa hari ke depan (baseline → model lebih kuat)
- [ ] Dashboard interaktif (Streamlit) dengan rekomendasi actionable
      (jam/hari aman untuk aktivitas luar ruangan)
- [ ] Tulis studi kasus singkat untuk ditaruh di portofolio
