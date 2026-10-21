# 🌫️ Prediksi & Analisis Kualitas Udara Jakarta

Analisis pola PM2.5 Jakarta dan model prediksi harian, dibangun dari data sensor
nyata [OpenAQ](https://openaq.org) dan disajikan lewat dashboard interaktif
(Streamlit) dengan rekomendasi aktivitas berdasarkan kategori ISPU.

Jakarta rutin berada di kategori udara "Sedang" hingga "Tidak Sehat". Proyek ini
memakai data sensor lingkungan nyata untuk menjawab dua pertanyaan: **seperti
apa pola polusinya**, dan **seberapa jauh nilai besok bisa diprediksi dari
data historis**.

## Temuan utama

- **Rata-rata PM2.5 ≈ 42 µg/m³**, sekitar 2,8× pedoman WHO untuk rata-rata 24 jam (15 µg/m³).
- **Pola musiman jelas:** musim kemarau (Mei–Okt) rata-rata 46,8 µg/m³ vs musim
  hujan (Nov–Mar) 34,1 µg/m³. Bulan terendah Januari (~26), tertinggi Mei (~49).
  Konsisten dengan penjelasan fisis: tanpa hujan, polutan tidak "tercuci".
- **Antar stasiun bedanya besar:** rata-rata bisa selisih hingga ~2× (sekitar 31 vs 65 µg/m³).
- **Baseline "besok = hari ini" sangat sulit dikalahkan**, karena PM2.5 harian
  punya autokorelasi tinggi. Model terbaik hanya menang tipis di RMSE (detail di bawah).

## Dashboard

Empat tab:

| Tab | Isi |
|---|---|
| 📈 Tren Historis | Rata-rata/min/maks per stasiun, grafik time series, pola musiman bulanan |
| 🔮 Prediksi | Pilih stasiun & tanggal acuan; menampilkan nilai aktual, hasil model, dan fitur yang dipakai |
| 💡 Rekomendasi | Kategori kualitas udara + saran aktivitas dari nilai PM2.5 (otomatis dari tab Prediksi, bisa diubah dengan slider) |
| 📊 Performa Model | Perbandingan model vs baseline di test set |

Kategori mengikuti ambang PM2.5 rata-rata 24 jam (BMKG / ISPU, PermenLHK No. 14/2020):

| Kategori | PM2.5 (µg/m³) |
|---|---|
| Baik | 0 – 15,5 |
| Sedang | 15,5 – 55,4 |
| Tidak Sehat | 55,4 – 150,4 |
| Sangat Tidak Sehat | 150,4 – 250,4 |
| Berbahaya | > 250,4 |

## Cara menjalankan

Dashboard **tidak butuh API key**, karena data dan model sudah ada di repo.

```bash
git clone https://github.com/PrincessEL-max/Kualitas-Udara.git
cd Kualitas-Udara
pip install -r requirements.txt
streamlit run app.py
```

Dites di Python 3.12. `scikit-learn` dikunci di `1.9.0` karena file model
`.joblib` sensitif terhadap versi.

**Mengambil ulang data (opsional):** daftar API key gratis di
[explore.openaq.org](https://explore.openaq.org) (OpenAQ v3 mewajibkannya), lalu:

```bash
export OPENAQ_API_KEY="apikey_kamu"
python fetch_data.py
```

Untuk membuka `eda.ipynb`, install juga `matplotlib` dan `jupyter`.

## Struktur proyek

```
Kualitas-Udara/
├── app.py                 # dashboard Streamlit
├── fetch_data.py          # ambil data harian PM2.5 dari OpenAQ API v3
├── eda.ipynb              # seluruh analisis: pembersihan, EDA, fitur, modeling
├── requirements.txt
├── data/
│   ├── locations.csv                   # 23 lokasi pemantauan di bounding box Jakarta
│   ├── pm25_jakarta_daily.csv          # data harian mentah per stasiun
│   ├── anomalies_removed.csv           # 4 titik anomali yang dibuang + alasannya
│   ├── pm25_jakarta_daily_cleaned.csv  # setelah anomali dibuang
│   ├── pm25_jakarta_daily_filled.csv   # grid harian lengkap, gap ≤3 hari diinterpolasi
│   └── pm25_jakarta_features.csv       # fitur siap training
└── model/
    ├── model_pm25.joblib              # Extra Trees terlatih
    └── model_columns.joblib           # urutan kolom fitur yang diharapkan model
```

## Data & pembersihan

- **Sumber:** OpenAQ v3, rata-rata harian PM2.5, Jan 2023 – Agu 2026. Dari 23 lokasi
  di bounding box Jakarta, 10 punya sensor PM2.5 (campuran sensor low-cost dan
  stasiun referensi). Totalnya ±4.670 baris harian mentah.
- **Anomali:** 4 nilai ekstrem di Jakarta Central (246–545 µg/m³) dibuang karena
  stasiun tetangga di tanggal yang sama menunjukkan nilai normal (41–75 µg/m³).
  Polusi udara itu fenomena regional, jadi lonjakan tunggal kemungkinan besar glitch
  sensor. Yang dibuang hanya titik datanya, bukan seluruh stasiunnya.
- **Missing value tersembunyi:** hari tanpa laporan sensor tidak muncul sebagai
  `NaN`, barisnya memang tidak ada. Setelah di-reindex ke grid harian penuh per
  stasiun (5.625 hari-stasiun), ternyata **±23% kosong**.
- **Interpolasi:** hanya gap pendek (≤3 hari) yang diisi. Gap panjang (puluhan hari,
  kemungkinan sensor offline) sengaja dibiarkan kosong agar tidak mengarang data
  berbulan-bulan. Panjang gap dihitung eksplisit, karena `interpolate(limit=3,
  limit_direction='both')` bawaan pandas bisa menembus gap sampai 6 hari.

## Modeling

**Tugas:** memprediksi PM2.5 sebuah hari dari data hingga hari sebelumnya (horizon 1 hari).

**Fitur:** `lag_1`, `lag_2`, `lag_3`, `lag_7`, `rolling_mean_3`, `rolling_mean_7`,
bulan, hari dalam minggu, `is_weekend`, `is_musim_kemarau` (Mei–Okt), dan
one-hot stasiun. Jumlah baris siap training: 3.801.

**Validasi:** split berdasarkan waktu (80/20, cutoff 10 Jul 2025; 3.420 baris train,
381 baris test), bukan acak, supaya data masa depan tidak bocor ke training.

**Target:** selisih (delta) dari nilai kemarin, bukan nilai absolut. Prediksi akhir =
`lag_1` + delta. Perubahan ini yang membuat model berbasis pohon bisa mendekati baseline.

| Model | MAE | RMSE | vs baseline (MAE) |
|---|---|---|---|
| **Baseline (persistence)** | **6,82** | 9,86 | – |
| Random Forest | 7,17 | 9,52 | −5,2% |
| **Extra Trees** (dipilih) | 7,33 | **9,46** | −7,5% |
| KNN (k=15) | 7,66 | 10,08 | −12,4% |
| HistGradientBoosting | 7,86 | 10,27 | −15,4% |
| Gradient Boosting | 8,41 | 10,81 | −23,4% |
| Ridge | 9,44 | 11,42 | −38,5% |
| Linear Regression | 9,45 | 11,43 | −38,6% |

*(Semua dievaluasi di test set; MAE & RMSE dalam µg/m³, makin kecil makin baik.)*

**Cara membaca hasil ini, apa adanya:**
- Tidak ada model yang mengalahkan baseline di MAE. Extra Trees hanya unggul di
  RMSE, artinya lebih jarang meleset ekstrem.
- Model linear gagal karena multikolinearitas: semua fitur lag/rolling diturunkan
  dari deret yang sama.
- Boosting lebih buruk dari Random Forest, tapi parameternya belum di-tuning.
- **Data cuaca (Open-Meteo) sudah dicoba dan tidak membantu.** Korelasi dengan
  PM2.5 lemah (r ≈ 0,18–0,28) dibanding sinyal autokorelasi, jadi tidak dipakai di model final.

## Keterbatasan

- **Model final dilatih ulang pada seluruh data**, termasuk test set. Angka
  performa di atas berasal dari model sebelum dilatih ulang. Perbandingan "prediksi
  vs nilai asli" di dashboard bukan evaluasi out-of-sample.
- **Dashboard memakai data historis**, bukan data real-time. Tanggal terbaru yang
  tersedia adalah akhir data tiap stasiun (beberapa stasiun berhenti melapor di 2024–2025).
- **BMKG 1 tidak didukung** karena datanya terlalu sedikit untuk dilatih.
- **Kualitas sensor beragam** (low-cost vs referensi), dan model memakai satu model
  gabungan dengan one-hot stasiun, jadi dinamika unik tiap stasiun belum tertangkap penuh.
- **Horizon hanya 1 hari**, di mana baseline persistence secara alami sudah kuat.

## Pengembangan lanjutan

- Horizon lebih panjang (3–7 hari), tempat model punya ruang lebih besar untuk unggul.
- Data **ramalan** cuaca asli, bukan cuaca aktual historis, agar fitur cuaca realistis untuk prediksi nyata.
- Variabel fisis tambahan: tinggi *boundary layer* atmosfer dan data lalu lintas.
- Tuning hyperparameter dan model per stasiun.

## Sumber

- Data kualitas udara: [OpenAQ](https://openaq.org)
- Data cuaca (eksperimen): [Open-Meteo](https://open-meteo.com)
- Ambang kategori: BMKG / ISPU, PermenLHK No. 14/2020
- Pedoman WHO: rata-rata 24 jam PM2.5 = 15 µg/m³
