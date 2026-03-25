# Traffic Volume Prediction - Weighted KNN

Prediksi Volume Lalu Lintas Interstate 94 (Minneapolis-St Paul, MN) menggunakan metode Weighted K-Nearest Neighbors (Weighted KNN) Regressor.  
Project ini memanfaatkan data historis volume kendaraan per jam beserta fitur cuaca dan waktu untuk membangun model prediksi yang explainable dan mudah direproduksi.

## Daftar Isi

- [Deskripsi Proyek](#deskripsi-proyek)
- [Dataset](#dataset)
- [Langkah Analisis](#langkah-analisis)
- [Model & Hasil Evaluasi](#model--hasil-evaluasi)
- [Keterbatasan](#keterbatasan)
- [Cara Menjalankan Notebook](#cara-menjalankan-notebook)
- [Referensi](#referensi)

---

## Deskripsi Proyek

Tujuan utama proyek ini adalah membangun model prediksi untuk memprakirakan volume lalu lintas berdasarkan waktu (jam, hari, bulan), cuaca (rain, snow, clouds, dsb), dan hari libur.  
Model yang dipilih adalah Weighted KNN, karena: 
- Sederhana, intuitif, dan mudah dieksperimenkan
- Cocok untuk studi kasus prediksi berbasis kemiripan pada data numerik + kategorikal (dengan encoding)
- Tidak memerlukan proses training “sebenarnya” (Lazy Learner)

## Dataset

- **Sumber:** [Metro Interstate Traffic Volume, UCI ML Repo (ID:492)](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
- **Ukuran:** 48.204 baris (2012-2018), fitur utama:
    - holiday (hari libur)
    - temp, rain_1h, snow_1h, clouds_all
    - weather_main, weather_description
    - date_time (tanggal + jam, CST)
    - traffic_volume (target: jumlah kendaraan per jam)

## Langkah Analisis

1. **Data Preparation:**
   - Import langsung dari UCI ML Repo (otomatis via ucimlrepo)
   - Data cleaning: imputer nilai anomali (contoh suhu = 0 K diganti mean) dan ekstrak fitur waktu (hour, day_of_week, month)
   - Transformasi kategori ke one-hot encoding
   - Standarisasi fitur (mean=0, std=1) untuk optimasi KNN

2. **Exploratory Data Analysis (EDA):**
   - Analisis distribusi traffic, outlier, pola musiman (jam/hari/bulan), efek cuaca
   - Insight: Waktu (jam & hari) adalah prediktor terkuat, cuaca memberikan pengaruh tambahan

3. **Modeling (Weighted KNN Regression):**
   - Jarak: Euclidean Distance di hyperdimensi numerik-encoded
   - K: Tuning secara manual pada subset data, dipilih K=3 (hasil MAE terendah ~496 kendaraan)
   - Penimbangan tetangga: Inverse distance weighting  
  
4. **Evaluasi dan Validasi:**
   - MAE: ±500 kendaraan/jam
   - RMSE: ±750 kendaraan/jam
   - 5-Fold Cross Validation untuk memastikan kestabilan hasil
   - Visualisasi error dan trend prediksi vs real

5. **Testing pada Data Baru:**
   - Uji prediksi dengan data luar (misal: data Oktober 2018) menggunakan pipeline yang sama (preprocessing, encoding, scaling)

## Model & Hasil Evaluasi

- **Mean Absolute Error (MAE):** ~500 unit kendaraan (sekitar 15% dari traffic puncak)
- **Root Mean Squared Error (RMSE):** ~750 kendaraan  
- **Validasi K-Fold:** MAE stabil di range 450-600, menandakan model konsisten
- **Analisis visual:** Residual cenderung normal (tidak ada bias besar), prediksi mengikuti fluktuasi aktual dengan baik, kecuali saat lonjakan ekstrem

### Contoh Output Prediksi:
```
Baris 1 [2018-10-24 20:00:00] | Cuaca: Haze | Libur: nan
Prediksi Volume: 1850.03 kendaraan
Baris ... dst ...
```

## Keterbatasan

- **Lazy Learner:** Waktu prediksi relatif lambat jika dataset besar (solusi: KD-Tree, ANN, dsb)
- **Tidak sensitif terhadap event luar biasa** (misal kecelakaan mendadak/ acara khusus yang tidak terdata)
- **Tidak memanfaatkan urutan waktu (time-series split):** fold validasi acak

## Cara Menjalankan Notebook

1. **Buka notebook [ML_Regression_WeightedKNN.ipynb](./ML_Regression_WeightedKNN.ipynb) di Colab** (disarankan)
2. **Install dependency**:
    ```python
    !pip install ucimlrepo pandas numpy matplotlib seaborn
    ```
3. **Load dataset dan jalankan setiap sel secara berurutan**
4. **Testing data baru:** Siapkan file CSV eksternal data baru (header harus sama dengan data asli)

## Referensi

- [Dataset UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
- [Scikit-learn KNN Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [Inverse Distance Weighting](https://en.wikipedia.org/wiki/Inverse_distance_weighting)

---

Silakan modifikasi bagian deskripsi, batasan, atau langkah analisis agar sesuai jika ada hal khusus yang ingin kamu tambahkan!  
Jika kamu butuh README dalam bahasa Inggris atau penyesuaian lain, silakan kabari kembali.