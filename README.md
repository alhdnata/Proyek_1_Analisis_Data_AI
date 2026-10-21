# Proyek 1 — Analisis Data dan Artificial Intelligence

## 📌 Deskripsi Proyek

Proyek ini merupakan implementasi sederhana dari materi seminar **Make Sense of Data with Analysis and AI**. Tujuannya adalah menunjukkan bagaimana data dapat diolah menjadi informasi yang berguna melalui proses analisis data, visualisasi, dan penerapan Machine Learning.

Program menggunakan Python untuk membersihkan data, menganalisis pendapatan produk, melihat tren penjualan, serta membangun model prediksi menggunakan algoritma Random Forest.

Proyek ini dibuat sebagai sarana pembelajaran dan demonstrasi penerapan analisis data dan AI dalam studi kasus penjualan.

## 🎯 Tujuan

- Memahami proses pengolahan dan pembersihan data.
- Mengidentifikasi pola dan tren dari data penjualan.
- Menyajikan informasi melalui visualisasi data.
- Menerapkan Machine Learning untuk memprediksi pendapatan.
- Mengevaluasi kinerja model menggunakan metrik regresi.
- Memahami bagaimana data dapat membantu proses pengambilan keputusan.

## 🛠️ Teknologi yang Digunakan

- **Python** — bahasa pemrograman utama.
- **Pandas** — pengolahan dan analisis data.
- **Matplotlib** — pembuatan grafik dan visualisasi.
- **Scikit-learn** — pembuatan dan evaluasi model Machine Learning.

## 📂 Struktur Proyek

```text
Proyek_1_Analisis_Data_AI/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── data_penjualan.csv
└── output/
    ├── analisis_produk.png
    ├── tren_bulanan.png
    ├── prediksi_vs_aktual.png
    └── laporan_analisis.csv
```

*Catatan: nama atau isi file dapat berbeda mengikuti versi kode yang digunakan. Folder output berisi hasil yang dihasilkan program.*

## 📊 Fitur Utama

### 1. Pembersihan Data

Program memeriksa data sebelum analisis dilakukan, termasuk penanganan data yang tidak lengkap atau tidak sesuai kebutuhan. Langkah ini penting agar hasil analisis lebih dapat dipercaya.

### 2. Analisis Pendapatan Produk

Data penjualan dikelompokkan untuk membandingkan pendapatan dan jumlah unit yang terjual. Hasilnya membantu mengidentifikasi produk dengan kontribusi pendapatan tertinggi.

### 3. Analisis Tren Penjualan

Visualisasi tren bulanan digunakan untuk melihat perubahan pendapatan dari waktu ke waktu dan membantu memahami pola penjualan pada dataset.

### 4. Prediksi Menggunakan Machine Learning

Algoritma Random Forest Regressor digunakan untuk mempelajari hubungan antara fitur data dan nilai pendapatan. Model kemudian menghasilkan prediksi yang dibandingkan dengan nilai aktual.

### 5. Evaluasi Model

Kinerja model dievaluasi menggunakan beberapa metrik:

- **MAE (Mean Absolute Error):** rata-rata selisih absolut antara prediksi dan nilai aktual.
- **RMSE (Root Mean Squared Error):** ukuran kesalahan prediksi yang memberikan penalti lebih besar pada kesalahan besar.
- **R² (Coefficient of Determination):** ukuran seberapa baik model menjelaskan variasi nilai target pada data evaluasi.

Nilai metrik perlu ditafsirkan berdasarkan karakteristik dataset dan metode pembagian data yang digunakan.

## ⚙️ Persyaratan

Sebelum menjalankan program, pastikan perangkat sudah memiliki:

- Python versi 3.10 atau lebih baru.
- pip untuk instalasi library.
- Visual Studio Code atau editor Python lainnya.

## 🚀 Cara Menjalankan

**1. Clone repositori**

```bash
git clone URL_REPOSITORY_KAMU
```

Ganti `URL_REPOSITORY_KAMU` dengan URL repositori GitHub milikmu.

**2. Masuk ke folder proyek**

```bash
cd Proyek_1_Analisis_Data_AI
```

**3. Instal dependensi**

```bash
python -m pip install -r requirements.txt
```

**4. Jalankan program**

```bash
python main.py
```

**5. Lihat hasil analisis**

Setelah program selesai, periksa folder `output` untuk melihat grafik dan laporan yang dihasilkan. Nama file yang tersedia bergantung pada implementasi program.

## 📈 Hasil yang Diharapkan

Program menghasilkan visualisasi untuk membantu memahami:

- Perbandingan pendapatan antarproduk.
- Perubahan pendapatan dari bulan ke bulan.
- Perbandingan nilai prediksi dan nilai aktual.
- Ringkasan hasil analisis dalam bentuk file laporan.

Hasil tersebut dapat digunakan untuk mempelajari hubungan antara analisis data, visualisasi, dan Machine Learning dalam mendukung pengambilan keputusan.

## 🔬 Dataset dan Batasan

Dataset yang disertakan merupakan **data sintetis untuk demonstrasi**, bukan data transaksi bisnis nyata. Oleh karena itu, hasil analisis dan prediksi hanya digunakan untuk tujuan pembelajaran dan tidak dapat langsung dijadikan dasar keputusan bisnis nyata.

Kinerja model juga bergantung pada kualitas data, fitur yang digunakan, dan cara evaluasi dilakukan. Model Machine Learning tidak menjamin prediksi yang selalu benar.

## 📚 Kaitan dengan Materi Seminar

Proyek ini mengimplementasikan konsep analisis data dan AI melalui beberapa tahapan, mulai dari persiapan data, analisis, visualisasi, hingga pemodelan prediktif.

Alur tersebut berkaitan dengan proses analisis data seperti **CRISP-DM**, khususnya pada tahap persiapan data, pemodelan, dan evaluasi. Proyek ini merupakan implementasi pembelajaran dan bukan reproduksi penuh seluruh materi seminar.

## 👨‍💻 Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan:

- Menggunakan dataset penjualan nyata.
- Menambahkan filter berdasarkan periode atau kategori produk.
- Membandingkan beberapa algoritma Machine Learning.
- Menambahkan validasi silang dan analisis kesalahan model.
- Membuat dashboard interaktif untuk menampilkan hasil analisis.

## 📄 Lisensi

Proyek ini dibuat untuk tujuan pembelajaran. Jika ingin digunakan atau dikembangkan kembali oleh pihak lain, silakan tambahkan ketentuan lisensi yang sesuai pada repositori.

---

**Kesimpulan:** Proyek ini menunjukkan alur dasar pengolahan data, visualisasi, dan prediksi menggunakan Machine Learning untuk membantu memahami informasi yang terkandung dalam data penjualan.
