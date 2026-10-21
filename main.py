"""Proyek 1 — Analisis Data Penjualan dan Prediksi Sederhana

Data contoh dibuat untuk pembelajaran. Bukan data transaksi bisnis nyata.
Jalankan dari folder proyek: python main.py
"""
from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "penjualan.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 10


def rupiah(value):
    return "Rp" + f"{value:,.0f}".replace(",", ".")


def load_and_clean_data():
    """Memuat data, memperbaiki tipe data, dan menangani nilai kosong/duplikat."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    original_rows = len(df)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    numeric_cols = ["jumlah", "harga_satuan", "diskon_persen", "pendapatan"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Isian kategori kosong ditandai sebagai 'Tidak diketahui'.
    for col in ["produk", "wilayah", "kanal"]:
        df[col] = df[col].fillna("Tidak diketahui").astype(str).str.strip()
        df.loc[df[col].eq(""), col] = "Tidak diketahui"

    # Baris tanpa tanggal atau nilai penting tidak digunakan dalam analisis.
    before_drop = len(df)
    df = df.dropna(subset=["tanggal", "jumlah", "harga_satuan", "pendapatan"])
    dropped_missing = before_drop - len(df)

    # Hitung ulang pendapatan dari komponen transaksi agar konsisten.
    df["jumlah"] = df["jumlah"].clip(lower=0)
    df["harga_satuan"] = df["harga_satuan"].clip(lower=0)
    df["diskon_persen"] = df["diskon_persen"].fillna(0).clip(0, 100)
    df["pendapatan"] = (df["jumlah"] * df["harga_satuan"] * (1 - df["diskon_persen"] / 100)).round(0)
    df["bulan"] = df["tanggal"].dt.to_period("M").astype(str)
    df["hari_dalam_minggu"] = df["tanggal"].dt.day_name()

    clean_path = OUTPUT_DIR / "data_bersih.csv"
    df.to_csv(clean_path, index=False)
    print("\n[1] PEMBERSIHAN DATA")
    print(f"Baris awal             : {original_rows}")
    print(f"Duplikat yang dihapus  : {duplicate_count}")
    print(f"Baris kosong penting   : {dropped_missing}")
    print(f"Baris siap dianalisis  : {len(df)}")
    print(f"Data bersih disimpan   : {clean_path}")
    return df


def descriptive_analysis(df):
    print("\n[2] ANALISIS DESKRIPTIF")
    total_revenue = df["pendapatan"].sum()
    total_transactions = len(df)
    average_revenue = df["pendapatan"].mean()
    total_units = df["jumlah"].sum()
    print(f"Total pendapatan       : {rupiah(total_revenue)}")
    print(f"Jumlah transaksi       : {total_transactions:,}")
    print(f"Rata-rata transaksi    : {rupiah(average_revenue)}")
    print(f"Total unit terjual     : {total_units:,.0f}")

    by_product = df.groupby("produk", as_index=False).agg(
        pendapatan=("pendapatan", "sum"), unit_terjual=("jumlah", "sum"), transaksi=("id_transaksi", "count")
    ).sort_values("pendapatan", ascending=False)
    by_region = df.groupby("wilayah", as_index=False)["pendapatan"].sum().sort_values("pendapatan", ascending=False)
    by_month = df.groupby("bulan", as_index=False)["pendapatan"].sum().sort_values("bulan")

    print("\nPendapatan berdasarkan produk:")
    for _, row in by_product.iterrows():
        print(f"  {row['produk']:<18} {rupiah(row['pendapatan']):>16} | unit {row['unit_terjual']:.0f}")
    print("\nWilayah dengan pendapatan tertinggi:")
    if not by_region.empty:
        print(f"  {by_region.iloc[0]['wilayah']}: {rupiah(by_region.iloc[0]['pendapatan'])}")

    by_product.to_csv(OUTPUT_DIR / "ringkasan_produk.csv", index=False)
    by_region.to_csv(OUTPUT_DIR / "ringkasan_wilayah.csv", index=False)
    by_month.to_csv(OUTPUT_DIR / "ringkasan_bulanan.csv", index=False)
    return by_product, by_region, by_month


def diagnostic_analysis(df):
    print("\n[3] ANALISIS DIAGNOSTIK — PEMERIKSAAN TRANSAKSI" )
    q1 = df["pendapatan"].quantile(0.25)
    q3 = df["pendapatan"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    unusual = df[(df["pendapatan"] < lower) | (df["pendapatan"] > upper)].copy()
    unusual.to_csv(OUTPUT_DIR / "transaksi_perlu_diperiksa.csv", index=False)
    print(f"Transaksi dengan pendapatan tidak biasa (metode IQR): {len(unusual)}")
    print("Catatan: nilai tidak biasa bukan otomatis kesalahan; perlu diperiksa oleh analis.")
    return unusual


def create_visualizations(df, by_product, by_region, by_month, unusual):
    # 1. Pendapatan per produk
    fig, ax = plt.subplots()
    data = by_product.sort_values("pendapatan")
    ax.barh(data["produk"], data["pendapatan"])
    ax.set_title("Total Pendapatan per Produk")
    ax.set_xlabel("Pendapatan (Rp)")
    ax.set_ylabel("Produk")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_pendapatan_produk.png", dpi=160)
    plt.show()

    # 2. Tren bulanan
    fig, ax = plt.subplots()
    ax.plot(by_month["bulan"], by_month["pendapatan"], marker="o", linewidth=2)
    ax.set_title("Tren Pendapatan Bulanan")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Pendapatan (Rp)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_tren_bulanan.png", dpi=160)
    plt.show()

    # 3. Pendapatan per wilayah
    fig, ax = plt.subplots()
    data = by_region.sort_values("pendapatan", ascending=False)
    ax.bar(data["wilayah"], data["pendapatan"])
    ax.set_title("Perbandingan Pendapatan per Wilayah")
    ax.set_xlabel("Wilayah")
    ax.set_ylabel("Pendapatan (Rp)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_pendapatan_wilayah.png", dpi=160)
    plt.show()

    # 4. Distribusi transaksi dan titik outlier
    fig, ax = plt.subplots()
    ax.hist(df["pendapatan"], bins=25, edgecolor="black", alpha=0.8)
    ax.set_title(f"Distribusi Pendapatan Transaksi ({len(unusual)} perlu diperiksa)")
    ax.set_xlabel("Pendapatan transaksi (Rp)")
    ax.set_ylabel("Frekuensi")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_distribusi_pendapatan.png", dpi=160)
    plt.show()


def predictive_analysis(df):
    print("\n[4] MODEL PREDIKSI PENDAPATAN TRANSAKSI")
    # Model demonstrasi: memperkirakan pendapatan dari karakteristik transaksi.
    features = ["produk", "wilayah", "kanal", "jumlah", "harga_satuan", "diskon_persen"]
    target = "pendapatan"
    model_df = df.dropna(subset=features + [target]).copy()
    X = model_df[features]
    y = model_df[target]
    categorical = ["produk", "wilayah", "kanal"]
    numeric = ["jumlah", "harga_satuan", "diskon_persen"]

    preprocessor = ColumnTransformer([
        ("kategori", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("angka", "passthrough", numeric),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    print(f"Data latih             : {len(X_train)} transaksi")
    print(f"Data uji               : {len(X_test)} transaksi")
    print(f"MAE                    : {rupiah(mae)}")
    print(f"RMSE                   : {rupiah(rmse)}")
    print(f"R²                     : {r2:.3f}")
    print("Interpretasi: MAE adalah rata-rata besar selisih prediksi dengan nilai aktual.")
    print("Penting: karena pendapatan dihitung dari jumlah, harga, dan diskon, model ini demonstrasi alur ML, bukan peramalan pasar.")

    result = X_test.copy()
    result["pendapatan_aktual"] = y_test
    result["pendapatan_prediksi"] = predictions.round(0)
    result["selisih_absolut"] = (result["pendapatan_aktual"] - result["pendapatan_prediksi"]).abs().round(0)
    result.to_csv(OUTPUT_DIR / "hasil_prediksi.csv", index=False)

    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.65)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title("Pendapatan Aktual vs Prediksi")
    ax.set_xlabel("Aktual (Rp)")
    ax.set_ylabel("Prediksi (Rp)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_aktual_vs_prediksi.png", dpi=160)
    plt.show()

    with open(OUTPUT_DIR / "metrik_model.txt", "w", encoding="utf-8") as f:
        f.write("Evaluasi model Random Forest Regressor\n")
        f.write(f"Data latih: {len(X_train)}\nData uji: {len(X_test)}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.4f}\n")


def write_summary(df, by_product, by_region, by_month, unusual):
    best_product = by_product.iloc[0]
    best_region = by_region.iloc[0]
    highest_month = by_month.loc[by_month["pendapatan"].idxmax()]
    report = f"""LAPORAN RINGKAS ANALISIS DATA PENJUALAN
======================================
Dataset: data/penjualan.csv (data sintetis untuk pembelajaran)
Jumlah transaksi bersih: {len(df)}
Periode: {df['tanggal'].min().date()} sampai {df['tanggal'].max().date()}
Total pendapatan: {rupiah(df['pendapatan'].sum())}
Total unit terjual: {df['jumlah'].sum():.0f}
Produk dengan pendapatan tertinggi: {best_product['produk']} ({rupiah(best_product['pendapatan'])})
Wilayah dengan pendapatan tertinggi: {best_region['wilayah']} ({rupiah(best_region['pendapatan'])})
Bulan dengan pendapatan tertinggi: {highest_month['bulan']} ({rupiah(highest_month['pendapatan'])})
Transaksi dengan nilai tidak biasa untuk pemeriksaan: {len(unusual)}

CATATAN
- Data merupakan data sintetis, bukan transaksi bisnis nyata.
- Nilai tidak biasa tidak otomatis salah; pemeriksaan manual tetap diperlukan.
- Model prediksi merupakan demonstrasi alur machine learning.
- Output grafik, tabel ringkasan, data bersih, dan evaluasi model ada di folder output/.
"""
    (OUTPUT_DIR / "laporan_ringkas.txt").write_text(report, encoding="utf-8")
    print("\n[5] LAPORAN RINGKAS")
    print(report)


def main():
    print("=" * 58)
    print("PROYEK 1 | ANALISIS DATA PENJUALAN DAN AI")
    print("Pembersihan • Deskriptif • Diagnostik • Prediksi")
    print("=" * 58)
    print("Dataset yang digunakan adalah data sintetis untuk pembelajaran.\n")
    df = load_and_clean_data()
    by_product, by_region, by_month = descriptive_analysis(df)
    unusual = diagnostic_analysis(df)
    create_visualizations(df, by_product, by_region, by_month, unusual)
    predictive_analysis(df)
    write_summary(df, by_product, by_region, by_month, unusual)
    print(f"Selesai. Periksa semua hasil di folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
