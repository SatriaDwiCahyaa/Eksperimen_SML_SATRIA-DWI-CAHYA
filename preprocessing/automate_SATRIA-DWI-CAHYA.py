"""
automate_SATRIA-DWI-CAHYA.py
Script otomatisasi preprocessing dataset Credit Card Fraud Detection.
Mengkonversi langkah preprocessing dari notebook eksperimen ke fungsi-fungsi
yang dapat dijalankan secara otomatis.

Tahapan preprocessing (sama dengan notebook):
1. Memuat dataset
2. Menghapus data duplikat
3. Standarisasi fitur Amount dan Time
4. Memisahkan fitur dan label
5. Membagi data train/test dengan stratifikasi
6. Penanganan ketidakseimbangan kelas dengan SMOTE
7. Menyimpan hasil preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

# Menentukan direktori dasar secara dinamis (lokasi file script ini)
DIREKTORI_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIREKTORI_ROOT = os.path.dirname(DIREKTORI_SCRIPT)


def muat_dataset(jalur_file: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    df = pd.read_csv(jalur_file)
    print(f"[INFO] Dataset dimuat: {df.shape[0]:,} baris, {df.shape[1]} kolom")
    return df


def hapus_duplikat(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat dari dataset."""
    jumlah_sebelum = len(df)
    df = df.drop_duplicates()
    jumlah_dihapus = jumlah_sebelum - len(df)
    print(f"[INFO] Duplikat dihapus: {jumlah_dihapus:,} baris "
          f"(sisa: {len(df):,} baris)")
    return df


def standarisasi_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menstandarisasi fitur 'Amount' dan 'Time'.
    Fitur V1-V28 sudah hasil PCA sehingga sudah terstandarisasi.
    """
    penskalaan_amount = StandardScaler()
    penskalaan_time = StandardScaler()

    df['Amount_Scaled'] = penskalaan_amount.fit_transform(df[['Amount']])
    df['Time_Scaled'] = penskalaan_time.fit_transform(df[['Time']])

    # Hapus kolom asli
    df = df.drop(columns=['Amount', 'Time'])

    print("[INFO] Fitur 'Amount' dan 'Time' berhasil distandarisasi")
    return df


def pisahkan_fitur_label(df: pd.DataFrame):
    """Memisahkan fitur (X) dan label target (y)."""
    X = df.drop(columns=['Class'])
    y = df['Class']
    print(f"[INFO] Fitur: {X.shape}, Label: {y.shape}")
    return X, y


def bagi_data(X, y, ukuran_test=0.2, random_state=42):
    """
    Membagi data menjadi train dan test dengan stratifikasi
    untuk menjaga proporsi kelas pada kedua subset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=ukuran_test,
        random_state=random_state,
        stratify=y
    )
    print(f"[INFO] Data train: {X_train.shape[0]:,} baris, "
          f"Data test: {X_test.shape[0]:,} baris")
    return X_train, X_test, y_train, y_test


def tangani_ketidakseimbangan(X_train, y_train, random_state=42):
    """
    Menerapkan SMOTE pada data training untuk menangani
    ketidakseimbangan kelas yang ekstrem.
    SMOTE HANYA diterapkan pada data training, TIDAK pada data test.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] SMOTE diterapkan: {X_train.shape[0]:,} -> "
          f"{X_resampled.shape[0]:,} baris")
    return X_resampled, y_resampled


def simpan_hasil(X_train, X_test, y_train, y_test, jalur_output: str):
    """Menyimpan data yang sudah dipreproses ke file CSV."""
    os.makedirs(jalur_output, exist_ok=True)

    X_train.to_csv(os.path.join(jalur_output, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(jalur_output, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(jalur_output, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(jalur_output, 'y_test.csv'), index=False)

    print(f"[INFO] Data berhasil disimpan ke '{jalur_output}'")
    print(f"       - X_train.csv : {X_train.shape}")
    print(f"       - X_test.csv  : {X_test.shape}")
    print(f"       - y_train.csv : {y_train.shape}")
    print(f"       - y_test.csv  : {y_test.shape}")


def jalankan_preprocessing():
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing.
    Mengembalikan data yang siap dilatih (X_train, X_test, y_train, y_test).
    """
    print("=" * 60)
    print("MEMULAI PIPELINE PREPROCESSING OTOMATIS")
    print("Dataset: Credit Card Fraud Detection")
    print("=" * 60)

    # Menentukan jalur file secara absolut berdasarkan lokasi script
    jalur_dataset = os.path.join(DIREKTORI_ROOT, 'creditcard_raw', 'creditcard.csv')
    jalur_output = os.path.join(DIREKTORI_SCRIPT, 'creditcard_preprocessing')

    # 1. Muat dataset
    df = muat_dataset(jalur_dataset)

    # 2. Hapus duplikat
    df = hapus_duplikat(df)

    # 3. Standarisasi fitur Amount dan Time
    df = standarisasi_fitur(df)

    # 4. Pisahkan fitur dan label
    X, y = pisahkan_fitur_label(df)

    # 5. Bagi data train/test dengan stratifikasi
    X_train, X_test, y_train, y_test = bagi_data(X, y)

    # 6. Tangani ketidakseimbangan kelas dengan SMOTE
    X_train, y_train = tangani_ketidakseimbangan(X_train, y_train)

    # 7. Simpan hasil preprocessing
    simpan_hasil(X_train, X_test, y_train, y_test, jalur_output)

    print("=" * 60)
    print("\u2705 PREPROCESSING SELESAI \u2014 Data siap digunakan untuk pelatihan model")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = jalankan_preprocessing()
