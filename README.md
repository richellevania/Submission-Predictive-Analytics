# Submission-Predictive-Analytics
[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

Repositori ini berisi proyek Machine Learning yang berfokus pada prediksi gagal bayar (default) kartu kredit. Proyek ini menggunakan teknik Deep Learning (Neural Network) untuk menganalisis data historis nasabah dan memprediksi kemungkinan mereka akan mengalami gagal bayar pada bulan berikutnya.

## Gambaran Umum Proyek

Tujuan utama dari proyek ini adalah untuk membangun model prediktif yang dapat membantu lembaga keuangan dalam mengidentifikasi nasabah berisiko tinggi. Dengan prediksi yang akurat, institusi dapat mengambil keputusan yang lebih tepat terkait pemberian kredit dan mitigasi risiko finansial.

## Struktur Repositori

* `credit_card.csv`: Dataset asli yang digunakan untuk pelatihan model.
* `Prediksi_Default_Pembayaran.ipynb`: Jupyter Notebook utama yang berisi seluruh alur proyek, mulai dari eksplorasi data, persiapan data, pemodelan, hingga evaluasi.
* `model_default_credit_card.keras`: Model Neural Network yang sudah terlatih.
* `scaler.pkl`: Objek `StandardScaler` yang digunakan untuk normalisasi data.
* `pca.pkl`: Objek `PCA` yang digunakan untuk reduksi dimensi.
* `Deteksi_Outlier.png`: Visualisasi hasil deteksi *outlier*.
* `Distribusi_Fitur.png`: Visualisasi distribusi fitur-fitur dalam dataset.
* `Korelasi_Matriks.png`: Visualisasi matriks korelasi antar fitur.
* `Visualisasi_Prediksi.png`: Visualisasi contoh hasil prediksi model.
* `README.md`: Berkas ini.

## Cara Menggunakan

Untuk menjalankan atau mereplikasi proyek ini di lingkungan lokal Anda:

1.  **Kloning Repositori:**
    ```bash
    git clone [https://github.com/richellevania/Submission-Predictive-Analytics.git](https://github.com/richellevania/Submission-Predictive-Analytics.git)
    cd Submission-Predictive-Analytics
    ```
2.  **Instal Dependensi:** Pastikan Anda memiliki Python 3.x. Instal pustaka yang diperlukan menggunakan `pip`:
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn joblib imbalanced-learn
    ```
3.  **Buka Jupyter Notebook:**
    ```bash
    jupyter notebook Prediksi_Default_Pembayaran.ipynb
    ```
    Ikuti langkah-langkah di dalam notebook untuk menjalankan seluruh alur proyek.


* Richelle Vania Thionanda (MC172D5X1392)
