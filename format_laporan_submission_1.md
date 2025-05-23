# Laporan Proyek Machine Learning - MC172D5X1392 Richelle Vania Thionanda

## Domain Proyek

**1. Latar Belakang**

Penggunaan kartu kredit telah menjadi bagian penting dari sistem keuangan modern. Namun, banyak lembaga keuangan menghadapi risiko kegagalan pembayaran (default) oleh nasabah. Dengan meningkatnya jumlah transaksi kredit dan kompleksitas penilaian kelayakan kredit, pendekatan manual tidak lagi cukup. Oleh karena itu, machine learning menjadi solusi yang potensial untuk membantu memprediksi kemungkinan gagal bayar nasabah dengan lebih akurat dan efisien.

**2. Urgensi dan Alasan Penyelesaian Masalah**

Prediksi ini penting karena dapat membantu lembaga keuangan:
- Mengurangi risiko kerugian finansial.
- Meningkatkan kualitas pengambilan keputusan pemberian kredit.
- Meningkatkan efisiensi dan akurasi dalam penilaian risiko kredit.

**Referensi Tambahan**
The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients (https://www.sciencedirect.com/science/article/abs/pii/S0957417407006719?via%3Dihub)

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana memprediksi apakah seorang nasabah akan mengalami gagal bayar pada bulan berikutnya berdasarkan data historis?
- Bagaimana mengatasi ketidakseimbangan kelas dalam data default pembayaran?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan model klasifikasi untuk mendeteksi potensi gagal bayar.
- Meningkatkan performa prediksi dengan preprocessing dan balancing data.


### Solution statements
- Membangun baseline model menggunakan Neural Network.
- Melakukan peningkatan kinerja model dengan hyperparameter tuning, dropout, dan batch normalization.
- Menangani imbalance data menggunakan SMOTE.
- Menggunakan metrik F1 Score untuk mengevaluasi performa model pada data tidak seimbang.

## Data Understanding
Dataset yang digunakan berasal dari [Default of Credit Card Clients Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset). Dataset tersebut kemudian diunggah ulang ke GitHub dan digunakan melalui tautan berikut: [GitHub Dataset Link](https://github.com/richellevania/Submission-Predictive-Analytics/blob/main/credit_card.csv.)  

### Variabel-variabel pada dataset adalah sebagai berikut:
Variabel-variabel dalam dataset:
- ID: ID unik setiap klien.
- LIMIT_BAL: Jumlah kredit yang diberikan dalam NT dollar (termasuk kredit individu dan keluarga/tambahan).
- SEX: Jenis kelamin (1=laki-laki, 2=perempuan).
- EDUCATION: Tingkat pendidikan (1=pascasarjana, 2=universitas, 3=SMA, 4=lainnya, 5=tidak diketahui, 6=tidak diketahui).
- MARRIAGE: Status perkawinan (1=menikah, 2=lajang, 3=lainnya).
- AGE: Usia dalam tahun.
- PAY_0: Status pembayaran di September 2005 (-1=bayar tepat waktu, 1=telat bayar satu bulan, 2=telat bayar dua bulan, ... 8=telat bayar delapan bulan, 9=telat bayar sembilan bulan atau lebih).
- PAY_2: Status pembayaran di Agustus 2005 (skala sama dengan di atas).
- PAY_3: Status pembayaran di Juli 2005 (skala sama dengan di atas).
- PAY_4: Status pembayaran di Juni 2005 (skala sama dengan di atas).
- PAY_5: Status pembayaran di Mei 2005 (skala sama dengan di atas).
- PAY_6: Status pembayaran di April 2005 (skala sama dengan di atas).
- BILL_AMT1: Jumlah tagihan di September 2005 (NT dollar).
- BILL_AMT2: Jumlah tagihan di Agustus 2005 (NT dollar).
- BILL_AMT3: Jumlah tagihan di Juli 2005 (NT dollar).
- BILL_AMT4: Jumlah tagihan di Juni 2005 (NT dollar).
- BILL_AMT5: Jumlah tagihan di Mei 2005 (NT dollar).
- BILL_AMT6: Jumlah tagihan di April 2005 (NT dollar).
- PAY_AMT1: Jumlah pembayaran sebelumnya di September 2005 (NT dollar).
- PAY_AMT2: Jumlah pembayaran sebelumnya di Agustus 2005 (NT dollar).
- PAY_AMT3: Jumlah pembayaran sebelumnya di Juli 2005 (NT dollar).
- PAY_AMT4: Jumlah pembayaran sebelumnya di Juni 2005 (NT dollar).
- PAY_AMT5: Jumlah pembayaran sebelumnya di Mei 2005 (NT dollar).
- PAY_AMT6: Jumlah pembayaran sebelumnya di April 2005 (NT dollar).
- default.payment.next.month: Status default pembayaran bulan berikutnya (1=ya, 0=tidak).

## Exploratory Data Analysis:
- **Mengecek Informasi & Statistik Deskriptif Dataset**: Menggunakan `.info` dan `.describe` untuk mengetahui informasi awal dan statistik deskriptif dari dataset.
- **Mengecek Nilai Unik per Kolom**: Mengevaluasi jumlah nilai unik pada setiap kolom.
- **Mengecek Distribusi Kelas**: Mengevaluasi kolom target dataset default.payment.next.month.
- **Visualisasi**: Histogram digunakan untuk distribusi dari fitur-fitur numerik dan mengecek korelasi antar fitur menggunakan heatmap.


## Data Preparation

### 1. Menghapus Kolom ID

Langkah pertama adalah menghapus **kolom 'ID'**. Kolom ini, meskipun berfungsi sebagai pengenal unik untuk setiap entri data, tidak memberikan informasi prediktif yang relevan bagi model. Menyimpan kolom ID hanya akan menambah dimensi yang tidak perlu dan dapat membingungkan model.

### 2. Mengecek Missing Value

Tidak ditemukan missing values (nilai yang hilang) di semua kolom yang ditampilkan. Ini adalah kabar baik karena berarti tidak ada langkah pra-pemrosesan data tambahan yang diperlukan untuk menangani nilai yang hilang, sehingga data sudah bersih dalam aspek ini dan siap untuk analisis atau pemodelan lebih lanjut.


### 3. Menghapus Baris Duplikat

Selanjutnya, **35 baris duplikat** dihapus dari *dataset*. Baris duplikat adalah entri data yang sama persis, yang bisa terjadi karena kesalahan dalam pengumpulan data atau penggabungan *dataset*. Adanya duplikat dapat menyebabkan bias dalam model, karena model akan "melihat" data yang sama berkali-kali, memberikan bobot yang tidak semestinya pada pola-pola tertentu. Penghapusan duplikat memastikan bahwa setiap observasi unik dan tidak ada pengulangan yang memengaruhi pembelajaran model.

### 4. Penanganan *Outlier* Menggunakan IQR dan *Capping*

*Outlier* atau data pencilan adalah nilai-nilai yang secara signifikan berbeda dari sebagian besar data lainnya. Penanganan *outlier* dilakukan menggunakan metode **Interquartile Range (IQR)** dan **capping**.

* **IQR** dihitung sebagai selisih antara kuartil ketiga ($Q3$) dan kuartil pertama ($Q1$).
* Batas atas dan batas bawah untuk mengidentifikasi *outlier* ditentukan menggunakan rumus:
    * $Batas Atas = Q3 + 1.5 \times IQR$
    * $Batas Bawah = Q1 - 1.5 \times IQR$
* **Capping** berarti nilai-nilai *outlier* yang berada di atas batas atas akan diganti dengan nilai batas atas, dan nilai-nilai *outlier* yang berada di bawah batas bawah akan diganti dengan nilai batas bawah.

Penanganan *outlier* ini penting karena *outlier* dapat mendistorsi statistik deskriptif dan memengaruhi kinerja model secara negatif, terutama model yang sensitif terhadap skala data seperti regresi linear atau K-Means. *Capping* membantu menjaga distribusi data agar tetap mendekati aslinya sambil membatasi dampak ekstrem dari *outlier*.

### 5. Normalisasi Fitur Numerik dengan StandardScaler

Normalisasi fitur numerik dilakukan menggunakan **StandardScaler**. Proses ini mengubah nilai-nilai fitur sehingga memiliki rata-rata nol dan standar deviasi satu. Rumusnya adalah:

$$X_{scaled} = \frac{X - \mu}{\sigma}$$

Di mana $\mu$ adalah rata-rata fitur dan $\sigma$ adalah standar deviasi fitur. Normalisasi ini sangat penting untuk algoritma pembelajaran mesin yang sensitif terhadap skala fitur, seperti Support Vector Machines (SVM), K-Nearest Neighbors (KNN), dan jaringan saraf. Tanpa normalisasi, fitur dengan rentang nilai yang besar akan mendominasi perhitungan jarak atau gradien, menyebabkan model kurang akurat dalam mempelajari pola dari fitur dengan rentang nilai yang lebih kecil.

### 5. Reduksi Dimensi Menggunakan PCA (15 Komponen Utama)

**Principal Component Analysis (PCA)** digunakan untuk reduksi dimensi dengan mengurangi jumlah fitur asli menjadi **15 komponen utama**. PCA adalah teknik statistik yang mengubah fitur-fitur yang berkorelasi menjadi satu set variabel baru yang tidak berkorelasi yang disebut komponen utama. Komponen utama pertama menangkap varians terbesar dalam data, komponen kedua menangkap varians terbesar berikutnya yang tidak dijelaskan oleh komponen pertama, dan seterusnya.

Alasan penggunaan PCA adalah:

* **Mengatasi *Curse of Dimensionality***: Dengan mengurangi jumlah fitur, PCA membantu mengurangi kompleksitas komputasi, mempercepat proses pelatihan model, dan mengurangi risiko *overfitting* (ketika model terlalu spesifik terhadap data pelatihan dan tidak bekerja dengan baik pada data baru).
* **Menghilangkan Multikolinearitas**: PCA menciptakan komponen yang tidak berkorelasi, yang dapat meningkatkan stabilitas dan interpretasi model, terutama dalam regresi.
* **Visualisasi Data**: Dengan mengurangi dimensi menjadi dua atau tiga komponen utama, data dapat divisualisasikan dengan lebih mudah untuk mengidentifikasi pola atau kelompok.

### 6. *Train-Test-Validation Split* Data

Langkah terakhir adalah membagi data menjadi tiga bagian: **data pelatihan (*train*)**, **data validasi (*validation*)**, dan **data uji (*test*)**.

* **Data Pelatihan (*Train Set*)**: Digunakan untuk melatih model. Model belajar pola dan hubungan dari data ini.
* **Data Validasi (*Validation Set*)**: Digunakan untuk menyetel *hyperparameter* model dan mengevaluasi kinerja model selama proses pengembangan. Ini membantu mencegah *overfitting* pada data uji.
* **Data Uji (*Test Set*)**: Digunakan untuk mengevaluasi kinerja akhir model secara objektif setelah model selesai dilatih dan disetel. Data ini tidak pernah digunakan selama pelatihan atau penyetelan *hyperparameter* untuk memastikan evaluasi yang tidak bias terhadap kemampuan generalisasi model.

Pembagian ini penting untuk memastikan bahwa model yang dikembangkan dapat bekerja dengan baik pada data yang belum pernah dilihat sebelumnya (generalisasi), bukan hanya menghafal data pelatihan.

---

## Modeling
Model utama yang digunakan adalah Neural Network dengan struktur:
- Dense(64) → BatchNorm → Dropout(0.3)
- Dense(32) → BatchNorm → Dropout(0.3)
- Output: Dense(1, activation='sigmoid')

**Parameter:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Callback: EarlyStopping & ReduceLROnPlateau

## Modeling Process

**Data Balancing (SMOTE):**
Karena dataset memiliki ketidakseimbangan kelas (sekitar 22% default), digunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menghasilkan sampel sintetis pada kelas minoritas sebelum pelatihan model.

Hasil setelah SMOTE:
- Jumlah sampel kelas 0 (tidak default): 14.001
- Jumlah sampel kelas 1 (default): 14.001
  
Langkah ini membantu model agar tidak terlalu bias terhadap kelas mayoritas.

**Callback**

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
```

EarlyStopping akan menghentikan pelatihan jika val_loss (loss validasi) tidak membaik selama 10 epoch berturut-turut, dan akan mengembalikan bobot model terbaik. Sementara itu, ReduceLROnPlateau akan mengurangi learning rate menjadi setengahnya jika val_loss tidak membaik selama 5 epoch, dengan batas minimum learning rate 0.0001. Ini bertujuan untuk mencegah overfitting dan membantu model menemukan konvergensi yang lebih baik.

**Model Training**

```python
history = model.fit(
    X_train_bal, y_train_bal,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

loss, accuracy, precision, recall, f1_score_val = model.evaluate(X_test, y_test, verbose=0)
```

**Improvement:**
- Untuk meningkatkan performa model Neural Network, dilakukan beberapa strategi sebagai berikut: Penambahan Layer Batch Normalization: Membantu mempercepat pelatihan dan membuat proses training lebih stabil dengan menormalisasi output dari layer sebelumnya.
- Dropout Regularization (Dropout Rate 0.3): Digunakan untuk menghindari overfitting dengan menghapus sebagian neuron secara acak pada saat training.
- Callback - EarlyStopping: Menghentikan pelatihan saat metrik validasi tidak membaik lagi, mencegah overfitting.
- Callback - ReduceLROnPlateau: Menurunkan learning rate saat metrik validasi stagnan, membantu konvergensi model yang lebih baik.
- 
**Kelebihan dan Kekurangan:**
- **Kelebihan Neural Network:**
  - Mampu menangkap hubungan non-linear dan kompleks antar fitur.
  - Fleksibel untuk berbagai jenis data dan skenario.
  - Dapat ditingkatkan dengan arsitektur yang lebih kompleks sesuai kebutuhan.
- **Kekurangan Neural Network:**
  - Memerlukan banyak data dan waktu pelatihan.
  - Rentan terhadap overfitting tanpa regularisasi yang baik.
  - Tidak secepat dan seintuitif algoritma klasik dalam interpretasi hasil (misalnya dibandingkan dengan decision tree).

  Model ini dipilih karena kemampuannya dalam menangani kompleksitas hubungan antar fitur serta fleksibilitas dalam arsitektur dan tuning. Meskipun hanya satu algoritma yang digunakan dalam proyek ini, proses tuning yang cermat dan langkah balancing data yang tepat membuat performanya kompetitif untuk menyelesaikan permasalahan klasifikasi default pembayaran ini.


## Evaluation

**Metrik yang digunakan:**

Model dievaluasi menggunakan metrik berikut untuk mengukur performa dan kualitas prediksinya:

### 1. *Accuracy*

*Accuracy* adalah persentase prediksi yang benar dari keseluruhan data. Metrik ini memberikan gambaran umum seberapa sering model membuat prediksi yang tepat.

* **Formula:**
    $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

    Di mana:
    * $TP$ (True Positive): Jumlah kasus positif yang diprediksi dengan benar sebagai positif.
    * $TN$ (True Negative): Jumlah kasus negatif yang diprediksi dengan benar sebagai negatif.
    * $FP$ (False Positive): Jumlah kasus negatif yang salah diprediksi sebagai positif.
    * $FN$ (False Negative): Jumlah kasus positif yang salah diprediksi sebagai negatif.

### 2. *Precision*

*Precision* adalah rasio antara *true positive* dan total prediksi positif. Metrik ini mengukur seberapa relevan hasil prediksi positif model.

* **Formula:**
    $$Precision = \frac{TP}{TP + FP}$$

### 3. *Recall*

*Recall* (juga dikenal sebagai *Sensitivity* atau Tingkat *True Positive*) adalah rasio antara *true positive* dan total aktual kasus positif. Metrik ini mengukur kemampuan model untuk menemukan semua kasus positif yang relevan.

* **Formula:**
    $$Recall = \frac{TP}{TP + FN}$$

### 4. *F1 Score*

*F1 Score* adalah rata-rata harmonik dari *precision* dan *recall*. Metrik ini sangat berguna ketika ada ketidakseimbangan kelas dalam dataset, karena memberikan keseimbangan antara *precision* dan *recall*.

* **Formula:**
    $$F1 Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

---

**Hasil Evaluasi Model (Sebelum Tuning)**:
```
Classification Report:

              precision    recall  f1-score   support

           0     0.8784    0.7802    0.8264      4667
           1     0.4448    0.6199    0.5180      1326

    accuracy                         0.7447      5993
   macro avg     0.6616    0.7000    0.6722      5993
weighted avg     0.7825    0.7447    0.7581      5993
```

**Hasil Evaluasi Model (Setelah Tuning)**:
```
Classification Report:
               precision    recall  f1-score   support

           0       0.84      0.93      0.89      4667
           1       0.63      0.39      0.48      1326

    accuracy                           0.81      5993
   macro avg       0.74      0.66      0.68      5993
weighted avg       0.80      0.81      0.80      5993
```


Berdasarkan hasil evaluasi, model setelah tuning menunjukkan peningkatan performa yang jelas dan menjadikannya model yang lebih baik secara keseluruhan. Peningkatan yang paling signifikan terlihat pada accuracy yang melonjak dari 0.7447 menjadi 0.81, serta weighted average F1-score yang naik dari 0.7581 menjadi 0.80, menunjukkan keseimbangan yang lebih baik antara presisi dan recall secara keseluruhan. Meskipun terdapat trade-off di mana recall untuk Class 1 menurun dari 0.6199 menjadi 0.39, hal ini diimbangi oleh peningkatan signifikan pada precision Class 1 dari 0.4448 menjadi 0.63, yang berarti prediksi Class 1 oleh model menjadi lebih akurat dan dapat diandalkan. Dengan demikian, meskipun ada penurunan dalam menangkap semua instansi Class 1, model yang sudah di-tuning jauh lebih baik dalam membuat prediksi yang benar secara umum dan memiliki kualitas prediksi Class 1 yang lebih tinggi.

Penggunaan SMOTE membantu meningkatkan jumlah data pada kelas minoritas, tetapi trade-off antara precision dan recall masih menjadi tantangan utama dalam kasus ketidakseimbangan data seperti ini. (Contoh Prediksi):
```
Hasil Prediksi: Berpotensi Default (1)
  Probabilitas Keyakinan Model: 0.5847
  Indikator Risiko:
    - Batas kredit rendah (< 100rb).
    - Terdapat riwayat keterlambatan pembayaran (PAY_0: 2).
    - Keterlambatan pembayaran bulan kedua (PAY_2: 2).
    - Usia relatif muda (< 30 tahun), mungkin pengalaman finansial kurang.
    - Rasio tagihan terhadap batas kredit tinggi (BILL_AMT1: 30000 dari LIMIT_BAL: 50000).
    - Tidak ada pembayaran dilakukan di bulan sebelumnya.
```

**Formula Metrik**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

