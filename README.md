# Deteksi Ulasan Produk Hasil Generatif AI Berbasis MLP dan Fitur Stilometrik (UTS)

Proyek UTS mata kuliah **Pembelajaran Mendalam**, Kelompok **CILUKBA** (IFC-Sore), Program Studi S-1 Teknik Informatika, Universitas Mikroskil.

| NIM | Nama |
|---|---|
| 231110729 | Delvin Ayers |
| 231111441 | Osfredo Vent Kholish |
| 231110743 | Thienry |
| 231111101 | Andrian Cheniago |

## Ringkasan
Model Multi-Layer Perceptron (DNN) dilatih untuk menduga apakah sebuah ulasan produk berbahasa Inggris ditulis manusia (OR) atau dihasilkan utuh oleh LLM (CG), hanya dari 11 fitur stilometrik. Evaluasi keandalan memakai tiga *random seed* (1998, 7, 42), *threshold* t* dengan FPR data validation ≤ 1%, uji McNemar *mid-p*, serta pengujian pada ulasan dari model AI yang tidak dipakai saat pelatihan (GLM-5.2 dan LFM2.5-1.2B). Proyek UTS menambahkan kompresi model ke TensorFlow Lite dan aplikasi website Gradio yang dijalankan secara lokal untuk memeriksa satu teks ulasan.

## Struktur
```
UTS_Deteksi_Ulasan_AI_MLP_Stilometrik.ipynb      # seluruh alur eksperimen + artefak aplikasi website
src/        # modul yang ditulis oleh notebook (stylometry, data, selection, model, evaluation, inference)
figures/    # grafik 300 DPI yang dipakai di laporan
models/     # model Keras S1 seed 1998
outputs/    # results.json: seluruh angka yang dikutip di laporan
app/        # aplikasi website Gradio (app.py, stylometry.py, inference.py, mlp_s1.tflite, config.json, requirements.txt)
requirements.txt
```

## Cara menjalankan
1. Pasang Python 3.13 (versi yang dipakai saat eksekusi: 3.13.18).
2. `python -m venv .venv` lalu aktifkan (`.venv\Scripts\activate` di Windows atau `source .venv/bin/activate` di Linux/macOS).
3. `pip install -r requirements.txt`
4. `jupyter notebook UTS_Deteksi_Ulasan_AI_MLP_Stilometrik.ipynb`, lalu jalankan **Kernel → Restart & Run All**.

Data dimuat dari Hugging Face Hub (`Flowerly/modern-fake-reviews`, `-glm`, `-lfm`). Bila Hub tidak dapat diakses, notebook mengunduh CSV resmi dari GitHub pembuat dataset (`PurplexyFlower/Fake-review`) dan membaginya ulang dengan prosedur resmi (`train_test_split`, seed 1998), sehingga ukurannya identik: 32.339 / 2.425 / 5.660. Seluruh eksekusi berjalan di CPU (sekitar 30 sampai 45 menit).

## Lisensi data
Modern Fake Reviews dirilis dengan lisensi CC-BY-4.0 oleh pembuatnya.

## Menjalankan aplikasi secara lokal
```
cd app
pip install -r requirements.txt
python app.py   # buka http://127.0.0.1:7860 di web browser
```
Cara pakai: tempel satu teks ulasan berbahasa Inggris (atau pilih contoh), tekan **Periksa Ulasan**, lalu baca kategori dugaan, catatan keandalan (FPR dan FNR sesuai panjang teks), dan peringatan bila teks terlalu pendek atau terlalu panjang. Tombol **Bersihkan** mengosongkan masukan dan hasil.
