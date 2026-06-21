# Klasifikasi Daun Nilam Deployment

Proyek ini adalah implementasi *deployment* dari model Machine Learning (Support Vector Machine) untuk mengklasifikasikan daun nilam ke dalam 3 jenis:
- Nilam Batik (0)
- Nilam Biasa (1)
- Nilam Seledri (2)

## Fitur Ekstraksi
Aplikasi menggunakan metode ekstraksi fitur berikut sebelum memprediksi:
1. **HSV**: Menghitung *Mean* dan *Standar Deviasi* dari Hue, Saturation, dan Value.
2. **GLCM**: Menghitung matriks kookurensi abu-abu (jarak 1, sudut 0) dan mendapatkan fitur *Contrast, Correlation, Energy, Homogeneity, Dissimilarity, ASM*.

## Menjalankan Proyek secara Lokal

### Persyaratan
Pastikan Python sudah terinstal di PC Anda. Disarankan menggunakan *virtual environment*.

### Langkah-langkah
1. Install semua *library* yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Aplikasi web akan terbuka secara otomatis di *browser* pada alamat `http://localhost:8501`.
