# API Analisis Sentimen Hotel

Layanan (service) ini menyediakan sebuah API endpoint untuk melakukan analisis sentimen pada teks ulasan hotel. API ini menerima teks dan mengembalikan salah satu dari tiga klasifikasi: `positive`, `negative`, atau `neutral`.

Layanan ini dibuat menggunakan Python dengan framework Flask dan model Machine Learning dari Scikit-learn.

---

## Prasyarat

-   Python 3.8+
-   pip (Package Installer for Python)

---

## Setup & Instalasi Lokal

Langkah-langkah berikut digunakan untuk menjalankan server API ini di lingkungan development lokal.

1.  **Siapkan Folder Proyek**
    Pastikan Anda memiliki semua file proyek dalam satu folder.

2.  **Buat Virtual Environment (Sangat Direkomendasikan)**
    Membuka terminal di dalam folder proyek, lalu jalankan:
    ```bash
    # Membuat environment baru bernama 'venv'
    python3 -m venv venv
    ```
    Aktifkan environment tersebut:
    ```bash
    # Untuk macOS/Linux
    source venv/bin/activate

    # Untuk Windows
    venv\Scripts\activate
    ```

3.  **Install Semua Dependensi**
    Gunakan file `requirements.txt` untuk menginstal semua library yang dibutuhkan dengan versi yang tepat.
    ```bash
    pip install -r requirements.txt
    ```

---

## Menjalankan Server API

Setelah instalasi selesai, Anda bisa menjalankan server development Flask.

```bash
python app.py