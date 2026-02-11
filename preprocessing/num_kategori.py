import pandas as pd
import os

# ==========================================
# KONFIGURASI NAMA FILE
# ==========================================
# Ganti 'data_training.csv' dengan nama file asli data Anda
file_input = 'data/7.final_sebelumsplit.csv' 
file_output = 'data/7.data_kategori.csv'

# ==========================================
# 1. LOAD DATA
# ==========================================
if not os.path.exists(file_input):
    print(f"❌ Error: File '{file_input}' tidak ditemukan di folder ini.")
    exit()

print(f"📂 Membaca file: {file_input}...")
df = pd.read_csv(file_input)

# ==========================================
# 2. KAMUS PENERJEMAH (MAPPING)
# ==========================================
# Sesuai request Anda:

# Usia
map_usia = {
    0: 'Remaja', 
    1: 'Dewasa', 
    2: 'Lansia'
}

# Pekerjaan
map_pekerjaan = {
    0: 'Pelajar', 
    1: 'Bekerja', 
    2: 'Tidak_Bekerja'
}

# Spesifikasi Teknis (Performa, Kamera, Storage, Baterai, Ergonomi)
# Sesuai request: Rendah, Sedang, Tinggi
map_spek = {
    0: 'Rendah', 
    1: 'Sedang', 
    2: 'Tinggi'
}

# Label / Target
map_label = {
    0: 'Ekonomis', 
    1: 'Produktivitas', 
    2: 'Gaming'
}

# ==========================================
# 3. PROSES KONVERSI
# ==========================================
try:
    # A. Konversi Profil
    df['Usia'] = df['Usia'].map(map_usia)
    df['Pekerjaan'] = df['Pekerjaan'].map(map_pekerjaan)

    # B. Konversi Spesifikasi (Looping untuk kolom yang jenisnya sama)
    cols_spek = ['Performa', 'Kamera', 'Storage', 'Baterai', 'Ergonomi']
    for col in cols_spek:
        df[col] = df[col].map(map_spek)

    # C. Konversi Label (Jika kolom Label ada)
    if 'Label' in df.columns:
        df['Label'] = df['Label'].map(map_label)

    print("✅ Konversi data selesai!")

    # ==========================================
    # 4. SIMPAN KE CSV BARU
    # ==========================================
    df.to_csv(file_output, index=False)
    print(f"💾 File berhasil disimpan sebagai: '{file_output}'")
    
    # Tampilkan 5 data pertama sebagai preview
    print("\n--- Preview 5 Data Pertama ---")
    print(df.head().to_string())

except Exception as e:
    print(f"⚠️ Terjadi kesalahan: {e}")