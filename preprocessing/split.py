import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# SCRIPT PEMBAGI DATA STATIS (FIXED SPLIT)
# Tujuannya: Agar data Latih & Uji tidak berubah-ubah saat dibandingkan manual
# =============================================================================

# 1. Load Data Mentah (Yang sudah bersih tanpa kolom Budget)
filename = "data/7.final_sebelumsplit.csv" 
try:
    df = pd.read_csv(filename)
    print(f"✅ Berhasil membaca file {filename}")
    print(f"   Total Data Awal: {len(df)} baris")
except FileNotFoundError:
    print(f"❌ Error: File {filename} tidak ditemukan.")
    exit()

# 2. DEFINISI FITUR (X) DAN TARGET (y)
X = df.drop(columns=["Label"])
y = df["Label"]

# 3. SPLIT DATA (80:20)
# random_state=42 : Kunci pengacakan agar hasilnya selalu sama siapapun yang menjalankan
# stratify=y      : Menjaga proporsi kelas (Ekonomis/Prod/Gaming) seimbang di kedua file
print("\n🔄 Sedang membagi data secara acak (80:20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. GABUNGKAN KEMBALI AGAR MUDAH DIBACA DI EXCEL
# (Kita satukan Fitur + Labelnya lagi)
train_data = X_train.copy()
train_data['Label'] = y_train

test_data = X_test.copy()
test_data['Label'] = y_test

# 5. SIMPAN KE FILE BARU
train_data.to_csv("data_latih_final.csv", index=False)
test_data.to_csv("data_uji_final.csv", index=False)

print("\n✅ SUKSES! DATA TELAH DISIMPAN SECARA PERMANEN.")
print("-" * 50)
print(f"1. File: data_latih_final.csv")
print(f"   - Jumlah: {len(train_data)} data")
print(f"   - FUNGSI: Gunakan ini untuk menghitung LIKELIHOOD di Excel.")
print("-" * 50)
print(f"2. File: data_uji_final.csv")
print(f"   - Jumlah: {len(test_data)} data")
print(f"   - FUNGSI: Gunakan ini untuk menghitung AKURASI & POSTERIOR di Excel.")
print("-" * 50)