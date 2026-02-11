import pandas as pd

# 1. Baca file CSV awal
input_file = "data/9.data_processing.csv"        # ganti dengan nama file Anda
output_file = "data/final.csv"

df = pd.read_csv(input_file)

# 2. Hapus kolom Budget jika ada
if "Budget" in df.columns:
    df = df.drop(columns=["Budget"])

# 3. Simpan ke file CSV baru
df.to_csv(output_file, index=False)

print("Kolom Budget berhasil dihapus.")
print("Data baru disimpan sebagai:", output_file)
