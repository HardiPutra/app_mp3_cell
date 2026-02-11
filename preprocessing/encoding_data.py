import pandas as pd

# =========================
# 1. BACA DATA CSV
# =========================
input_file = "data/7.mixdata_before_preprocessing.csv"
df = pd.read_csv(input_file)

# =========================
# 2. HAPUS KOLOM 'No' JIKA ADA
# =========================
if "No" in df.columns:
    df = df.drop(columns=["No"])

# =========================
# 3. ENCODING KATEGORIKAL
# =========================
map_usia = {"A": "Remaja", "B": "Dewasa", "C": "Lansia"}
map_pekerjaan = {"A": "Pelajar", "B": "Bekerja", "C": "Tidak Bekerja"}
map_rsh = {"A": "Rendah", "B": "Sedang", "C": "Tinggi"}
map_label = {"A": "Ekonomis", "B": "Produktivitas", "C": "Gaming"}

df_kat = df.copy()
df_kat["Usia"] = df_kat["Usia"].map(map_usia)
df_kat["Pekerjaan"] = df_kat["Pekerjaan"].map(map_pekerjaan)

for col in ["Budget", "Performa", "Kamera", "Storage", "Baterai", "Ergonomi"]:
    df_kat[col] = df_kat[col].map(map_rsh)

df_kat["Label"] = df_kat["Label"].map(map_label)

# Cek error encoding kategorikal
if df_kat.isnull().any().any():
    print("⚠️ Peringatan: Ada data yang gagal di-encoding (kategorikal)")
    print(df_kat.isnull().sum())
else:
    print("✅ Encoding kategorikal berhasil")

# Simpan hasil kategorikal
df_kat.to_csv("data/8.data_encoded_kategorikal.csv", index=False)

# =========================
# 4. ENCODING NUMERIK
# =========================
map_usia_num = {"Remaja": 0, "Dewasa": 1, "Lansia": 2}
map_pekerjaan_num = {"Pelajar": 0, "Bekerja": 1, "Tidak Bekerja": 2}
map_rsh_num = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}
map_label_num = {"Ekonomis": 0, "Produktivitas": 1, "Gaming": 2}

df_num = df_kat.copy()
df_num["Usia"] = df_num["Usia"].map(map_usia_num)
df_num["Pekerjaan"] = df_num["Pekerjaan"].map(map_pekerjaan_num)

for col in ["Budget", "Performa", "Kamera", "Storage", "Baterai", "Ergonomi"]:
    df_num[col] = df_num[col].map(map_rsh_num)

df_num["Label"] = df_num["Label"].map(map_label_num)

# Cek error encoding numerik
if df_num.isnull().any().any():
    print("⚠️ Peringatan: Ada data yang gagal di-encoding (numerik)")
    print(df_num.isnull().sum())
else:
    print("✅ Encoding numerik berhasil")

# Simpan hasil numerik
df_num.to_csv("data/9.data_processing.csv", index=False)
