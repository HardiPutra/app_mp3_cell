import pandas as pd
import re

# =========================
# KONFIGURASI FILE
# =========================
input_file = "data/Spredseet_Clean.xlsx"
output_file = "data/data_spredseetbersih.csv"

# =========================
# 1. BACA FILE EXCEL
# =========================
df = pd.read_excel(input_file)

# =========================
# 2. HAPUS KOLOM 'No' JIKA ADA
# =========================
if "No" in df.columns:
    df = df.drop(columns=["No"])

# =========================
# 3. GANTI NAMA KOLOM SESUAI FITUR
# =========================
df.columns = [
    "Usia",
    "Pekerjaan",
    "Budget",
    "Performa",
    "Kamera",
    "Storage",
    "Baterai",
    "Ergonomi",
    "Label"
]

# =========================
# 4. AMBIL HURUF A / B / C SAJA
# =========================
def ambil_kode(jawaban):
    if pd.isna(jawaban):
        return None
    match = re.match(r"([ABC])", str(jawaban).strip())
    return match.group(1) if match else None

for col in df.columns:
    df[col] = df[col].apply(ambil_kode)

# =========================
# 5. SIMPAN KE CSV
# =========================
df.to_csv(output_file, index=False)

print("✅ Data berhasil dibersihkan dan disimpan ke:", output_file)
