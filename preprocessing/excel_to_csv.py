import pandas as pd

# =========================
# KONFIGURASI FILE
# =========================
input_excel = "data/2.Data_Mp3_Clean.xlsx"
output_csv = "data/data_clean.csv"

# =========================
# 1. BACA FILE EXCEL
# =========================
df = pd.read_excel(input_excel)

# =========================
# 2. HAPUS KOLOM 'No' JIKA ADA
# =========================
if "No" in df.columns:
    df = df.drop(columns=["No"])

# =========================
# 3. SIMPAN KE FILE CSV
# =========================
df.to_csv(output_csv, index=False)

print("✅ File Excel berhasil dikonversi ke CSV.")
print("📁 Output:", output_csv)
