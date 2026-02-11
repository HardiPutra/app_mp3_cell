import pandas as pd 
import joblib
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 0. FUNGSI BANTUAN (UNTUK PLOT GAMBAR)
# ==========================================
def save_confusion_matrix(cm, title, filename, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Label Asli')
    plt.xlabel('Prediksi Model')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"🖼️  Gambar disimpan: {filename}")

# ==========================================
# 1. KONFIGURASI PATH
# ==========================================
if not os.path.exists("model"):
    os.makedirs("model")

# ==========================================
# 2. LOAD DATASET
# ==========================================
print("📂 Memuat dataset...")
try:
    if os.path.exists("data/data_latih_final.csv"):
        df_train = pd.read_csv("data/data_latih_final.csv")
        df_test = pd.read_csv("data/data_uji_final.csv")
    else:
        df_train = pd.read_csv("data_latih_final.csv")
        df_test = pd.read_csv("data_uji_final.csv")
        
except FileNotFoundError:
    print("❌ Error: File CSV tidak ditemukan.")
    exit()

# Pisahkan Fitur dan Target
X_train = df_train.drop(columns=["Label"])
y_train = df_train["Label"]

X_test = df_test.drop(columns=["Label"])
y_test = df_test["Label"]

target_names = ["Ekonomis", "Produktivitas", "Gaming"]

print(f"✅ Data Latih: {len(X_train)} baris")
print(f"✅ Data Uji  : {len(X_test)} baris")
print("-" * 60)

# =================================================================
# 3. SKENARIO A: BASELINE (SEBELUM TUNING / ALPHA STANDARD)
# =================================================================
print("\n" + "="*60)
print("   SKENARIO A: MODEL BASELINE (Alpha Default = 1.0)")
print("="*60)

# Model Standar (Laplace Smoothing Biasa)
model_base = CategoricalNB(alpha=1.0, force_alpha=True, min_categories=3)
model_base.fit(X_train, y_train)

# Prediksi & Evaluasi Baseline
y_pred_base = model_base.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
cm_base = confusion_matrix(y_test, y_pred_base)

print(f"🎯 AKURASI AWAL : {acc_base * 100:.2f}%")
print("-" * 60)
print("📊 Detail Laporan (Baseline):")
print(classification_report(y_test, y_pred_base, target_names=target_names))

# Simpan Gambar CM Baseline
save_confusion_matrix(cm_base, "Confusion Matrix (Baseline Alpha=1.0)", "model/cm_baseline.png", target_names)


# =================================================================
# 4. SKENARIO B: HYPERPARAMETER TUNING (MENCARI ALPHA TERBAIK)
# =================================================================
print("\n" + "-"*60)
print("🤖 Melakukan Tuning Alpha (Grid Search)...")
print("-"*60)

# Range Alpha yang diuji
param_grid = {
    'alpha': [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=CategoricalNB(force_alpha=True, min_categories=3), 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_['alpha']

print(f"✅ Tuning Selesai!")
print(f"💎 Alpha Terbaik Ditemukan: {best_alpha}")


# =================================================================
# 5. SKENARIO C: HASIL AKHIR (SETELAH TUNING)
# =================================================================
print("\n" + "="*60)
print(f"   SKENARIO B: MODEL OPTIMASI (Alpha Terbaik = {best_alpha})")
print("="*60)

# Prediksi menggunakan model terbaik
y_pred_tuned = best_model.predict(X_test)

# Evaluasi Tuned
acc_tuned = accuracy_score(y_test, y_pred_tuned)
cm_tuned = confusion_matrix