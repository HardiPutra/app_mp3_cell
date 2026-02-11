import streamlit as st
import pandas as pd
import joblib
import json
import os
import io
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader

# ==============================================================================
# 1. KONFIGURASI HALAMAN & CSS MODERN
# ==============================================================================
st.set_page_config(
    page_title="MP3 Cellular Smart System",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --- FUNGSI LOAD CSS EKSTERNAL ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles/style.css")

# ==============================================================================
# 2. FUNGSI UTILITAS
# ==============================================================================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("model/naive_bayes_model.pkl")
    except FileNotFoundError:
        return None, None
        
    try:
        with open("model/hasil_evaluasi_train.json", "r") as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        eval_data = None
        
    return model, eval_data

def save_to_history(data_dict):
    """Menyimpan data prediksi ke file CSV histori."""
    file_path = "history_data.csv"
    df_new = pd.DataFrame([data_dict])
    
    # Cek apakah file sudah ada
    if not os.path.exists(file_path):
        df_new.to_csv(file_path, index=False)
    else:
        # Append mode, pastikan tidak menulis header lagi
        df_new.to_csv(file_path, mode='a', header=False, index=False)

def get_label_name(index):
    return {0: "Ekonomis", 1: "Produktivitas", 2: "Gaming"}.get(index, "Unknown")

# ====== RECOMMENDATION CSV HELPERS ======
from pathlib import Path
import csv, re

BASE_DIR = Path(__file__).parent
RECO_CSV = BASE_DIR / "data" / "recommendations.csv"
RECO_IMG_DIR = BASE_DIR / "assets" / "reco_images"
RECO_IMG_DIR.mkdir(parents=True, exist_ok=True)

def _safe_filename(filename: str) -> str:
    # simple sanitize
    name = Path(filename).name
    name = re.sub(r'[^A-Za-z0-9._-]', '_', name)
    return name

def load_recommendations():
    """Load CSV -> dict {category: [ {nama, harga, desc, img}, ... ] }"""
    rec = {}
    if not RECO_CSV.exists():
        return rec
    with open(RECO_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            cat = r['category']
            rec.setdefault(cat, []).append({
                'nama': r['name'],
                'harga': r['price'],
                'desc': r['desc'],
                'img': str(RECO_IMG_DIR / r['img']) if r.get('img') else ''
            })
    return rec

def save_recommendations_from_list(rows):
    """rows: list of dicts with keys category,name,price,desc,img (img = filename)"""
    RECO_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RECO_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['category','name','price','desc','img'])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                'category': r.get('category',''),
                'name': r.get('name',''),
                'price': r.get('price',''),
                'desc': r.get('desc',''),
                'img': r.get('img','')
            })

def save_uploaded_image(uploaded_file):
    """Save Streamlit uploaded image to assets/reco_images and return filename"""
    if uploaded_file is None:
        return ""
    safe = _safe_filename(uploaded_file.name)
    dest = RECO_IMG_DIR / safe
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(uploaded_file.getbuffer())
    os.replace(tmp, dest)
    return safe
# ====== END HELPERS ======


def calculate_feature_importance(df, features, target_col='Prediksi_Sistem'):
    """
    Menghitung tingkat kepentingan fitur menggunakan Mutual Information
    untuk menunjukkan seberapa kuat pengaruh fitur terhadap klasifikasi
    """
    
    X = df[features]
    y = df[target_col]
    
    # Hitung Mutual Information Score
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Normalisasi ke persentase (0-100%)
    if mi_scores.max() > 0:
        importance_pct = (mi_scores / mi_scores.max()) * 100
    else:
        importance_pct = mi_scores
    
    # Buat DataFrame untuk visualisasi
    importance_df = pd.DataFrame({
        'Fitur': features,
        'Importance': mi_scores,
        'Importance_Pct': importance_pct
    }).sort_values('Importance', ascending=False)
    
    return importance_df

def generate_feature_insights(df, features, target_col='Prediksi_Sistem'):
    """
    Generate insight otomatis tentang hubungan fitur dengan kategori
    """
    insights = []
    target_order = ["Ekonomis", "Produktivitas", "Gaming"]
    
    # Mapping nama kategori untuk fitur
    feature_maps = {
        "Usia": {0: "Remaja", 1: "Dewasa", 2: "Lansia"},
        "Pekerjaan": {0: "Pelajar", 1: "Pekerja", 2: "Tidak Bekerja"},
        "default": {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
    }
    
    for feat in features[:3]:  # Top 3 fitur paling penting
        # Hitung distribusi untuk setiap kategori
        for label_idx, label_name in enumerate(target_order):
            df_subset = df[df[target_col] == label_idx]
            if len(df_subset) > 0:
                # Cari nilai fitur yang paling dominan
                mode_val = df_subset[feat].mode()[0] if len(df_subset[feat].mode()) > 0 else None
                if mode_val is not None:
                    count = (df_subset[feat] == mode_val).sum()
                    pct = (count / len(df_subset)) * 100
                    
                    # Mapping nilai ke kategori
                    if feat in feature_maps:
                        val_name = feature_maps[feat].get(mode_val, f"Nilai {mode_val}")
                    else:
                        val_name = feature_maps["default"].get(mode_val, f"Nilai {mode_val}")
                    
                    if pct >= 60:  # Hanya tampilkan insight kuat (>60%)
                        insights.append({
                            'fitur': feat,
                            'kategori': label_name,
                            'nilai': val_name,
                            'persentase': pct
                        })
    
    return insights

# ==============================================================================
# FUNGSI BARU: REKOMENDASI HP
# ==============================================================================
def get_phone_recommendations(label_text):
    # baca dari CSV (jika tidak ada, fallback ke hardcoded)
    rec = load_recommendations()
    if rec:
        return rec.get(label_text, [])
    # fallback (jika kamu belum mengisi CSV)
    return {
        "Ekonomis": [
            {"nama":"Infinix Smart 8","harga":"Rp 1.100.000","desc":"Baterai awet","img":""}
        ],
        "Produktivitas": [],
        "Gaming": []
    }.get(label_text, [])

# ==============================================================================
# FUNGSI BARU: GENERATE PDF REPORT (PERBAIKAN VISUAL)
# ==============================================================================
matplotlib.use('Agg')
def create_pdf_report(df_hist):
    buffer = io.BytesIO()
    # Gunakan margin yang pas, tidak terlalu mepet
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=20, bottomMargin=30)
    elements = []
    
    # --- IMPORT TAMBAHAN YANG DIPERLUKAN ---
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    # --- PALET WARNA MODERN (Flat Design) ---
    COLOR_PRIMARY = rl_colors.HexColor("#2C3E50")  # Navy Blue (Header)
    COLOR_ACCENT  = rl_colors.HexColor("#3498DB")  # Bright Blue
    COLOR_SUCCESS = rl_colors.HexColor("#27AE60")  # Emerald Green
    COLOR_WARNING = rl_colors.HexColor("#F39C12")  # Orange
    COLOR_BG_GRAY = rl_colors.HexColor("#ECF0F1")  # Soft Gray (Zebra Striping)
    
    # --- STYLE ---
    styles = getSampleStyleSheet()
    
    # Style Judul Header (Tengah)
    style_header_title = ParagraphStyle(
        'HeaderTitle', 
        parent=styles['Title'], 
        fontSize=22, 
        fontName='Helvetica-Bold', 
        textColor=COLOR_PRIMARY, 
        alignment=TA_CENTER,  # Rata Tengah
        spaceAfter=5
    )
    
    style_header_sub = ParagraphStyle(
        'HeaderSub', 
        parent=styles['Normal'], 
        fontSize=11, 
        textColor=rl_colors.gray, 
        alignment=TA_CENTER, # Rata Tengah
        spaceAfter=2
    )

    style_header_date = ParagraphStyle(
        'HeaderDate', 
        parent=styles['Normal'], 
        fontSize=9, 
        textColor=rl_colors.gray, 
        alignment=TA_CENTER # Rata Tengah
    )

    # Style Judul Section (Visualisasi & Tabel) - RATA TENGAH
    style_h2 = ParagraphStyle(
        'CustomH2', 
        parent=styles['Heading2'], 
        fontSize=14, 
        fontName='Helvetica-Bold', 
        textColor=COLOR_PRIMARY, 
        alignment=TA_CENTER, # <--- DIUBAH KE TENGAH
        spaceBefore=20, 
        spaceAfter=10
    )
    
    # -----------------------------------------------------------
    # 1. HEADER (KOP LAPORAN MODERN)
    # -----------------------------------------------------------
    logo_path = "assets/logo.png" 
    
    if os.path.exists(logo_path):
        utils_img = ImageReader(logo_path)
        orig_w, orig_h = utils_img.getSize()
        
        target_height = 1.2 * inch
        aspect_ratio = orig_w / orig_h
        target_width = target_height * aspect_ratio
        
        if target_width > 2.5 * inch: 
            target_width = 2.5 * inch
            target_height = target_width / aspect_ratio

        logo_img = RLImage(logo_path, width=target_width, height=target_height)
    else:
        logo_img = Spacer(1, 1)

    tgl_cetak = pd.Timestamp.now().strftime("%d %B %Y")
    
    text_content = [
        Paragraph("MP3 CELLULAR SOLOK", style_header_title),
        Paragraph("Laporan Analisis Preferensi Pelanggan & Stok", style_header_sub),
        Paragraph(f"Dicetak pada: {tgl_cetak}", style_header_date)
    ]

    header_table_data = [[logo_img, text_content]]
    
    t_header = Table(header_table_data, colWidths=[1.5*inch, 5.5*inch])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'CENTER'),
        ('LINEBELOW', (0,0), (-1,-1), 2, COLOR_PRIMARY),
        ('BOTTOMPADDING', (0,0), (-1,-1), 15),
    ]))
    
    elements.append(t_header)
    elements.append(Spacer(1, 0.4 * inch))
    
    # -----------------------------------------------------------
    # 2. KPI CARDS (KOTAK RINGKASAN - LEBIH SMOOTH)
    # -----------------------------------------------------------
    total_data = len(df_hist)
    top_cat = "-"
    top_pct = 0
    if not df_hist.empty:
        top_cat = df_hist['Hasil_Prediksi'].mode()[0]
        top_cat_count = df_hist[df_hist['Hasil_Prediksi'] == top_cat].shape[0]
        top_pct = (top_cat_count / total_data) * 100

    def make_kpi_cell(title, value, footer, bg_color):
        return Table([
        [Paragraph(title, ParagraphStyle('kpi_t', fontSize=9, textColor=rl_colors.white, alignment=1))],
        [Paragraph(value, ParagraphStyle('kpi_v', fontSize=22, fontName='Helvetica-Bold', textColor=rl_colors.white, alignment=1, leading=26))],
        [Paragraph(footer, ParagraphStyle('kpi_f', fontSize=8, textColor=rl_colors.whitesmoke, alignment=1))]
    ], colWidths=[2.1*inch], style=[
        ('BACKGROUND', (0,0), (-1,-1), bg_color),

        # Padding agar teks tidak terlalu mepet
        ('TOPPADDING', (0,0), (-1,-1), 15),
        ('BOTTOMPADDING', (0,0), (-1,-1), 15),

        # Posisi teks tetap di tengah
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ])


    kpi1 = make_kpi_cell("TOTAL PELANGGAN", f"{total_data}", "Data Terekam", COLOR_ACCENT)
    kpi2 = make_kpi_cell("MINAT TERTINGGI", top_cat.upper(), f"{top_pct:.1f}% Dominasi", COLOR_SUCCESS)
    kpi3 = make_kpi_cell("STATUS PERFORMA", "STABIL", "Analisis Sistem", COLOR_WARNING)

    t_kpi = Table([[kpi1, "", kpi2, "", kpi3]], colWidths=[2.1*inch, 0.15*inch, 2.1*inch, 0.15*inch, 2.1*inch])
    elements.append(t_kpi)
    elements.append(Spacer(1, 0.4 * inch))

    # -----------------------------------------------------------
    # 3. GRAFIK (CLEAN LOOK & BULAT SEMPURNA)
    # -----------------------------------------------------------
    elements.append(Paragraph("Visualisasi Data", style_h2))

    if not df_hist.empty:
        # A. DONUT CHART (Diperbaiki)
        counts = df_hist['Hasil_Prediksi'].value_counts()
        colors_map = {'Ekonomis': '#3498DB', 'Produktivitas': '#2ECC71', 'Gaming': '#E74C3C'}
        chart_colors = [colors_map.get(x, '#95a5a6') for x in counts.index]

        fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
        
        # width=0.6: Mempertebal donat (memperkecil lubang) agar teks muat
        # pctdistance=0.75: Posisi teks persentase di tengah area berwarna
        wedges, texts, autotexts = ax1.pie(
            counts, labels=counts.index, autopct='%1.0f%%', startangle=90, 
            colors=chart_colors, pctdistance=0.75,
            wedgeprops=dict(width=0.6, edgecolor='w') 
        )
        
        # MEMBUAT BULAT SEMPURNA (TIDAK LONJONG)
        ax1.axis('equal') 
        
        plt.setp(autotexts, size=9, weight="bold", color="white")
        plt.setp(texts, size=9)
        ax1.set_title("Distribusi Kategori", fontsize=10, pad=10, color='#2C3E50', fontweight='bold')
        
        img_buf1 = io.BytesIO()
        plt.savefig(img_buf1, format='png', dpi=150, bbox_inches='tight')
        img_buf1.seek(0)
        rl_img1 = RLImage(img_buf1, width=3*inch, height=3*inch)
        plt.close(fig1)

        # B. BAR CHART (Warna Lebih Menarik)
        df_hist['Tgl_Saja'] = pd.to_datetime(df_hist['Tanggal']).dt.date
        daily_counts = df_hist['Tgl_Saja'].value_counts().sort_index().tail(5)
        
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
        
        # Menggunakan warna berbeda tiap batang atau warna aksen cerah
        bar_colors = ['#3498DB', '#2ECC71', '#F1C40F', '#E67E22', '#E74C3C']
        # Jika data kurang dari 5, ambil warna secukupnya, jika lebih, ulang warnanya
        current_colors = bar_colors[:len(daily_counts)] if len(daily_counts) <= 5 else (bar_colors * ((len(daily_counts)//5)+1))[:len(daily_counts)]
        
        bars = ax2.bar(daily_counts.index.astype(str), daily_counts.values, color=current_colors, width=0.6)
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax2.xaxis.grid(False)
        
        ax2.set_title("Tren 5 Hari Terakhir", fontsize=10, pad=10, color='#2C3E50', fontweight='bold')
        plt.xticks(fontsize=8, rotation=0)
        plt.yticks(fontsize=8)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

        img_buf2 = io.BytesIO()
        plt.savefig(img_buf2, format='png', dpi=150, bbox_inches='tight')
        img_buf2.seek(0)
        rl_img2 = RLImage(img_buf2, width=3.8*inch, height=3*inch)
        plt.close(fig2)

        t_chart = Table([[rl_img1, rl_img2]], colWidths=[3.5*inch, 3.5*inch])
        t_chart.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.append(t_chart)
    
    elements.append(Spacer(1, 0.2 * inch))

    # -----------------------------------------------------------
    # 4. TABEL DATA (DENGAN NAMA KOLOM LENGKAP)
    # -----------------------------------------------------------
    elements.append(Paragraph("Aktivitas Transaksi Terbaru", style_h2))
    
    # Header Tabel yang Diminta
    data_table = [['TANGGAL', 'USIA', 'PEKERJAAN', 'HASIL AI', 'REKOMENDASI']]
    
    if not df_hist.empty:
        last_10 = df_hist.tail(10).sort_values(by='Tanggal', ascending=False)
        map_usia = {0: "Remaja", 1: "Dewasa", 2: "Lansia"}
        map_kerja = {0: "Pelajar", 1: "Pekerja", 2: "Tdk Kerja"}
        
        for idx, row in last_10.iterrows():
            # Konversi data numerik ke label jika perlu (handling error)
            usia_val = int(row['Usia']) if str(row['Usia']).isdigit() else row['Usia']
            usia_txt = map_usia.get(usia_val, str(usia_val))
            
            kerja_val = int(row['Pekerjaan']) if str(row['Pekerjaan']).isdigit() else row['Pekerjaan']
            kerja_txt = map_kerja.get(kerja_val, str(kerja_val))
            
            rek = "HP Entry Level"
            if row['Hasil_Prediksi'] == "Gaming": rek = "Poco F5 / Infinix GT"
            elif row['Hasil_Prediksi'] == "Produktivitas": rek = "Samsung A54 / Reno"
            elif row['Hasil_Prediksi'] == "Ekonomis": rek = "Infinix Smart / Redmi"
            
            # Format Tanggal Pendek
            tgl_short = str(row['Tanggal'])[0:16] # Ambil sampai jam menit

            data_table.append([tgl_short, usia_txt, kerja_txt, row['Hasil_Prediksi'], rek])
    
    # Styling Tabel
    col_widths = [1.4*inch, 1*inch, 1.2*inch, 1.4*inch, 2*inch]
    t_data = Table(data_table, colWidths=col_widths)
    
    t_data.setStyle(TableStyle([
        # Header Style
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('TOPPADDING', (0,0), (-1,0), 12),
        
        # Body Style
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('ALIGN', (0,1), (-1,-1), 'CENTER'),
        ('TEXTCOLOR', (0,1), (-1,-1), rl_colors.HexColor("#2c3e50")),
        ('BOTTOMPADDING', (0,1), (-1,-1), 8),
        ('TOPPADDING', (0,1), (-1,-1), 8),
        
        # Garis & Striping
        ('GRID', (0,0), (-1,-1), 0.5, rl_colors.HexColor("#bdc3c7")),
        ('ROWBACKGROUNDS', (1,0), (-1,-1), [rl_colors.white, COLOR_BG_GRAY]),
    ]))
    
    elements.append(t_data)
    
    # BUILD PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Load Model
model, eval_data_train = load_resources()

if model is None:
    st.error("❌ Model tidak ditemukan! Harap jalankan 'train_model.py' terlebih dahulu.")
    st.stop()

# ==============================================================================
# 3. MANAJEMEN SESSION STATE & NAVIGASI
# ==============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = ""

# STATE UNTUK RESET UPLOADER
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# ==============================================================================
# 4. HALAMAN-HALAMAN APLIKASI
# ==============================================================================

# --- A. HALAMAN HOME ---
def show_home():
    # 1. Judul & Deskripsi Utama (Hero Section)
    # Kita beri sedikit jarak atas agar tidak terlalu mepet
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='hero-text'>Sistem Cerdas MP3 Cellular Solok</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='sub-hero'>
            Temukan smartphone impian yang paling pas dengan gaya hidup dan kebutuhan Anda 
            menggunakan teknologi kecerdasan buatan.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

    # 2. FITUR CARDS (Tanpa Angka, Desain Box Modern)
    c1, c2, c3 = st.columns(3)

    # --- KARTU 1: ISI DATA ---
    with c1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📝</div>
                <div class="feature-title">Isi Preferensi</div>
                <div class="feature-desc">
                    Ceritakan sedikit tentang kebiasaan Anda. Apakah untuk gaming, kerja, atau fotografi?
                </div>
            </div>
        """, unsafe_allow_html=True)

    # --- KARTU 2: ANALISIS AI ---
    with c2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Analisis Cerdas</div>
                <div class="feature-desc">
                    Algoritma <i>Naive Bayes</i> kami akan memproses data Anda untuk menemukan pola terbaik.
                </div>
            </div>
        """, unsafe_allow_html=True)

    # --- KARTU 3: DAPAT REKOMENDASI ---
    with c3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">✨</div>
                <div class="feature-title">Hasil Akurat</div>
                <div class="feature-desc">
                    Dapatkan rekomendasi kategori dan tipe HP yang spesifik sesuai budget Anda.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 3. TOMBOL AKSI (CTA)
    # Menggunakan layout kolom agar tombol berada di tengah
    c_spacer_L, c_buttons, c_spacer_R = st.columns([1, 2, 1])
    
    with c_buttons:
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔐 Login Staff", use_container_width=True):
                go_to('login')
        
        with col_btn2:
            # Tombol Utama (Primary)
            if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
                go_to('prediksi_public')

# --- B. HALAMAN PREDIKSI PUBLIK ---
# --- B. HALAMAN PREDIKSI PUBLIK (LAYOUT BARU, TEKS ASLI) ---
def show_prediksi_public():
    # Header & Tombol Kembali
    col_back, col_title = st.columns([1, 10])
    with col_back:
        if st.button("⬅️", help="Kembali ke Home"): go_to('home')
    with col_title:
        st.markdown(
            "<h2 style='text-align: center; margin-top: -10px;'>🕵️ Analisis Kebutuhan Smartphone</h2>", 
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-bottom: 30px; font-size: 0.95rem;'>
            Jawablah pertanyaan di bawah ini sesuai kondisi Anda sebenarnya.
        </div>
        """, 
        unsafe_allow_html=True
    )

    # --- FUNGSI RESET ---
    def reset_callback():
        keys = ["k_usia", "k_kerja", "k_perf", "k_cam", "k_stor", "k_bat", "k_ergo"]
        for key in keys:
            st.session_state[key] = None
    
    # Mulai Form
    with st.form("form_prediksi_public"):
        # Layout 2 Kolom Besar (Kiri & Kanan)
        left_col, right_col = st.columns(2, gap="large")
        
        # --- KOLOM KIRI (PROFIL) ---
        with left_col:
            # Header Visual
            st.markdown('<div class="section-header">👤 Profil & Kebiasaan</div>', unsafe_allow_html=True)
            
            # 1. USIA
            opt_usia = ["A. 13 – 22 tahun", "B. 23 - 45 tahun", "C. 45 tahun keatas"]
            q1 = st.radio("1. Berapakah Usia anda?", opt_usia, key="k_usia", index=None)

            st.markdown("<br>", unsafe_allow_html=True)

            # 2. PEKERJAAN
            opt_kerja = ["A. Pelajar / Mahasiswa", "B. Karyawan / Wirausaha", "C. Tidak bekerja / IRT / Pensiunan"]
            q2 = st.radio("2. Apa Pekerjaan anda sekarang?", opt_kerja, key="k_kerja", index=None)

            st.markdown("<br>", unsafe_allow_html=True)

            # 3. PERFORMA
            opt_performa = [
                "A. Saya jarang buka aplikasi berat atau buka banyak aplikasi sekaligus",
                "B. Saya kadang buka aplikasi berat atau banyak aplikasi sekaligus",
                "C. Saya sering buka aplikasi berat seperti game atau banyak aplikasi sekaligus"
            ]
            q3 = st.radio("Pernyataan mana yang paling sesuai dengan  anda saat menggunakan HP?", opt_performa, key="k_perf", index=None)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 4. KAMERA
            opt_kamera = ["A. Seperlunya, yang penting bisa normal dipakai foto/video ", "B. Perlu hasil cukup bagus", "C. Perlu hasil kualitas terbaik"]
            q4 = st.radio("4. Seberapa perlu atau penting Kamera HP untuk anda? ", opt_kamera, key="k_cam", index=None)


        # --- KOLOM KANAN (TEKNIS) ---
        with right_col:
            # Header Visual
            st.markdown('<div class="section-header">⚙️ Spesifikasi Teknis</div>', unsafe_allow_html=True)
            
            # 5. STORAGE
            opt_storage = ["A. Kadang-kadang, seperlunya saja", "B. Cukup Sering", "C. Sangat Sering"]
            q5 = st.radio("5. Seberapa sering anda dalam menyimpan file, seperti (foto,video,dokument)?", opt_storage, key="k_stor", index=None)

            st.markdown("<br>", unsafe_allow_html=True)

            # 6. BATERAI
            opt_baterai = [
                "A. Tidak masalah isi daya >3x sehari",
                "B. Tidak masalah isi daya 2-3x sehari",
                "C. Ingin tahan seharian (1x isi daya)"
            ]
            q6 = st.radio("6. Pernyataan mana yang paling sesuai dengan kebiasaan anda dalam mengisi daya HP?", opt_baterai, key="k_bat", index=None)

            st.markdown("<br>", unsafe_allow_html=True)

            # 7. ERGONOMI
            opt_ergo = [
                "A. Saya kadang menggunakan HP dalam waktu lama, ukuran dan kenyamanan bukan hal utama.",
                "B. Saya Cukup Sering menggunakan HP dalam waktu lama, ukuran dan kenyamanan diperlukan.",
                "C. Saya Sangat sering menggunakan HP dalam waktu lama, ukuran dan kenyamanan sangat penting."
            ]
            q7 = st.radio("Pernyataan mana yang paling sesuai dengan kebiasaan Anda?", opt_ergo, key="k_ergo", index=None)


        st.markdown("<br>", unsafe_allow_html=True)

        # Tombol Aksi (Submit & Reset)
        c_space_L, c_btn_submit, c_btn_reset, c_space_R = st.columns([2, 4, 2, 2])
        
        with c_btn_submit:
            submitted = st.form_submit_button(
                "🔍 ANALISIS HASIL SEKARANG", 
                type="primary", 
                use_container_width=True
            )
        
        with c_btn_reset:
            reset_clicked = st.form_submit_button(
                "🔄 Ulangi / Reset", 
                type="secondary", 
                use_container_width=True, 
                on_click=reset_callback
            )

    # --- LOGIKA SAAT TOMBOL DITEKAN ---
    if submitted:
        if None in [q1, q2, q3, q4, q5, q6, q7]:
            st.error("⚠️ Mohon lengkapi semua pertanyaan di atas agar analisis akurat.")
        else:
            # 2. KONVERSI KE ANGKA
            usia = opt_usia.index(q1)
            pekerjaan = opt_kerja.index(q2)
            performa = opt_performa.index(q3)
            kamera = opt_kamera.index(q4)
            storage = opt_storage.index(q5)
            baterai = opt_baterai.index(q6)
            ergonomi = opt_ergo.index(q7)

            input_df = pd.DataFrame([[usia, pekerjaan, performa, kamera, storage, baterai, ergonomi]], 
                                    columns=["Usia", "Pekerjaan", "Performa", "Kamera", "Storage", "Baterai", "Ergonomi"])
            try:
                # PREDIKSI
                pred_val = model.predict(input_df)[0]
                pred_label = get_label_name(pred_val)
                
                # SETUP HASIL (Gambar & Teks)
                if pred_val == 2: # Gaming
                    img_url = "https://cdn-icons-png.flaticon.com/512/808/808439.png"
                    caption_text = "Gaming / High-Performance"
                    alert_type = "success"
                    rek_title = "Rekomendasi: Seri **Gaming / Flagship**"
                    rek_desc = "Anda membutuhkan HP dengan **Chipset Kencang**, Layar Responsif (High Refresh Rate), dan Sistem Pendingin yang baik."
                elif pred_val == 1: # Produktivitas
                    img_url = "https://cdn-icons-png.flaticon.com/512/3062/3062634.png"
                    caption_text = "Produktivitas / Professional"
                    alert_type = "info"
                    rek_title = "Rekomendasi: Seri **Mid-Range / All Rounder**"
                    rek_desc = "Anda membutuhkan HP seimbang (*All-Rounder*). Kamera jernih, Multitasking lancar (RAM 8GB+), dan Penyimpanan luas."
                else: # Ekonomis
                    img_url = "https://cdn-icons-png.flaticon.com/512/2503/2503504.png"
                    caption_text = "Ekonomis / Daily Driver"
                    alert_type = "warning"
                    rek_title = "Rekomendasi: Seri **Entry Level / Basic**"
                    rek_desc = "Anda membutuhkan HP yang **Efisien & Terjangkau**. Cukup untuk komunikasi (WhatsApp), Media Sosial ringan, dan browsing."


                # TAMPILAN HASIL (Layout Kartu Hasil)
                st.markdown("<br>", unsafe_allow_html=True)

                c_res_img, c_res_txt = st.columns([1, 3])
                
                with c_res_img:
                    st.image(img_url, width=120)
                    st.markdown(f"<div style='text-align:center; font-weight:bold; color:#555;'>{caption_text}</div>", unsafe_allow_html=True)

                with c_res_txt:
                    st.markdown(f"<h2 style='margin:0; color:#333;'>Hasil Analisis: <span style='color:#007bff'>{pred_label}</span></h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:1.1rem; margin-top:10px;'>{rek_desc}</p>", unsafe_allow_html=True)
                    
                    if alert_type == "success": st.success(rek_title)
                    elif alert_type == "info": st.info(rek_title)
                    else: st.warning(rek_title)

                st.markdown("</div>", unsafe_allow_html=True)
                
                # --- FITUR REKOMENDASI HP ---
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📱 Rekomendasi Smartphone Tersedia:")
                
                rec_list = get_phone_recommendations(pred_label)
                
                cols = st.columns(3)

                for i, hp in enumerate(rec_list):
                    with cols[i % 3]:

                        # CARD CONTAINER OPEN
                        st.markdown("<br>", unsafe_allow_html=True)

                        # ===== GAMBAR (Python, BUKAN HTML) =====
                        img_path = hp.get('img', '')

                        if img_path and os.path.exists(img_path):
                            st.image(img_path, width=120)
                        elif img_path and img_path.startswith("http"):
                            st.image(img_path, width=120)
                        else:
                            st.image("assets/default_phone.png", width=120)

                        # ===== TEKS =====
                        st.markdown(f"""
                            <div style="font-weight:bold; font-size:1rem; margin-bottom:5px;">
                                {hp.get('nama','-')}
                            </div>
                            <div style="color:#28a745; font-weight:bold; margin-bottom:5px;">
                                {hp.get('harga','-')}
                            </div>
                            <div style="font-size:0.8rem; color:#666; min-height:40px;">
                                {hp.get('desc','')}
                            </div>
                        """, unsafe_allow_html=True)

                        # CARD CONTAINER CLOSE
                        st.markdown("</div>", unsafe_allow_html=True)

                # Simpan ke Histori
                history_entry = {
                    "Tanggal": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "Usia": input_df.iloc[0,0], "Pekerjaan": input_df.iloc[0,1],
                    "Performa": input_df.iloc[0,2], "Kamera": input_df.iloc[0,3],
                    "Storage": input_df.iloc[0,4], "Baterai": input_df.iloc[0,5],
                    "Ergonomi": input_df.iloc[0,6], "Hasil_Prediksi": pred_label,
                }
                save_to_history(history_entry)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")
# --- C. HALAMAN LOGIN ---
def show_login():
    col_back, col_title = st.columns([1, 10])
    with col_back:
        if st.button("⬅️ Home"): go_to('home')
    # with col_title:
    #     st.markdown("<h1 style='text-align: center; margin-top: -15px;'>Login</h1>", unsafe_allow_html=True)
        
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        st.markdown("<h1 style='text-align: center; margin-top: -15px;'>Login</h1>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            c_kiri, c_tengah, c_kanan = st.columns([1, 1, 1])
            with c_tengah:
                submitted = st.form_submit_button("Masuk", use_container_width=True)
            
            if submitted:
                users = {"admin": "admin123", "pemilik": "pemilik123"}
                if username in users and users[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.user_role = "Admin" if username == "admin" else "Pemilik Toko"
                    go_to('admin_dashboard')
                else:
                    st.error("Username atau Password salah.")

# --- D. DASHBOARD ADMIN ---
def show_manage_data():
    st.header("🛠️ Manajemen Data HP (Admin)")
    st.markdown("Tambah / Edit / Hapus daftar HP per kategori. Gambar disimpan di `assets/reco_images/`.")

    # Load current data
    rec = load_recommendations()
    # Flatten to rows for editor
    rows = []
    for cat, items in rec.items():
        for it in items:
            rows.append({
                'category': cat,
                'name': it.get('nama',''),
                'price': it.get('harga',''),
                'desc': it.get('desc',''),
                'img': Path(it.get('img','')).name if it.get('img') else ''
            })

    import pandas as pd
    if len(rows) == 0:
        df = pd.DataFrame(columns=['category','name','price','desc','img'])
    else:
        df = pd.DataFrame(rows)

    st.markdown("### Editor (ubah langsung lalu klik 'Simpan Perubahan'):")
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="editor_reco"
    )

    st.markdown("---")
    c_left, c_right = st.columns([2,1])

    with c_left:
        st.markdown("### Tambah Rekomendasi Baru")
        new_cat = st.selectbox("Kategori", ["Ekonomis","Produktivitas","Gaming"], key="new_cat")
        new_name = st.text_input("Nama HP", key="new_name")
        new_price = st.text_input("Harga", key="new_price")
        new_desc = st.text_area("Deskripsi", key="new_desc")
        uploaded = st.file_uploader("Upload Gambar (jpg/png)", type=['png','jpg','jpeg'], key="reco_upload")

        if st.button("➕ Tambah"):
            img_fname = ""
            if uploaded:
                img_fname = save_uploaded_image(uploaded)
            rows.append({
                'category': new_cat,
                'name': new_name,
                'price': new_price,
                'desc': new_desc,
                'img': img_fname
            })
            save_recommendations_from_list(rows)
            st.rerun()

    with c_right:
        if st.button("💾 Simpan Perubahan Editor"):
            # validate and save edited dataframe
            cleaned = []
            import pandas as pd
            edf = edited.fillna('')
            for _, r in edf.iterrows():
                cleaned.append({
                    'category': r['category'],
                    'name': r['name'],
                    'price': r['price'],
                    'desc': r['desc'],
                    'img': r['img']
                })
            save_recommendations_from_list(cleaned)
            st.success("Perubahan disimpan.")
            st.rerun()

    st.markdown("### Preview saat ini")
    rec_after = load_recommendations()
    # show thumbnails
    for cat in ["Ekonomis","Produktivitas","Gaming"]:
        st.markdown(f"**{cat}**")
        items = rec_after.get(cat, [])
        cols = st.columns(3)
        for i, it in enumerate(items):
            c = cols[i % 3]
            with c:
                img_path = it.get('img') or ""
                if img_path and Path(img_path).exists():
                    st.image(img_path, width=140)
                else:
                    st.image("assets/default_phone.png", width=140)
                st.markdown(f"**{it.get('nama','-')}**")
                st.markdown(f"{it.get('harga','-')}")
                st.markdown(f"<div style='font-size:0.8rem; color:#666'>{it.get('desc','')}</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

def show_admin_dashboard():
    # 1. Pastikan state menu aktif sudah ada
    if 'active_admin_menu' not in st.session_state:
        st.session_state.active_admin_menu = "Evaluasi & Performa AI" # Default menu

    # Fungsi kecil untuk ganti menu saat tombol diklik
    def set_menu(menu_name):
        st.session_state.active_admin_menu = menu_name

    with st.sidebar:
        c_kiri, c_logo, c_kanan = st.columns([0.5, 3, 0.5])
        with c_logo:
            st.image("assets/logo.png", use_container_width=True)
        
        st.markdown(f"<div style='text-align: center;'>👤 Login: <b>{st.session_state.user_role}</b></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: grey; font-size: 0.8rem;'>NAVIGASI UTAMA</p>", unsafe_allow_html=True)
        
        
        # MENU 1: Evaluasi
        # Cek apakah ini menu yang aktif? Jika ya, warnanya Biru (primary), jika tidak abu (secondary)
        btn_type = "primary" if st.session_state.active_admin_menu == "Evaluasi & Performa AI" else "secondary"
        if st.button("📊 Evaluasi & Performa AI", type=btn_type, use_container_width=True):
            set_menu("Evaluasi & Performa AI")
            st.rerun() # Paksa refresh agar konten utama berubah

        # MENU 2: Data Histori
        btn_type = "primary" if st.session_state.active_admin_menu == "Data Histori" else "secondary"
        if st.button("💾 Data Histori", type=btn_type, use_container_width=True):
            set_menu("Data Histori")
            st.rerun()

        # MENU 3: Simulasi
        btn_type = "primary" if st.session_state.active_admin_menu == "Manajemen Data" else "secondary"
        if st.button("🛠️ Manajemen Data", type=btn_type, use_container_width=True):
            set_menu("Manajemen Data")
            st.rerun()


        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_role = ""
            go_to('home')
            st.rerun()

    # --- LOGIKA KONTEN UTAMA ---
    # Ambil nilai menu dari session state untuk menentukan konten mana yang muncul
    admin_menu = st.session_state.active_admin_menu
    # --- MENU 1: EVALUASI & PERFORMA AI ---
    if admin_menu == "Evaluasi & Performa AI":
        # Ganti st.title dan st.write dengan st.markdown HTML
        st.markdown("<h1 style='text-align: center;'>🧠 Evaluasi Kinerja Model AI</h1>", unsafe_allow_html=True)

        st.markdown("""
            <p style='text-align: center;'>
                Halaman ini menampilkan performa model pada data Testing dan memungkinkan Anda menguji data baru.
            </p>
    """, unsafe_allow_html=True)
        # =========================================================
        # BAGIAN 1: PERFORMA DATA TESTING (Kiri: CM, Kanan: Metrik)
        # =========================================================
        st.subheader("1. Performa Model")
        # st.caption("Bagian ini menggunakan **Data Testing** (Data ujian) untuk mengukur seberapa pintar model saat ini.")

        # Siapkan Data Testing untuk Confusion Matrix
        FILE_TESTING = "data/data_uji_final.csv"
        
        col_eval_kiri, col_eval_kanan = st.columns([3, 2])
        
        # --- KOLOM KIRI: CONFUSION MATRIX ---
        with col_eval_kiri:
            st.markdown('<div class="dashboard-card"><div class="card-title">Confusion Matrix</div>', unsafe_allow_html=True)
            
            if os.path.exists(FILE_TESTING):
                try:
                    df_test = pd.read_csv(FILE_TESTING)
                    required_cols = ["Usia", "Pekerjaan", "Performa", "Kamera", "Storage", "Baterai", "Ergonomi"]
                    
                    if all(col in df_test.columns for col in required_cols):
                        # Prediksi ulang untuk visualisasi
                        y_pred_test = model.predict(df_test[required_cols])
                        y_true_test = df_test['Label']
                        target_order = ["Ekonomis", "Produktivitas", "Gaming"]
                        
                        # Plot Heatmap
                        cm = confusion_matrix(y_true_test, y_pred_test)
                        fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=target_order, yticklabels=target_order, ax=ax_cm)
                        ax_cm.set_xlabel("Prediksi Model", fontsize=9)
                        ax_cm.set_ylabel("Data Aktual", fontsize=9)
                        ax_cm.tick_params(labelsize=8)
                        st.pyplot(fig_cm, use_container_width=True)
                        
                        st.caption("💡**Tips:** Kotak biru gelap diagonal adalah tebakan yang BENAR. Kotak di luar diagonal adalah KESALAHAN.")
                    else:
                        st.error("Format kolom CSV testing tidak sesuai.")
                except Exception as e:
                    st.error(f"Error load data testing: {e}")
            else:
                st.warning(f"⚠️ File '{FILE_TESTING}' tidak ditemukan. Confusion Matrix tidak dapat dibuat.")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # --- KOLOM KANAN: METRIK ANGKA ---
        with col_eval_kanan:
            st.markdown('<div class="dashboard-card"><div class="card-title">Skor Kinerja</div>', unsafe_allow_html=True)
            
            if eval_data_train:
                garis_tipis = "<hr style='margin: 3px 0; border: none; border-top: 1px solid #e0e0e0;'>"
                # Tampilkan Metrik Vertikal agar rapi
                st.metric("🎯 Akurasi (Ketepatan Total)", f"{eval_data_train['accuracy']*100:.2f}%", help="Persentase tebakan benar dari seluruh data uji.")
                st.markdown(garis_tipis, unsafe_allow_html=True)
                st.metric("✨ Precision (Ketelitian)", f"{eval_data_train['report']['macro avg']['precision']*100:.2f}%", help="Seberapa akurat saat model memprediksi kategori tertentu.")
                st.markdown(garis_tipis, unsafe_allow_html=True)
                st.metric("📡 Recall (Sensitivitas)", f"{eval_data_train['report']['macro avg']['recall']*100:.2f}%", help="Kemampuan model dalam menemukan kembali seluruh data yang sebenarnya positif.")
                st.markdown(garis_tipis, unsafe_allow_html=True)
                st.metric("⚖️ F1-Score (Keseimbangan)", f"{eval_data_train['report']['weighted avg']['f1-score']*100:.2f}%", help="Rata-rata harmonik antara presisi dan recall.")
            else:
                st.warning("Data hasil evaluasi JSON tidak ditemukan.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

        # =========================================================
        # BAGIAN 2: UPLOAD DATA BARU (Lebar Diperkecil)
        # =========================================================
        st.subheader("2. Uji Validasi Data Baru")

        c_space1, c_upload, c_space2 = st.columns([1, 2, 1])
        
        with c_upload:

            # State management untuk upload
            if 'df_evaluasi' not in st.session_state:
                st.session_state.df_evaluasi = None

            uploaded_file = st.file_uploader(
                "📂 Upload File CSV Disini", 
                type=["csv"], 
                key=f"uploader_{st.session_state.uploader_key}"
            )
            
            # Tombol Reset File
            if st.session_state.df_evaluasi is not None:
                if st.button("🗑️ Hapus / Ganti File", type="secondary"):
                    st.session_state.df_evaluasi = None
                    st.session_state.uploader_key += 1
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Proses File Upload
        if uploaded_file is not None:
            try:
                st.session_state.df_evaluasi = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error membaca file: {e}")

        

        # =========================================================
        # BAGIAN 3: DASHBOARD HASIL DATA BARU (RE-LAYOUT)
        # =========================================================
        if st.session_state.df_evaluasi is not None:
            df_upload = st.session_state.df_evaluasi
            required_cols = ["Usia", "Pekerjaan", "Performa", "Kamera", "Storage", "Baterai", "Ergonomi"]
            
            if all(col in df_upload.columns for col in required_cols):
                # Lakukan Prediksi
                if 'Prediksi_Sistem' not in df_upload.columns:
                    df_upload['Prediksi_Sistem'] = model.predict(df_upload[required_cols])
                    df_upload['Label_Prediksi'] = df_upload['Prediksi_Sistem'].apply(get_label_name)
                    st.session_state.df_evaluasi = df_upload
                
                target_order = ["Ekonomis", "Produktivitas", "Gaming"]
                total_data = len(df_upload)
                
                st.info(f"✅ Berhasil memproses **{total_data} data**. Berikut analisisnya:")
                
                # =========================================================
                # BARIS 1: A (Distribusi) & C (Validasi Akurasi)
                # =========================================================
                row1_col1, row1_col2 = st.columns(2)
                
                # --- [A] DISTRIBUSI HASIL (Kiri Atas) ---
               
                with row1_col1:
                    st.markdown('<div class="dashboard-card"><div class="card-title">Distribusi Hasil Prediksi</div>', unsafe_allow_html=True)
                    
                    # 1. Hitung Jumlah Data
                    counts = df_upload['Label_Prediksi'].value_counts()
                    counts = counts.reindex([t for t in target_order if t in counts.index])

                    if not counts.empty:
                        fig_v1, ax_v1 = plt.subplots(figsize=(5, 3))
                        
                        # Generate warna
                        colors = plt.cm.viridis([0.2, 0.5, 0.8]) 
                        if len(counts) == 2: colors = plt.cm.viridis([0.3, 0.7])
                        if len(counts) == 1: colors = plt.cm.viridis([0.5])

                        # --- FUNGSI CUSTOM UNTUK TEXT LABEL ---
                        # Ini yang membuat format: "15\n(45%)"
                        def make_autopct(values):
                            def my_autopct(pct):
                                total = sum(values)
                                val = int(round(pct*total/100.0))
                                return f"{val}\n({pct:.0f}%)"
                            return my_autopct

                        # 2. Plot Pie Chart
                        wedges, texts, autotexts = ax_v1.pie(
                            counts, 
                            labels=counts.index, 
                            autopct=make_autopct(counts), # Panggil fungsi custom tadi
                            startangle=90,     
                            colors=colors,
                            # LOGIKA POSISI:
                            # Lebar ring (width) = 0.4 (dari radius 0.6 s/d 1.0)
                            # Titik tengah ring = 0.8
                            pctdistance=0.70,  
                            textprops={'fontsize': 8},
                            wedgeprops=dict(width=0.6, edgecolor='w') 
                        )

                        # Styling Angka (Putih & Tebal)
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_weight('bold')
                            autotext.set_fontsize(7) # Ukuran font sedikit dibesarkan

                        # Styling Label di Luar (Nama Kategori)
                        for text in texts:
                            text.set_fontsize(8)
                            text.set_color('#333')

                        # 3. Info Total di Tengah Lobang
                        total_data = int(counts.sum())
                        ax_v1.text(0, 0, f"Total\n{total_data}", ha='center', va='center', fontsize=9, fontweight='bold', color='#333')

                        ax_v1.set_xlabel("")
                        ax_v1.axis('equal')  
                        
                        st.pyplot(fig_v1, use_container_width=True)
                    else:
                        st.info("Data kosong.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                # --- [C] VALIDASI AKURASI (Kanan Atas) ---
                with row1_col2:
                    # Cukup panggil class-nya, karena CSS-nya sudah kita perbaiki
                    st.markdown(
                        '<div class="dashboard-card"><div class="card-title">Validasi Akurasi</div>', 
                        unsafe_allow_html=True
                    )
                    if 'Label' in df_upload.columns:
                        acc_val = accuracy_score(df_upload['Label'], df_upload['Prediksi_Sistem'])
                        
                        # --- PENGGANTI st.metric AGAR RATA TENGAH ---
                        st.markdown(
                            f"""
                            <div style="text-align: center; margin-bottom: 15px;">
                                <p style="margin: 0; font-size: 0.9rem; color: var(--text-color);">Akurasi Data Ini</p>
                                <p style="margin: 0; font-size: 2.2rem; color: var(--text-color);">
                                    {acc_val*100:.2f}%
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        # ---------------------------------------------
                        
                        cm_val = confusion_matrix(df_upload['Label'], df_upload['Prediksi_Sistem'])
                        fig_v3, ax_v3 = plt.subplots(figsize=(5, 2.2))
                        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Oranges', xticklabels=target_order, yticklabels=target_order, ax=ax_v3)
                        ax_v3.tick_params(labelsize=7)
                        st.pyplot(fig_v3, use_container_width=True)
                    else:
                        st.info("Tidak ada kolom 'Label' (Kunci Jawaban) di file CSV. Confusion matrix tidak dapat dibuat.")
                        # Spacer agar tinggi kartu seimbang
                        st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

               # =========================================================
                # BARIS 2: PERBANDINGAN (Kiri) & TABEL (Kanan)
                # =========================================================
                row2_col1, row2_col2 = st.columns(2)

                # --- [B] DETAIL PERBANDINGAN FITUR (Kiri) ---
                with row2_col1:
                    st.markdown('<div class="dashboard-card"><div class="card-title">Detail Perbandingan Fitur</div>', unsafe_allow_html=True)
                    
                    # 1. Pilihan Fitur
                    default_ix = required_cols.index("Performa") if "Performa" in required_cols else 0
                    feat_compare = st.selectbox("🔍 Pilih Fitur:", required_cols, index=default_ix, key="sel_compare_feat")
                    
                    # 2. Siapkan Data
                    if feat_compare == "Usia":
                        val_map = {0: "Remaja", 1: "Dewasa", 2: "Lansia"}
                    elif feat_compare == "Pekerjaan":
                        val_map = {0: "Pelajar", 1: "Pekerja", 2: "Tdk Bekerja"}
                    else:
                        val_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
                    
                    df_viz = df_upload.copy()
                    target_col = 'Label_Asli' if 'Label_Asli' in df_viz.columns else 'Label_Prediksi'
                    df_viz[feat_compare] = df_viz[feat_compare].map(val_map)
                    
                    # Hitung Persentase
                    cross_tab = pd.crosstab(df_viz[target_col], df_viz[feat_compare], normalize='index') * 100
                    valid_order = [t for t in target_order if t in cross_tab.index]
                    cross_tab = cross_tab.reindex(valid_order)
                    
                    # 3. Visualisasi Stacked Bar Chart
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Tinggi sedikit disesuaikan agar sejajar dengan tabel
                    fig_comp, ax_comp = plt.subplots(figsize=(6, 3.2)) 
                    
                    cross_tab.plot(kind='barh', stacked=True, colormap='viridis', ax=ax_comp, width=0.7)
                    
                    ax_comp.set_xlabel("Persentase (%)", fontsize=8)
                    ax_comp.set_ylabel("")
                    ax_comp.set_xlim(0, 100)
                    ax_comp.legend(title="", bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3, fontsize=8, frameon=False)
                    ax_comp.tick_params(labelsize=8)
                    
                    for c in ax_comp.containers:
                        ax_comp.bar_label(c, fmt='%.0f%%', label_type='center', fontsize=8, color='white', weight='bold')

                    ax_comp.spines['top'].set_visible(False)
                    ax_comp.spines['right'].set_visible(False)
                    
                    st.pyplot(fig_comp, use_container_width=True)

                    # 4. KETERANGAN OTOMATIS (Versi Minimalis)
                    try:
                        keterangan_parts = []
                        for kategori in target_order:
                            if kategori in cross_tab.index:
                                pilihan_favorit = cross_tab.loc[kategori].idxmax()
                                persen_favorit = cross_tab.loc[kategori].max()
                                keterangan_parts.append(f"<b>{kategori}</b>: {persen_favorit:.0f}% pilih {pilihan_favorit}")
                        
                        if keterangan_parts:
                            # Gabungkan dengan pemisah " • " agar satu baris (hemat tempat)
                            pesan_gabung = " &bull; ".join(keterangan_parts)
                            
                            st.markdown(
                                f"<div style='margin-top: 5px; font-size: 0.8rem; color: var(--text-color)'>💡 {pesan_gabung}</div>", 
                                unsafe_allow_html=True
                            )
                        else:
                            st.caption("Data belum cukup.")

                    except Exception as e:
                        st.caption("")
                    
                    st.markdown('</div>', unsafe_allow_html=True)


                # --- [C] DETAIL DATA TABEL (Kanan) ---
                with row2_col2:
                    st.markdown('<div class="dashboard-card"><div class="card-title">Tabel Detail Data</div>', unsafe_allow_html=True)
                    
                    # Cek apakah ada kolom Label (Kunci Jawaban)
                    if 'Label' in df_upload.columns:
                        # 1. Buat Status (Benar/Salah)
                        df_upload['Status'] = df_upload.apply(lambda x: "✅" if x['Label'] == x['Prediksi_Sistem'] else "❌", axis=1)
                        # 2. Buat Label Asli jadi Teks
                        df_upload['Label_Asli'] = df_upload['Label'].apply(get_label_name)
                        view_cols_val = required_cols + ['Label_Asli', 'Label_Prediksi', 'Status']
                    else:
                        view_cols_val = required_cols + ['Label_Prediksi']

                    # Tampilkan Dataframe
                    # Height diatur agar kurang lebih sama tingginya dengan kartu di sebelah kiri
                    st.dataframe(
                        df_upload[view_cols_val], 
                        height=420, 
                        use_container_width=True, 
                        hide_index=True
                    )
                    
                    if 'Label' in df_upload.columns:
                        pesan_tabel = "📝 <b>Catatan:</b> Simbol ✅ menandakan prediksi <b>Akurat</b>, sedangkan ❌ menandakan <b>Meleset</b>."
                    else:
                        pesan_tabel = f"📝 Menampilkan <b>{len(df_upload)} baris data</b>. Kolom 'Status' tidak tersedia karena file ini tidak memiliki Kunci Jawaban."
                    st.markdown(
                        f"""
                        <div style='
                            font-size: 0.8rem; 
                            color: var(--text-color); 
                            opacity: 0.8; 
                            margin-top: 5px;'>
                            {pesan_tabel}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

    # --- MENU 2: DATA HISTORI ---
    elif admin_menu == "Data Histori":
        st.title("💾 Data Histori Transaksi")
        st.write("Kelola data riwayat prediksi pelanggan di sini.")

        # --- [PERBAIKAN 1] INISIALISASI SESSION STATE DI PALING ATAS (WAJIB) ---
        # Letakkan ini di luar blok try/if file exists agar selalu aman
        if "pdf_bytes" not in st.session_state:
            st.session_state["pdf_bytes"] = None
        if "pdf_data_len" not in st.session_state:
            st.session_state["pdf_data_len"] = -1

        if os.path.exists("history_data.csv"):
            try:
                # 1. Baca Data Asli
                df_hist = pd.read_csv("history_data.csv")
                
                # 2. Tambahkan Kolom Sementara 'Hapus'
                if "Pilih Hapus" not in df_hist.columns:
                    df_hist.insert(0, "Pilih Hapus", False)
                
                # 3. Tampilkan Data Editor
                st.info("Centang kotak pada baris yang ingin dihapus, lalu klik tombol merah di bawah.")
                
                edited_df = st.data_editor(
                    df_hist,
                    column_config={
                        "Pilih Hapus": st.column_config.CheckboxColumn(
                            "Hapus?",
                            help="Centang untuk menghapus data ini",
                            default=False,
                        )
                    },
                    disabled=df_hist.columns.drop("Pilih Hapus"),
                    hide_index=True,
                    use_container_width=True,
                    key="editor_histori"
                )
                
                # 4. Layout Tombol Aksi
                col_btn1, col_btn2 = st.columns([1, 4])
                
                # --- TOMBOL HAPUS DATA ---
                with col_btn1:
                    if st.button("🗑️ Hapus Data Terpilih", type="primary"):
                        # Ambil data yang TIDAK dicentang (yang dipertahankan)
                        df_baru = edited_df[edited_df["Pilih Hapus"] == False]
                        df_baru = df_baru.drop(columns=["Pilih Hapus"])
                        
                        # Simpan CSV Baru
                        df_baru.to_csv("history_data.csv", index=False)
                        
                        # --- [PERBAIKAN 2] RESET STATE (JANGAN GUNAKAN DEL) ---
                        # Reset nilai agar PDF dibuat ulang dengan data baru
                        st.session_state["pdf_bytes"] = None
                        st.session_state["pdf_data_len"] = -1
                        
                        st.success("✅ Data berhasil dihapus!")
                        st.rerun()
                
                # --- AREA DOWNLOAD ---
                with col_btn2:
                    c_csv, c_pdf = st.columns(2)
                    
                    # A. DOWNLOAD CSV
                    with c_csv:
                        csv_data = df_hist.drop(columns=["Pilih Hapus"]).to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Unduh CSV (Raw)",
                            data=csv_data,
                            file_name="histori_transaksi.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="btn_dl_csv"
                        )
                    
                    # B. DOWNLOAD PDF (SATU TOMBOL & STABIL)
                    with c_pdf:
                        current_len = len(df_hist)

                        # Tombol eksplisit untuk membuat laporan PDF
                        if st.button("🔄 Buat / Perbarui Laporan PDF", key="btn_generate_pdf", use_container_width=True):
                            try:
                                # Generate PDF hanya saat tombol ditekan
                                buffer = create_pdf_report(df_hist.drop(columns=["Pilih Hapus"]))
                                st.session_state["pdf_bytes"] = buffer.getvalue()  # simpan sebagai bytes
                                st.session_state["pdf_data_len"] = current_len
                                # st.success("✅ Laporan PDF berhasil dibuat. Silakan unduh di bawah.")
                            except Exception as e:
                                st.error(f"Gagal membuat PDF: {e}")
                                st.session_state["pdf_bytes"] = None
                                st.session_state["pdf_data_len"] = -1

                        # Tampilkan tombol Download hanya jika PDF sudah dibuat dan jumlah data masih sama
                        if st.session_state.get("pdf_bytes") is not None and st.session_state.get("pdf_data_len") == current_len:
                            tgl_str = pd.Timestamp.now().strftime('%Y%m%d')
                            st.download_button(
                                label="",
                                data=st.session_state["pdf_bytes"],
                                file_name=f"Laporan_MP3Cell_{tgl_str}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="btn_dl_pdf_final"
                            )

            except pd.errors.ParserError:
                st.error("⚠️ Data histori korup/rusak.")
                if st.button("🗑️ Reset & Hapus Semua Data"):
                    os.remove("history_data.csv")
                    st.rerun()
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        else:
            st.info("Belum ada data histori tersimpan.")
    # --- MENU 3: SIMULASI PREDIKSI ---
    elif admin_menu == "Manajemen Data":
        show_manage_data()

# ==============================================================================
# 5. ROUTING
# ==============================================================================

# 1. Cek dulu status login
if st.session_state.logged_in:
    # Jika sudah login, paksa masuk ke Dashboard (kecuali user memang mau logout)
    # Ini mencegah user terlempar ke 'home' saat refresh
    if st.session_state.page == 'login': 
        st.session_state.page = 'admin_dashboard'

# 2. Jalankan Halaman
if st.session_state.page == 'home': 
    show_home()
elif st.session_state.page == 'prediksi_public': 
    show_prediksi_public()
elif st.session_state.page == 'login': 
    show_login()
elif st.session_state.page == 'admin_dashboard':
    # Double check keamanan
    if st.session_state.logged_in: 
        show_admin_dashboard()
    else:
        st.warning("Akses ditolak. Silakan Login kembali.")
        go_to('login')