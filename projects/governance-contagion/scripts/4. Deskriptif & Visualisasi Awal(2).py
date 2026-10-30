import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# 1. KONFIGURASI DIREKTORI & STYLE VISUALISASI
# =============================================================================
INPUT_DIR = Path("annotation_phase")
OUT_DIR = Path("descriptive_results")
OUT_DIR.mkdir(exist_ok=True)

# Menggunakan style visualisasi akademik yang bersih
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'figure.autolayout': True
})

# =============================================================================
# 2. EKSEKUSI ANALISIS DESKRIPTIF TERPISAH (NEWS VS SOCIAL)
# =============================================================================
def main():
    master_path = INPUT_DIR / r"C:\Users\Danial MW\Documents\KAKANG MKS\LOMBA\CFP BPKH 2026\Data\4. annotation_phase\master_auto_annotated.csv"
    if not master_path.exists():
        print(f"[ERROR] File {master_path} tidak ditemukan. Pastikan Fase 2 (auto-annotation) sudah dijalankan.")
        return

    print("="*80)
    print("MEMULAI FASE 3: DESCRIPTIVE ANALYSIS & GOVERNANCE DISCOURSE MAPPING")
    print("="*80)

    # Load master corpus berlabel otomatis
    df = pd.read_csv(master_path, sep=';', engine='python', on_bad_lines='skip')

    # Pemisahan Corpus berdasarkan platform/asal data
    # News corpus diidentifikasi dari keberadaan 'article_id' atau sumber berita, Social dari platform/post_id
    if 'platform' in df.columns:
        df_news = df[df['platform'].isna() | (df['platform'] == 'News')].copy()
        df_social = df[df['platform'].notna() & (df['platform'] != 'News')].copy()
    else:
        # Fallback jika kolom platform tidak ada secara eksplisit
        df_news = df[df['eval_text'].str.contains("Reporter:|Editor:", na=False)].copy()
        df_social = df[~df.index.isin(df_news.index)].copy()

    print(f" -> Ukuran News Corpus   : {len(df_news)} dokumen")
    print(f" -> Ukuran Social Corpus : {len(df_social)} dokumen/post")

    # =========================================================================
    # 3. ANALISIS STATISTIK & TABULASI SILANG
    # =========================================================================
    
    # A. Distribusi Relevance Label
    print("\n--- Distribusi Relevance Label (News) ---")
    print(df_news['auto_relevance'].value_counts())
    
    print("\n--- Distribusi Relevance Label (Social) ---")
    print(df_social['auto_relevance'].value_counts() if not df_social.empty else "Data Social Kosong")

    # Simpan ringkasan statistik ke file teks/CSV untuk lampiran paper
    summary_relevance = pd.DataFrame({
        'News_Count': df_news['auto_relevance'].value_counts(),
        'Social_Count': df_social['auto_relevance'].value_counts() if not df_social.empty else 0
    }).fillna(0)
    summary_relevance.to_csv(OUT_DIR / "summary_relevance_distribution.csv", sep=';', encoding='utf-8-sig')

    # =========================================================================
    # 4. GENERASI VISUALISASI AKADEMIK
    # =========================================================================
    
    # Visualisasi 1: Perbandingan Relevance Label Berita vs Media Sosial
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_data = pd.DataFrame({
        'News Corpus': df_news['auto_relevance'].value_counts(),
        'Social Corpus': df_social['auto_relevance'].value_counts() if not df_social.empty else pd.Series()
    }).fillna(0)
    
    plot_data.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', alpha=0.85)
    ax.set_title("Perbandingan Kategori Relevansi: Berita Arus Utama vs Media Sosial", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Kategori Relevansi (Relevance Gate Tier)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Jumlah Dokumen / Post", fontsize=10, fontweight='bold')
    plt.xticks(rotation=25, ha='right')
    plt.legend(title="Corpus Type")
    
    chart_path_1 = OUT_DIR / "comparison_relevance_news_vs_social.png"
    plt.savefig(chart_path_1, dpi=300)
    plt.close()
    print(f"\n[VISUALISASI 1] Tersimpan di: {chart_path_1}")

    # Visualisasi 2: Distribusi Dimensi Risiko (Risk Dimensions) pada News Corpus
    if 'auto_risk_dimension' in df_news.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        risk_counts = df_news['auto_risk_dimension'].value_counts().head(5)
        risk_counts.plot(kind='barh', ax=ax, color='#2b5c8f', edgecolor='black', alpha=0.85)
        ax.set_title("Dominasi Dimensi Risiko Tata Kelola pada Pemberitaan Berita", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Frekuensi Kemunculan", fontsize=10, fontweight='bold')
        ax.set_ylabel("Dimensi Risiko", fontsize=10, fontweight='bold')
        
        chart_path_2 = OUT_DIR / "news_risk_dimensions.png"
        plt.savefig(chart_path_2, dpi=300)
        plt.close()
        print(f"[VISUALISASI 2] Tersimpan di: {chart_path_2}")

    print("\n" + "="*80)
    print(f"FASE 3 SELESAI. Seluruh tabel ringkasan dan grafik visualisasi tersimpan di folder '{OUT_DIR}'.")
    print("="*80)

if __name__ == "__main__":
    main()