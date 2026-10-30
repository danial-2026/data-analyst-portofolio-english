import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# 1. KONFIGURASI DIREKTORI & STYLE
# =============================================================================
INPUT_CSV = Path(r"C:\Users\Danial MW\Documents\KAKANG MKS\LOMBA\CFP BPKH 2026\Data\4. annotation_phase\master_auto_annotated.csv")
OUT_DIR = Path("descriptive_results")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'figure.autolayout': True
})

# =============================================================================
# 2. MAIN EXECUTION: ANALISIS BASELINE TREN & INTENSITAS DISKURSUS
# =============================================================================
def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] File {INPUT_CSV} tidak ditemukan. Selesaikan Fase 2 terlebih dahulu.")
        return

    print("="*80)
    print("FASE 3.1: BASELINE KONDISI GOVERNANCE & TIME-SERIES INTENSITY MAPPING")
    print("="*80)

    df = pd.read_csv(INPUT_CSV, sep=';', engine='python', on_bad_lines='skip')

    # Pemisahan corpus
    if 'platform' in df.columns:
        df_news = df[df['platform'].isna() | (df['platform'] == 'News')].copy()
        df_social = df[df['platform'].notna() & (df['platform'] != 'News')].copy()
    else:
        df_news = df[df['eval_text'].str.contains("Reporter:|Editor:", na=False)].copy()
        df_social = df[~df.index.isin(df_news.index)].copy()

    # Normalisasi Kolom Tanggal
    # Berita menggunakan 'publish_date', Sosial menggunakan 'published_date'
    news_date_col = 'publish_date' if 'publish_date' in df_news.columns else 'published_date'
    social_date_col = 'published_date' if 'published_date' in df_social.columns else 'publish_date'

    df_news['dt'] = pd.to_datetime(df_news[news_date_col], errors='coerce', utc=True)
    df_social['dt'] = pd.to_datetime(df_social[social_date_col], errors='coerce', utc=True)

    # Ekstraksi Tahun-Bulan (YYYY-MM)
    df_news['year_month'] = df_news['dt'].dt.to_period('M').astype(str)
    df_social['year_month'] = df_social['dt'].dt.to_period('M').astype(str)

    # =========================================================================
    # A. ANALISIS VOLUME BULANAN (TIME-SERIES)
    # =========================================================================
    news_monthly = df_news['year_month'].value_counts().sort_index()
    social_monthly = df_social['year_month'].value_counts().sort_index() if not df_social.empty else pd.Series()

    monthly_trend = pd.DataFrame({
        'News_Volume': news_monthly,
        'Social_Volume': social_monthly
    }).fillna(0)
    
    monthly_trend.to_csv(OUT_DIR / "monthly_governance_risk_volume.csv", sep=';', encoding='utf-8-sig')
    print("\n--- Ringkasan Volume Bulanan Tersimpan ---")

    # Visualisasi Time-Series Tren Volume Diskursus (2024 - 2026)
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_trend.plot(kind='line', marker='o', ax=ax, linewidth=2.5, alpha=0.9)
    ax.set_title("Evolusi Bulanan Volume Diskursus Risiko Tata Kelola (2024–2026)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Periode (Tahun-Bulan)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Jumlah Dokumen / Postingan", fontsize=10, fontweight='bold')
    plt.xticks(rotation=45)
    ax.legend(["News Corpus", "Social Corpus (Instagram)"])
    
    chart_path_time = OUT_DIR / "monthly_governance_risk_volume_trend.png"
    plt.savefig(chart_path_time, dpi=300)
    plt.close()
    print(f"[VISUALISASI] Tren bulanan tersimpan di: {chart_path_time}")

    # =========================================================================
    # B. ANALISIS SUMBER BERITA TERATAS (SOURCE PROMINENCE)
    # =========================================================================
    if 'source_name' in df_news.columns:
        print("\n--- Top 5 Sumber Berita Paling Prominen ---")
        top_sources = df_news['source_name'].value_counts().head(5)
        print(top_sources)
        
        fig, ax = plt.subplots(figsize=(9, 5))
        top_sources.plot(kind='barh', ax=ax, color='#1f77b4', edgecolor='black', alpha=0.85)
        ax.set_title("Top 5 Sumber Media Arus Utama Pengawal Isu Tata Kelola BPKH", fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Jumlah Pemberitaan", fontsize=10, fontweight='bold')
        ax.set_ylabel("Sumber Media", fontsize=10, fontweight='bold')
        
        chart_path_source = OUT_DIR / "top_news_sources.png"
        plt.savefig(chart_path_source, dpi=300)
        plt.close()
        print(f"[VISUALISASI] Top sources tersimpan di: {chart_path_source}")

    # =========================================================================
    # C. ANALISIS ENGAGEMENT MEDIA SOSIAL
    # =========================================================================
    if not df_social.empty and 'engagement_total' in df_social.columns:
        print("\n--- Statistik Engagement Instagram ---")
        eng_summary = df_social['engagement_total'].describe()
        print(eng_summary)
        
        # Simpan ringkasan engagement
        df_social[['identifier' if 'identifier' in df_social.columns else 'post_id', 
                   'eval_text', 'like_count', 'comment_count', 'engagement_total']].sort_values(
            by='engagement_total', ascending=False
        ).head(10).to_csv(OUT_DIR / "top_engaged_social_posts.csv", sep=';', index=False, encoding='utf-8-sig')
        print("[INFO] Daftar 10 post Instagram dengan engagement tertinggi telah diexport.")

    print("\n" + "="*80)
    print("FASE 3.1 SELESAI. Baseline tren volume dan struktur sumber berhasil dipetakan.")
    print("="*80)

if __name__ == "__main__":
    main()