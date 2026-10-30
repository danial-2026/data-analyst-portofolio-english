import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 1. KONFIGURASI PATH & DICTIONARY Leksikon
# =============================================================================
# Gunakan master social_clean.csv yang sudah menggabungkan seluruh corpus bersih
INPUT_CSV = Path(r"C:\Users\Danial MW\Documents\KAKANG MKS\LOMBA\CFP BPKH 2026\Data\3. audit_output\social_clean.csv") 
OUT_DIR = Path("annotation_phase")
OUT_DIR.mkdir(exist_ok=True)

# Leksikon Risiko (Risk Dimensions)
RISK_DICT = {
    "Governance": ["pansus", "kpk", "audit", "korupsi", "tata kelola", "pengawasan", "bpk", "dpr", "komisi viii", "penyelewengan"],
    "Financial": ["likuiditas", "investasi", "nilai manfaat", "bipih", "bpih", "biaya haji", "subsidi", "imbal hasil", "alokasi"],
    "Public_Trust": ["percaya", "transparan", "akuntabel", "kecewa", "protes", "tuding", "kritik", "reputasi", "adil", "publik"]
}

# Leksikon Relevansi
CORE_BPKH = ["bpkh", "badan pengelola keuangan haji", "dana haji", "keuangan haji"]
HAJJ_GOVERNANCE = ["kuota", "kemenag", "menteri agama", "penyelenggaraan haji", "kasus haji"]

def classify_risk(text):
    text = str(text).lower()
    risks = []
    for risk_type, keywords in RISK_DICT.items():
        if any(kw in text for kw in keywords):
            risks.append(risk_type)
    return ", ".join(risks) if risks else "General"

def classify_relevance(text):
    text = str(text).lower()
    if any(kw in text for kw in CORE_BPKH):
        return "TIER_1_DIRECT_BPKH"
    elif any(kw in text for kw in HAJJ_GOVERNANCE) and any(kw in text for kw in RISK_DICT["Governance"]):
        return "TIER_2_GOVERNANCE_CONTAGION"
    elif any(kw in text for kw in RISK_DICT["Public_Trust"]):
        return "TIER_3_PUBLIC_TRUST"
    else:
        return "IRRELEVANT"

# =============================================================================
# 2. MAIN EXECUTION
# =============================================================================
def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] File {INPUT_CSV} tidak ditemukan. Pastikan file berada di direktori kerja.")
        return

    print("="*80)
    print("MEMULAI FASE 2: RELEVANCE GATE & RISK CLASSIFICATION")
    print("="*80)

    # Membaca file master corpus bersih dengan delimiter ';'
    df = pd.read_csv(INPUT_CSV, sep=';', engine='python', on_bad_lines='skip')
    
    # Penanganan kolom teks dinamis (berita menggunakan clean_content, sosmed menggunakan clean_text)
    def extract_text(row):
        c_content = row.get('clean_content')
        if pd.notna(c_content) and str(c_content).strip() != '':
            return str(c_content)
        c_text = row.get('clean_text')
        if pd.notna(c_text) and str(c_text).strip() != '':
            return str(c_text)
        return ""

    print("Mengekstrak dan memproses teks korpus...")
    df['eval_text'] = df.apply(extract_text, axis=1)

    # 2. Automated Labeling Berdasarkan Struktur Leksikon Baru
    print("Menjalankan klasifikasi leksikal otomatis...")
    df['auto_relevance'] = df['eval_text'].apply(classify_relevance)
    df['auto_risk_dimension'] = df['eval_text'].apply(classify_risk)

    # 3. Simpan Master Auto-Annotated
    master_out = OUT_DIR / "master_auto_annotated.csv"
    df.to_csv(master_out, sep=';', index=False, encoding='utf-8-sig')
    
    # 4. Buat Sampel Anotasi Manusia (Target 300 data untuk Cohen's Kappa)
    sample_size = min(300, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    
    # Tentukan ID unik (toleransi kolom berita vs sosmed)
    df_sample['identifier'] = df_sample.apply(
        lambda r: r.get('article_id') if pd.notna(r.get('article_id')) else r.get('post_id'), axis=1
    )
    
    columns_for_annotation = [
        'identifier', 'eval_text', 
        'auto_relevance', 'auto_risk_dimension',
        'Annotator_1_Relevance', 'Annotator_1_Risk', 'Annotator_1_Sentiment',
        'Annotator_2_Relevance', 'Annotator_2_Risk', 'Annotator_2_Sentiment'
    ]
    
    df_annotation_template = df_sample[['identifier', 'eval_text', 'auto_relevance', 'auto_risk_dimension']].copy()
    for col in ['Annotator_1_Relevance', 'Annotator_1_Risk', 'Annotator_1_Sentiment',
                'Annotator_2_Relevance', 'Annotator_2_Risk', 'Annotator_2_Sentiment']:
        df_annotation_template[col] = "" # Kolom kosong untuk diisi manual oleh Annotator 1 & 2

    sample_out = OUT_DIR / "human_annotation_sample.csv"
    df_annotation_template.to_csv(sample_out, sep=';', index=False, encoding='utf-8-sig')

    print(f"\nSelesai! Hasil tersimpan di folder: {OUT_DIR}")
    print(f" -> Master Dataset (Auto Labeled): {len(df)} baris")
    print(f" -> Template Anotasi Manusia      : {sample_size} baris siap dievaluasi.")
    print("="*80)

if __name__ == "__main__":
    main()