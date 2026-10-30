# -*- coding: utf-8 -*-
"""
BPKH Governance Contagion — Robustness / Sensitivity Analysis (NEWS ONLY)

Purpose
-------
Assess whether the monthly EWS signal is robust to reasonable perturbations of:
1) the semantic similarity threshold; and
2) EWS weights for SEI / NCI / Anomaly.

This is a robustness/sensitivity analysis, not a ground-truth accuracy test.
No F1, accuracy, or predictive-validity claim is made.

Baseline monthly file
---------------------
The final analytical monthly file is expected to contain:
    year_month, NCI, Anomaly_Score, EWS
(and optionally EWS_Level).

The script reads NCI directly. It does NOT reconstruct or invent NCI.
Because SEI is not stored as a standalone column in that file, a reference SEI
is recovered algebraically from the published baseline equation:
    EWS = 0.45*SEI + 0.35*NCI + 0.20*Anomaly
This algebraic recovery is only for audit/reference. Sensitivity scenarios
recompute SEI from the 561-news corpus for each semantic threshold.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# =============================================================================
# 1. PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

NEWS_PATH = Path(os.getenv(
    "BPKH_NEWS_CLEAN",
    str(PROJECT_ROOT / "Data" / "3. audit_output" / "news_clean.csv")
))

MONTHLY_PATH = Path(os.getenv(
    "BPKH_MONTHLY_FINAL",
    str(PROJECT_ROOT / "Data" / "Output Utama" / "9. monthly_nci_temporal_ews_gcr.csv")
))

OUTPUT_DIR = Path(os.getenv(
    "BPKH_OUTPUT_DIR",
    str(PROJECT_ROOT / "Data" / "Output Utama" / "robustness_sensitivity")
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. SENSITIVITY DESIGN
# =============================================================================
SEMANTIC_THRESHOLDS = [0.38, 0.40, 0.41, 0.42, 0.43, 0.44, 0.46]

# Each tuple is (SEI, NCI, Anomaly) and sums to 1.00.
WEIGHT_SETS = {
    "Baseline": (0.45, 0.35, 0.20),
    "W1": (0.40, 0.40, 0.20),
    "W2": (0.50, 0.30, 0.20),
    "W3": (0.45, 0.40, 0.15),
    "W4": (0.45, 0.30, 0.25),
    "W5": (0.50, 0.35, 0.15),
    "W6": (0.40, 0.35, 0.25),
}

SCENARIOS = [
    ("Baseline", 0.42, "Baseline"),
    ("S1", 0.38, "Baseline"),
    ("S2", 0.40, "Baseline"),
    ("S3", 0.41, "Baseline"),
    ("S4", 0.43, "Baseline"),
    ("S5", 0.44, "Baseline"),
    ("S6", 0.46, "Baseline"),
    ("S7", 0.42, "W1"),
    ("S8", 0.42, "W2"),
    ("S9", 0.42, "W3"),
    ("S10", 0.42, "W4"),
    ("S11", 0.42, "W5"),
    ("S12", 0.42, "W6"),
    ("S13", 0.40, "W1"),
    ("S14", 0.44, "W2"),
    ("S15", 0.40, "W4"),
    ("S16", 0.44, "W3"),
]
BASELINE_SCENARIO = "Baseline"

# =============================================================================
# 3. FROZEN PARAMETERS FROM FINAL NEWS-ONLY SEMANTIC SCRIPT
# =============================================================================
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MARGIN_THRESHOLD = 0.025
MAX_DIMENSIONS_PER_DOC = 4
CRISIS_GATE = 0.12
GENERIC_HAJJ_CUTOFF = 0.52
STRONG_SCORE_CUTOFF = 0.50

TAXONOMY = {
    "Quota_Governance": (
        "krisis atau kontroversi tata kelola kuota haji, pembagian kuota, "
        "alokasi kuota tambahan, kuota khusus dan reguler, penyimpangan "
        "pembagian kuota, kebijakan kuota yang dipersoalkan"
    ),
    "Corruption_Investigation": (
        "dugaan korupsi haji, korupsi kuota haji, suap, gratifikasi, "
        "penyelidikan atau penyidikan KPK, tersangka, penggeledahan, "
        "pemeriksaan terkait perkara, tindak pidana korupsi, penyalahgunaan "
        "wewenang, penyelewengan"
    ),
    "Governance_Oversight": (
        "pengawasan, audit, pemeriksaan, DPR, KPK, BPK, pansus, "
        "pertanggungjawaban pemerintah, kelemahan tata kelola, "
        "pengawasan kelembagaan dan kebijakan publik"
    ),
    "BPKH_Governance": (
        "tata kelola BPKH, Badan Pengelola Keuangan Haji, akuntabilitas "
        "BPKH, transparansi BPKH, pengawasan BPKH, Dewan Pengawas BPKH, "
        "Badan Pelaksana BPKH, pimpinan BPKH, kebijakan kelembagaan BPKH"
    ),
    "Hajj_Fund_Management": (
        "pengelolaan dana haji, dana setoran jamaah, keuangan haji, "
        "nilai manfaat, investasi dana haji, sukuk, penempatan dana, "
        "hasil investasi, pengelolaan keuangan oleh BPKH"
    ),
    "Public_Trust_Reputation": (
        "kepercayaan publik terhadap BPKH atau pengelolaan dana haji, "
        "reputasi institusi, persepsi negatif, kekhawatiran masyarakat, "
        "ketidakpercayaan, kritik publik, tuntutan akuntabilitas dan "
        "keterbukaan informasi"
    ),
    "Clarification_Response": (
        "klarifikasi BPKH, penjelasan BPKH, bantahan, tanggapan, "
        "respons terhadap tuduhan atau kritik, pernyataan resmi, "
        "BPKH menegaskan, langkah perbaikan tata kelola"
    ),
}

DIMENSIONS = list(TAXONOMY.keys())
EXTERNAL = ["Quota_Governance", "Corruption_Investigation", "Governance_Oversight"]
BPKH = ["BPKH_Governance", "Hajj_Fund_Management"]
TRUST = ["Public_Trust_Reputation"]
RESPONSE = "Clarification_Response"

MITIGATION_SEED = (
    "BPKH memperoleh penghargaan, kinerja investasi baik, opini WTP, "
    "pengelolaan dana aman, transparan, akuntabel, komitmen integritas, "
    "program pencegahan korupsi, edukasi, klarifikasi resmi, "
    "perbaikan tata kelola dan penguatan pengawasan"
)

GENERIC_HAJJ_SEED = (
    "ibadah haji, keberangkatan jamaah, kepulangan jamaah, layanan jamaah, "
    "hotel, transportasi, visa, kesehatan jamaah, cuaca, ritual, "
    "akomodasi dan informasi operasional haji"
)

# =============================================================================
# 4. HELPERS
# =============================================================================
def percentile_rank(series):
    return pd.Series(series, dtype=float).rank(method="average", pct=True)


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def read_csv_flexible(path, required_columns=None):
    """Read comma/semicolon CSV and clean BOM from headers."""
    last_error = None
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            df = clean_columns(df)
            if required_columns is None or set(required_columns).issubset(df.columns):
                return df, sep
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read {path}: {last_error}")


def load_news(path):
    df, _ = read_csv_flexible(path)
    text_candidates = ["clean_content", "raw_content", "content", "article_text", "text"]
    date_candidates = ["publish_date", "published_date", "date", "tanggal"]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if text_col is None or date_col is None:
        raise ValueError(
            "Could not identify news text/date columns. Available columns: "
            + str(df.columns.tolist())
        )

    out = pd.DataFrame({
        "text": df[text_col].fillna("").astype(str),
        "date": pd.to_datetime(df[date_col], errors="coerce", utc=True),
    })
    out = out[out["date"].notna()].copy()
    out = out[(out["date"] >= "2024-01-01") & (out["date"] <= "2026-09-03 23:59:59+00:00")]
    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    # Use tz-naive monthly labels to avoid the pandas timezone warning.
    out["year_month"] = out["date"].dt.tz_convert(None).dt.to_period("M").astype(str)
    return out


def compute_embeddings(news):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from exc

    print(f"[MODEL] Loading {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    seed_names = list(TAXONOMY.keys())
    seed_texts = [TAXONOMY[x] for x in seed_names]
    doc_emb = model.encode(
        news["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    seed_emb = model.encode(
        seed_texts + [GENERIC_HAJJ_SEED, MITIGATION_SEED],
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    sims = np.matmul(np.asarray(doc_emb), np.asarray(seed_emb).T)
    return pd.DataFrame(
        sims,
        columns=seed_names + ["Generic_Hajj", "Mitigation"],
    )


def semantic_monthly(raw, news, threshold):
    """Recompute semantic signals and monthly SEI for one threshold."""
    dim_raw = raw[DIMENSIONS].copy()
    adjusted = dim_raw.to_numpy() - 0.80 * raw["Generic_Hajj"].to_numpy()[:, None]
    adjusted = np.maximum(adjusted, 0.0)
    adj = pd.DataFrame(adjusted, columns=DIMENSIONS)

    # Apply the same document-level threshold logic as the final semantic script.
    for i in range(len(raw)):
        generic = float(raw.loc[i, "Generic_Hajj"])
        candidates = []
        for dim in DIMENSIONS:
            score = float(raw.loc[i, dim])
            adj_score = float(adj.loc[i, dim])
            if (
                score >= threshold
                and adj_score >= MARGIN_THRESHOLD
                and (generic < GENERIC_HAJJ_CUTOFF or score >= STRONG_SCORE_CUTOFF)
            ):
                candidates.append((dim, adj_score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        # Assignment list is not required downstream, but this reproduces the
        # max-four-dimensions constraint of the final semantic mapping.
        _ = candidates[:MAX_DIMENSIONS_PER_DOC]

    external = adj[EXTERNAL].max(axis=1).to_numpy()
    bpkh = adj[BPKH].max(axis=1).to_numpy()
    trust = adj[TRUST].max(axis=1).to_numpy()
    response = adj[RESPONSE].to_numpy()
    mitigation = raw["Mitigation"].to_numpy()

    corruption = adj["Corruption_Investigation"].to_numpy()
    quota = adj["Quota_Governance"].to_numpy()
    oversight = adj["Governance_Oversight"].to_numpy()

    crisis_validity = 0.45 * corruption + 0.40 * quota + 0.15 * oversight
    crisis_flag = ((corruption >= CRISIS_GATE) | (quota >= CRISIS_GATE)).astype(int)
    mitigation_dominant = ((mitigation > crisis_validity) & (response >= 0.10)).astype(int)

    contagion = np.cbrt(
        np.maximum(crisis_validity, 0)
        * np.maximum(bpkh, 0)
        * np.maximum(trust, 0)
    )
    contagion *= crisis_flag
    contagion *= np.where(mitigation_dominant == 1, 0.35, 1.0)

    tmp = pd.DataFrame({
        "year_month": news["year_month"],
        "external": crisis_validity,
        "bpkh": bpkh,
        "trust": trust,
        "crisis_rate": crisis_flag,
        "gci_doc": contagion,
    })

    monthly = tmp.groupby("year_month", sort=True).agg(
        external_mean=("external", "mean"),
        bpkh_mean=("bpkh", "mean"),
        trust_mean=("trust", "mean"),
        crisis_rate=("crisis_rate", "mean"),
        gci_doc_mean=("gci_doc", "mean"),
        volume=("gci_doc", "size"),
    ).reset_index()

    monthly["external_signal"] = percentile_rank(monthly["external_mean"])
    monthly["bpkh_signal"] = percentile_rank(monthly["bpkh_mean"])
    monthly["trust_signal"] = percentile_rank(monthly["trust_mean"])
    monthly["crisis_rate_signal"] = percentile_rank(monthly["crisis_rate"])
    monthly["GCI"] = np.cbrt(
        monthly["external_signal"]
        * monthly["bpkh_signal"]
        * monthly["trust_signal"]
    )
    monthly["GCI_3M_baseline"] = monthly["GCI"].rolling(3, min_periods=1).mean()
    monthly["GCI_momentum"] = (
        monthly["GCI"] - monthly["GCI_3M_baseline"]
    ).clip(lower=0)
    monthly["volume_signal"] = percentile_rank(monthly["volume"])
    monthly["SEI"] = (
        0.60 * monthly["GCI"]
        + 0.15 * monthly["GCI_momentum"]
        + 0.15 * monthly["volume_signal"]
        + 0.10 * monthly["crisis_rate_signal"]
    )
    return monthly


def load_baseline_monthly(path):
    required = ["year_month", "NCI", "Anomaly_Score", "EWS"]
    df, sep = read_csv_flexible(path, required_columns=required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Final monthly file is missing required columns: " + str(missing)
            + "\nAvailable: " + str(df.columns.tolist())
        )

    keep = required + (["EWS_Level"] if "EWS_Level" in df.columns else [])
    out = df[keep].copy()
    out["year_month"] = out["year_month"].astype(str).str.strip()
    for col in ["NCI", "Anomaly_Score", "EWS"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required).sort_values("year_month").reset_index(drop=True)

    out["SEI_reference"] = (
        out["EWS"]
        - 0.35 * out["NCI"]
        - 0.20 * out["Anomaly_Score"]
    ) / 0.45
    out["SEI_reference"] = out["SEI_reference"].clip(0, 1)

    print(f"[LOAD] Monthly delimiter: {sep!r}")
    print(f"[LOAD] Monthly rows available: {len(out):,}")
    print("[LOAD] Monthly baseline components: SEI_reference, NCI, Anomaly_Score")
    print("[INFO] NCI is read directly from the final monthly analytical file; it is NOT reconstructed.")
    print("[INFO] SEI_reference is algebraically recovered from the published baseline EWS formula.")
    return out


def add_ews(monthly, w_sei, w_nci, w_anom):
    out = monthly.copy()
    out["EWS_Score"] = (
        w_sei * out["SEI"]
        + w_nci * out["NCI"]
        + w_anom * out["Anomaly_Score"]
    ).clip(0, 1)
    yellow = out["EWS_Score"].quantile(0.75)
    red = out["EWS_Score"].quantile(0.90)
    out["EWS_Level"] = np.where(
        out["EWS_Score"] >= red,
        "RED",
        np.where(out["EWS_Score"] >= yellow, "YELLOW", "GREEN"),
    )
    return out, float(yellow), float(red)


def main():
    if not NEWS_PATH.exists():
        raise FileNotFoundError(f"News corpus not found: {NEWS_PATH}")
    if not MONTHLY_PATH.exists():
        raise FileNotFoundError(f"Monthly baseline not found: {MONTHLY_PATH}")

    print("[LOAD] News corpus:", NEWS_PATH.resolve())
    news = load_news(NEWS_PATH)
    print(f"[LOAD] Final news documents available: {len(news):,}")
    if len(news) != 561:
        print(f"[WARN] Expected 561 final news documents; loaded {len(news)}.")

    print("[LOAD] Monthly analytical baseline:", MONTHLY_PATH.resolve())
    baseline = load_baseline_monthly(MONTHLY_PATH)

    raw = compute_embeddings(news)

    semantic_cache = {}
    for threshold in sorted({t for _, t, _ in SCENARIOS}):
        print(f"[SCENARIO] semantic threshold = {threshold:.2f}")
        semantic_cache[threshold] = semantic_monthly(raw, news, threshold)

    baseline_series = baseline.set_index("year_month")["EWS"]
    if "EWS_Level" in baseline.columns:
        baseline_levels = (
            baseline.set_index("year_month")["EWS_Level"]
            .astype(str)
            .str.upper()
        )
    else:
        q75 = baseline_series.quantile(0.75)
        q90 = baseline_series.quantile(0.90)
        baseline_levels = pd.Series(
            np.where(
                baseline_series >= q90,
                "RED",
                np.where(baseline_series >= q75, "YELLOW", "GREEN"),
            ),
            index=baseline_series.index,
        )

    rows = []
    score_store = baseline[["year_month", "EWS"]].copy()
    score_store = score_store.rename(columns={"EWS": "Baseline_EWS"})
    score_store["Baseline_Level"] = baseline_levels.reindex(
        score_store["year_month"]
    ).to_numpy()

    for scenario, threshold, weight_name in SCENARIOS:
        weights = WEIGHT_SETS[weight_name]
        scenario_monthly = semantic_cache[threshold]
        scenario_input = scenario_monthly.merge(
            baseline[["year_month", "NCI", "Anomaly_Score"]],
            on="year_month",
            how="inner",
        )
        result, yellow, red = add_ews(scenario_input, *weights)
        s = result.set_index("year_month")["EWS_Score"]
        levels = result.set_index("year_month")["EWS_Level"]
        common = baseline_series.index.intersection(s.index)
        if len(common) == 0:
            raise ValueError("No overlapping monthly observations between semantic and baseline data.")

        rho = spearmanr(
            baseline_series.loc[common],
            s.loc[common],
        ).statistic
        same_rate = float(
            (
                baseline_levels.loc[common].to_numpy()
                == levels.loc[common].to_numpy()
            ).mean()
        )

        rows.append({
            "Scenario": scenario,
            "Semantic threshold": threshold,
            "EWS weights (SEI/NCI/Anomaly)": (
                f"{weights[0]:.2f}/{weights[1]:.2f}/{weights[2]:.2f}"
            ),
            "Spearman rho": 1.0 if scenario == BASELINE_SCENARIO else float(rho),
            "Same-classification rate": 1.0 if scenario == BASELINE_SCENARIO else same_rate,
            "EWS_yellow_threshold": yellow,
            "EWS_red_threshold": red,
        })

        indexed = result.set_index("year_month")
        score_store[f"{scenario}_EWS"] = indexed.reindex(
            score_store["year_month"]
        )["EWS_Score"].to_numpy()
        score_store[f"{scenario}_Level"] = indexed.reindex(
            score_store["year_month"]
        )["EWS_Level"].to_numpy()

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "robustness_sensitivity_table.csv", index=False)
    score_store.to_csv(OUTPUT_DIR / "robustness_sensitivity_ews_scores.csv", index=False)

    md = table.copy()
    md["Spearman rho"] = md["Spearman rho"].map(lambda x: f"{x:.3f}")
    md["Same-classification rate"] = md["Same-classification rate"].map(lambda x: f"{100*x:.1f}%")
    md["Semantic threshold"] = md["Semantic threshold"].map(lambda x: f"{x:.2f}")
    md["EWS_yellow_threshold"] = md["EWS_yellow_threshold"].map(lambda x: f"{x:.3f}")
    md["EWS_red_threshold"] = md["EWS_red_threshold"].map(lambda x: f"{x:.3f}")
    def dataframe_to_markdown(df):
        cols = list(df.columns)

        header = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"

        rows = []
        for _, row in df.iterrows():
            values = []
            for value in row:
                if pd.isna(value):
                    values.append("")
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")

        return "\n".join([header, separator] + rows)

    md_text = dataframe_to_markdown(md)
    md_text = (
        "# Robustness Sensitivity Analysis\n\n"
        + md_text
        + "\n"
    )
    (OUTPUT_DIR / "robustness_sensitivity_table.md").write_text(
        md_text,
        encoding="utf-8",
    )

    stable = table[table["Scenario"] != BASELINE_SCENARIO]
    summary = [
        "BPKH Governance Contagion — Robustness / Sensitivity Analysis",
        "",
        f"News corpus: {len(news):,} documents",
        f"Monthly observations: {len(baseline):,}",
        f"Semantic thresholds tested: {', '.join(f'{x:.2f}' for x in sorted({t for _, t, _ in SCENARIOS}))}",
        f"Number of scenarios: {len(SCENARIOS)}",
        "NCI and Anomaly_Score are held at the supplied final monthly analytical baseline; semantic threshold changes recompute SEI from the 561-news corpus.",
        "This is a conditional robustness analysis, not a ground-truth accuracy validation.",
        "",
        f"Minimum non-baseline Spearman rho: {stable['Spearman rho'].min():.3f}",
        f"Minimum non-baseline same-classification rate: {stable['Same-classification rate'].min()*100:.1f}%",
        f"Median non-baseline Spearman rho: {stable['Spearman rho'].median():.3f}",
        f"Median non-baseline same-classification rate: {stable['Same-classification rate'].median()*100:.1f}%",
    ]
    (OUTPUT_DIR / "robustness_sensitivity_summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\n=== ROBUSTNESS TABLE ===")
    print(
        md[
            [
                "Scenario",
                "Semantic threshold",
                "EWS weights (SEI/NCI/Anomaly)",
                "Spearman rho",
                "Same-classification rate",
            ]
        ].to_string(index=False)
    )
    print("\n[OK] Outputs written to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
