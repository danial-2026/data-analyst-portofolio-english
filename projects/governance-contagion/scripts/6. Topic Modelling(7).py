import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
INPUT_NEWS_CSV = Path(
    r"C:\Users\Danial MW\Documents\KAKANG MKS\LOMBA\CFP BPKH 2026"
    r"\Data\3. audit_output\news_clean.csv"
)

INPUT_SOCIAL_CSV = Path(
    r"C:\Users\Danial MW\Documents\KAKANG MKS\LOMBA\CFP BPKH 2026"
    r"\Data\3. audit_output\instagram_clean.csv"
)

OUT_DIR = Path("governance_contagion_v4_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("2024-01-01", tz="UTC")
END_DATE = pd.Timestamp("2026-09-03 23:59:59", tz="UTC")

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Semantic gates
SIM_THRESHOLD = 0.42
MARGIN_THRESHOLD = 0.025
MAX_DIMENSIONS_PER_DOC = 4

# Crisis validity
CRISIS_GATE = 0.12
STRONG_CRISIS_GATE = 0.20
BPKH_GATE = 0.10
TRUST_GATE = 0.08

# Event deduplication
EVENT_WINDOW_DAYS = 7
EVENT_SIM_THRESHOLD = 0.72

# EWS
YELLOW_Q = 0.75
RED_Q = 0.90
MIN_MONTHS_FOR_ANOMALY = 8

# =============================================================================
# 2. THEORETICAL TAXONOMY
# =============================================================================
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

EXTERNAL_DIMS = [
    "Quota_Governance",
    "Corruption_Investigation",
    "Governance_Oversight",
]

BPKH_DIMS = [
    "BPKH_Governance",
    "Hajj_Fund_Management",
]

TRUST_DIMS = [
    "Public_Trust_Reputation",
]

RESPONSE_DIM = "Clarification_Response"

# Explicit positive/mitigation content. These are NOT crisis evidence.
MITIGATION_SEED = (
    "BPKH memperoleh penghargaan, kinerja investasi baik, opini WTP, "
    "pengelolaan dana aman, transparan, akuntabel, komitmen integritas, "
    "program pencegahan korupsi, edukasi, klarifikasi resmi, "
    "perbaikan tata kelola dan penguatan pengawasan"
)

# Generic Hajj control
GENERIC_HAJJ_SEED = (
    "ibadah haji, keberangkatan jamaah, kepulangan jamaah, layanan jamaah, "
    "hotel, transportasi, visa, kesehatan jamaah, cuaca, ritual, "
    "akomodasi dan informasi operasional haji"
)

NOISE_PATTERNS = [
    r"\bberlangganan\b", r"\blogin\b", r"\bsign\s*in\b",
    r"\bsubscribe\b", r"\bnewsletter\b", r"\bklik\s+di\s+sini\b",
    r"\bbaca\s+juga\b", r"\bselengkapnya\b", r"\badvertorial\b",
]

# =============================================================================
# 3. HELPERS
# =============================================================================
def read_csv_flexible(path):
    try:
        return pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, sep=",", engine="python", on_bad_lines="skip")


def choose_column(df, candidates):
    available = [c for c in candidates if c in df.columns]
    if not available:
        return None
    scores = {}
    for c in available:
        scores[c] = (
            df[c].astype(str)
            .replace({"nan": "", "None": ""})
            .str.strip()
            .ne("")
            .sum()
        )
    return max(scores, key=scores.get)


def robust_parse_dates(series):
    raw = series.astype(str).str.strip()

    normal = pd.to_datetime(raw, errors="coerce", utc=True)
    dayfirst = pd.to_datetime(raw, errors="coerce", dayfirst=True, utc=True)
    parsed = normal.fillna(dayfirst)

    # Epoch fallback
    numeric = pd.to_numeric(raw, errors="coerce")
    sec_mask = numeric.between(1e9, 2e9)
    ms_mask = numeric.between(1e12, 2e12)

    if sec_mask.any():
        parsed.loc[sec_mask] = pd.to_datetime(
            numeric.loc[sec_mask], unit="s", errors="coerce", utc=True
        )
    if ms_mask.any():
        parsed.loc[ms_mask] = pd.to_datetime(
            numeric.loc[ms_mask], unit="ms", errors="coerce", utc=True
        )

    return parsed


def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    for pat in NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.I)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def minmax_array(values):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def percentile_rank(series):
    s = pd.Series(series, dtype=float)
    if len(s) <= 1:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return s.rank(method="average", pct=True)


def safe_corr(a, b):
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    if len(a) < 4 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(a.corr(b))


# =============================================================================
# 4. LOAD CORPUS WITH AUDIT
# =============================================================================
def load_news():
    if not INPUT_NEWS_CSV.exists():
        return None

    raw = read_csv_flexible(INPUT_NEWS_CSV)
    text_col = choose_column(
        raw, ["clean_content", "raw_content", "content", "article_text", "text"]
    )
    date_col = choose_column(
        raw, ["publish_date", "published_date", "date", "tanggal"]
    )
    id_col = choose_column(raw, ["article_id", "id", "url", "article_url"])

    if not text_col or not date_col:
        raise ValueError(
            f"News text/date tidak ditemukan. text={text_col}, date={date_col}"
        )

    out = pd.DataFrame({
        "unified_id": (
            raw[id_col].astype(str)
            if id_col else np.arange(len(raw)).astype(str)
        ),
        "platform_source": "News",
        "platform": "News",
        "unified_text": raw[text_col].fillna("").astype(str),
        "unified_date": raw[date_col],
    })

    print(f"[LOAD] News: {len(out):,} | text={text_col} | date={date_col}")
    return out


def load_social():
    if not INPUT_SOCIAL_CSV.exists():
        return None

    raw = read_csv_flexible(INPUT_SOCIAL_CSV)
    text_col = choose_column(
        raw, ["clean_text", "clean_content", "raw_text", "caption", "text"]
    )
    date_col = choose_column(
        raw, ["published_date", "publish_date", "created_at", "date", "tanggal"]
    )
    id_col = choose_column(raw, ["post_id", "id", "url", "post_url"])
    platform_col = choose_column(raw, ["platform", "source", "platform_source"])

    if not text_col or not date_col:
        raise ValueError(
            f"Social text/date tidak ditemukan. text={text_col}, date={date_col}"
        )

    platform = (
        raw[platform_col].fillna("Social").astype(str)
        if platform_col else pd.Series(["Instagram"] * len(raw))
    )

    out = pd.DataFrame({
        "unified_id": (
            raw[id_col].astype(str)
            if id_col else np.arange(len(raw)).astype(str)
        ),
        "platform_source": "Social",
        "platform": platform,
        "unified_text": raw[text_col].fillna("").astype(str),
        "unified_date": raw[date_col],
    })

    print(f"[LOAD] Social: {len(out):,} | text={text_col} | date={date_col}")
    return out


def load_corpus():
    frames = []
    news = load_news()
    social = load_social()

    if news is not None:
        frames.append(news)
    if social is not None:
        frames.append(social)

    if not frames:
        raise FileNotFoundError("Tidak ada corpus yang dapat dibaca.")

    df = pd.concat(frames, ignore_index=True)
    df["_parsed_date"] = robust_parse_dates(df["unified_date"])
    df["clean_text"] = df["unified_text"].map(clean_text)
    df["word_count"] = df["clean_text"].str.split().str.len()

    print("\n[FILTER AUDIT]")
    for source, g in df.groupby("platform_source"):
        valid_date = g["_parsed_date"].notna()
        in_period = (
            (g["_parsed_date"] >= START_DATE)
            & (g["_parsed_date"] <= END_DATE)
        )
        valid_text = g["word_count"] >= 8
        print(
            f"  {source}: total={len(g):,} | "
            f"valid_date={int(valid_date.sum()):,} | "
            f"in_period={int(in_period.sum()):,} | "
            f"valid_text={int(valid_text.sum()):,}"
        )

    df = df[
        (df["_parsed_date"] >= START_DATE)
        & (df["_parsed_date"] <= END_DATE)
        & (df["word_count"] >= 8)
    ].copy()

    df = df.drop_duplicates(
        subset=["unified_id", "platform_source"], keep="first"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("Corpus kosong setelah filtering.")

    df["year_month"] = df["_parsed_date"].dt.strftime("%Y-%m")

    print(f"\n[LOAD] Final documents: {len(df):,}")
    print(df["platform"].value_counts(dropna=False).to_string())

    return df


# =============================================================================
# 5. SEMANTIC SCORING
# =============================================================================
def semantic_mapping(df):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install terlebih dahulu: pip install sentence-transformers"
        ) from exc

    print(f"\n[MODEL] {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    seed_names = list(TAXONOMY.keys())
    seed_texts = [TAXONOMY[x] for x in seed_names]

    print(f"[EMBED] Documents: {len(df):,}")
    doc_emb = model.encode(
        df["clean_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    all_seeds = seed_texts + [GENERIC_HAJJ_SEED, MITIGATION_SEED]
    seed_emb = model.encode(
        all_seeds,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    sims = np.matmul(np.asarray(doc_emb), np.asarray(seed_emb).T)

    raw = pd.DataFrame(
        sims,
        columns=seed_names + ["Generic_Hajj", "Mitigation"],
        index=df.index,
    )

    # Generic-Hajj adjusted scores.
    adjusted = raw[DIMENSIONS].to_numpy() - (
        0.80 * raw["Generic_Hajj"].to_numpy()[:, None]
    )
    adjusted = np.maximum(adjusted, 0.0)

    adj = pd.DataFrame(
        adjusted, columns=DIMENSIONS, index=df.index
    )

    # Multi-label dimensions.
    assignments = []
    for idx in df.index:
        candidates = []
        generic = float(raw.loc[idx, "Generic_Hajj"])

        for dim in DIMENSIONS:
            score = float(raw.loc[idx, dim])
            adj_score = float(adj.loc[idx, dim])

            # Strong evidence can overcome generic Hajj similarity.
            if (
                score >= SIM_THRESHOLD
                and adj_score >= MARGIN_THRESHOLD
                and (
                    generic < 0.52
                    or score >= 0.50
                )
            ):
                candidates.append((dim, adj_score))

        candidates.sort(key=lambda z: z[1], reverse=True)
        assignments.append(candidates[:MAX_DIMENSIONS_PER_DOC])

    df["Risk_Dimensions"] = [
        ";".join(x[0] for x in a) if a else "Residual"
        for a in assignments
    ]

    df["Top_Risk_Dimension"] = [
        a[0][0] if a else "Residual" for a in assignments
    ]
    df["Top_Risk_Score"] = [
        a[0][1] if a else 0.0 for a in assignments
    ]

    for rank in range(3):
        df[f"Risk_{rank+1}"] = [
            a[rank][0] if len(a) > rank else "" for a in assignments
        ]
        df[f"Risk_{rank+1}_Score"] = [
            a[rank][1] if len(a) > rank else 0.0 for a in assignments
        ]

    # Axis scores.
    df["External_Crisis_Score"] = adj[EXTERNAL_DIMS].max(axis=1)
    df["BPKH_Exposure_Score"] = adj[BPKH_DIMS].max(axis=1)
    df["Public_Trust_Score"] = adj[TRUST_DIMS].max(axis=1)
    df["Response_Score"] = adj[RESPONSE_DIM]
    df["Mitigation_Score"] = raw["Mitigation"]

    # Crisis validity:
    # corruption/quota evidence is stronger than generic oversight.
    corruption = adj["Corruption_Investigation"]
    quota = adj["Quota_Governance"]
    oversight = adj["Governance_Oversight"]

    df["Crisis_Validity_Score"] = (
        0.45 * corruption
        + 0.40 * quota
        + 0.15 * oversight
    )

    # Explicit crisis evidence is required for contagion.
    df["Crisis_Event_Flag"] = (
        (corruption >= CRISIS_GATE)
        | (quota >= CRISIS_GATE)
    ).astype(int)

    df["Strong_Crisis_Event_Flag"] = (
        (corruption >= STRONG_CRISIS_GATE)
        | (quota >= STRONG_CRISIS_GATE)
    ).astype(int)

    # Positive/mitigation communication should not inflate crisis.
    df["Mitigation_Dominant_Flag"] = (
        (df["Mitigation_Score"] > df["Crisis_Validity_Score"])
        & (df["Response_Score"] >= 0.10)
    ).astype(int)

    # Event-aware document contagion.
    external = np.maximum(df["Crisis_Validity_Score"].to_numpy(), 0)
    bpkh = np.maximum(df["BPKH_Exposure_Score"].to_numpy(), 0)
    trust = np.maximum(df["Public_Trust_Score"].to_numpy(), 0)

    base_contagion = np.cbrt(external * bpkh * trust)

    # Suppress documents without crisis evidence and mitigation-dominant stories.
    validity_gate = np.where(
        df["Crisis_Event_Flag"].to_numpy() == 1, 1.0, 0.0
    )
    mitigation_penalty = np.where(
        df["Mitigation_Dominant_Flag"].to_numpy() == 1, 0.35, 1.0
    )

    df["Document_Contagion_Score"] = (
        base_contagion * validity_gate * mitigation_penalty
    )

    # Three interpretable states.
    state = np.full(len(df), "S0_No_Contagion", dtype=object)

    crisis_mask = df["Crisis_Event_Flag"].to_numpy() == 1
    exposure_mask = df["BPKH_Exposure_Score"].to_numpy() >= BPKH_GATE
    trust_mask = df["Public_Trust_Score"].to_numpy() >= TRUST_GATE

    state[crisis_mask] = "S1_External_Crisis"

    state[crisis_mask & exposure_mask] = "S2_BPKH_Exposure"

    state[
        crisis_mask & exposure_mask & trust_mask
    ] = "S3_Potential_Contagion"

    state[
        (state == "S3_Potential_Contagion")
        & (df["Mitigation_Dominant_Flag"].to_numpy() == 1)
    ] = "S2_Exposure_With_Response"

    df["Contagion_State"] = state

    return df, raw, adj


# =============================================================================
# 6. EVENT / EPISODE CLUSTERING
# =============================================================================
def assign_event_ids(df):
    """
    Groups highly similar crisis documents occurring within a short time window.
    This prevents multiple media reports of the same episode from becoming
    multiple independent crisis events.
    """
    crisis = df[df["Crisis_Event_Flag"] == 1].copy()

    if crisis.empty:
        df["event_id"] = ""
        df["event_size"] = 0
        return df

    crisis = crisis.sort_values("_parsed_date")
    event_ids = {}
    event_counter = 0
    previous_indices = []

    for idx, row in crisis.iterrows():
        assigned = None

        for prev_idx in previous_indices[-30:]:
            prev = crisis.loc[prev_idx]
            delta_days = abs(
                (row["_parsed_date"] - prev["_parsed_date"]).total_seconds()
            ) / 86400.0

            if delta_days > EVENT_WINDOW_DAYS:
                continue

            # Use risk-score profile similarity rather than full text.
            a = np.array([
                row["Crisis_Validity_Score"],
                row["BPKH_Exposure_Score"],
                row["Public_Trust_Score"],
            ]).reshape(1, -1)

            b = np.array([
                prev["Crisis_Validity_Score"],
                prev["BPKH_Exposure_Score"],
                prev["Public_Trust_Score"],
            ]).reshape(1, -1)

            profile_sim = cosine_similarity(a, b)[0, 0]

            if profile_sim >= EVENT_SIM_THRESHOLD:
                assigned = event_ids[prev_idx]
                break

        if assigned is None:
            event_counter += 1
            assigned = f"E{event_counter:03d}"

        event_ids[idx] = assigned
        previous_indices.append(idx)

    df["event_id"] = ""
    for idx, event_id in event_ids.items():
        df.loc[idx, "event_id"] = event_id

    sizes = pd.Series(event_ids).value_counts()
    df["event_size"] = df["event_id"].map(sizes).fillna(0).astype(int)

    return df


# =============================================================================
# 7. MONTHLY GCI / SEI / EWS
# =============================================================================
def build_monthly(df):
    rows = []

    for month, g in df.groupby("year_month", sort=True):
        news = g[g["platform_source"] == "News"]
        social = g[g["platform_source"] == "Social"]

        crisis_docs = g[g["Crisis_Event_Flag"] == 1]
        exposure_docs = g[
            (g["Crisis_Event_Flag"] == 1)
            & (g["BPKH_Exposure_Score"] >= BPKH_GATE)
        ]
        contagion_docs = g[
            g["Contagion_State"].isin(
                ["S3_Potential_Contagion", "S2_Exposure_With_Response"]
            )
        ]

        row = {
            "year_month": month,
            "total_documents": len(g),
            "news_documents": len(news),
            "social_documents": len(social),

            "crisis_documents": len(crisis_docs),
            "bpkh_exposure_documents": len(exposure_docs),
            "contagion_candidate_documents": len(contagion_docs),

            "external_mean": g["Crisis_Validity_Score"].mean(),
            "bpkh_mean": g["BPKH_Exposure_Score"].mean(),
            "trust_mean": g["Public_Trust_Score"].mean(),
            "response_mean": g["Response_Score"].mean(),
            "mitigation_mean": g["Mitigation_Score"].mean(),

            "crisis_rate": g["Crisis_Event_Flag"].mean(),
            "bpkh_exposure_rate": (
                (g["Crisis_Event_Flag"] == 1)
                & (g["BPKH_Exposure_Score"] >= BPKH_GATE)
            ).mean(),
            "trust_risk_rate": (
                (g["Crisis_Event_Flag"] == 1)
                & (g["BPKH_Exposure_Score"] >= BPKH_GATE)
                & (g["Public_Trust_Score"] >= TRUST_GATE)
            ).mean(),

            "document_gci_mean": g["Document_Contagion_Score"].mean(),
            "document_gci_max": g["Document_Contagion_Score"].max(),
        }

        # Unique crisis episodes.
        event_ids = crisis_docs["event_id"].replace("", np.nan).dropna()
        row["unique_crisis_events"] = event_ids.nunique()

        rows.append(row)

    monthly = pd.DataFrame(rows)

    # Percentile ranks are more stable than min-max for cross-month interpretation.
    for src, dest in [
        ("external_mean", "external_signal"),
        ("bpkh_mean", "bpkh_signal"),
        ("trust_mean", "trust_signal"),
        ("document_gci_mean", "gci_signal"),
        ("crisis_rate", "crisis_rate_signal"),
    ]:
        monthly[dest] = percentile_rank(monthly[src])

    # Final GCI: requires all three components.
    monthly["GCI"] = np.cbrt(
        monthly["external_signal"]
        * monthly["bpkh_signal"]
        * monthly["trust_signal"]
    )

    # Event density prevents three duplicated articles from appearing as
    # three independent events.
    monthly["event_density"] = (
        monthly["unique_crisis_events"]
        / monthly["total_documents"].replace(0, np.nan)
    ).fillna(0)

    monthly["volume_signal"] = percentile_rank(monthly["total_documents"])

    # Momentum = current GCI relative to recent 3-month baseline.
    monthly["GCI_3M_baseline"] = (
        monthly["GCI"].rolling(3, min_periods=1).mean()
    )
    monthly["GCI_momentum"] = (
        monthly["GCI"] - monthly["GCI_3M_baseline"]
    ).clip(lower=0)

    # Signal Escalation Index.
    monthly["SEI"] = (
        0.60 * monthly["GCI"]
        + 0.15 * monthly["GCI_momentum"]
        + 0.15 * monthly["volume_signal"]
        + 0.10 * monthly["crisis_rate_signal"]
    )

    return monthly


def anomaly_detection(monthly):
    monthly["Anomaly_Score"] = 0.0
    monthly["Anomaly_Flag"] = 0

    if len(monthly) < MIN_MONTHS_FOR_ANOMALY:
        return monthly

    features = [
        "GCI",
        "SEI",
        "external_signal",
        "bpkh_signal",
        "trust_signal",
        "volume_signal",
    ]

    X = monthly[features].fillna(0).to_numpy()

    contamination = min(
        0.20,
        max(0.05, 2.0 / max(len(monthly), 1))
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X)

    raw = -model.decision_function(X)
    monthly["Anomaly_Score"] = percentile_rank(raw)

    cutoff = monthly["Anomaly_Score"].quantile(0.90)
    monthly["Anomaly_Flag"] = (
        monthly["Anomaly_Score"] >= cutoff
    ).astype(int)

    return monthly


def classify_ews(monthly):
    yellow = monthly["SEI"].quantile(YELLOW_Q)
    red = monthly["SEI"].quantile(RED_Q)

    def level(x):
        if x >= red:
            return "RED"
        if x >= yellow:
            return "YELLOW"
        return "GREEN"

    monthly["EWS_Level"] = monthly["SEI"].map(level)

    # Response coverage is a descriptive ratio of response signal
    # relative to the combined response + external-crisis signal.
    # It is bounded in [0, 1] and is NOT part of GCI.
    monthly["Response_Coverage"] = (
        monthly["response_mean"] /
        (monthly["response_mean"] + monthly["external_mean"] + 1e-9)
    ).clip(0, 1)

    return monthly, float(yellow), float(red)


# =============================================================================
# 8. LAG ANALYSIS
# =============================================================================
def build_lag_analysis(monthly):
    rows = []

    targets = {
        "BPKH_Exposure": "bpkh_signal",
        "Public_Trust": "trust_signal",
        "GCI": "GCI",
    }

    for target_name, target_col in targets.items():
        for lag in range(0, 4):
            if lag == 0:
                x = monthly["external_signal"]
                y = monthly[target_col]
                n = len(monthly)
            else:
                x = monthly["external_signal"].iloc[:-lag]
                y = monthly[target_col].iloc[lag:]
                n = len(x)

            rows.append({
                "relationship": f"External_Crisis_t-{lag} -> {target_name}_t",
                "lag_months": lag,
                "target": target_name,
                "n": n,
                "pearson_r": safe_corr(x, y),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 9. PROFILES + TOP EVENTS
# =============================================================================
def build_profile(df):
    rows = []

    for dim in DIMENSIONS:
        mask = df["Risk_Dimensions"].str.contains(
            rf"(^|;){re.escape(dim)}(;|$)",
            regex=True,
            na=False,
        )

        rows.append({
            "dimension": dim,
            "documents": int(mask.sum()),
            "prevalence_pct": round(
                100 * mask.mean(), 2
            ),
            "mean_score": round(
                df.loc[mask, f"Risk_1_Score"].mean()
                if mask.any() else 0.0,
                4,
            ),
        })

    return pd.DataFrame(rows)


def build_top_events(df, n=20):
    candidates = df[
        df["Document_Contagion_Score"] > 0
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Pick strongest document from each event first.
    candidates = candidates.sort_values(
        ["Document_Contagion_Score", "_parsed_date"],
        ascending=[False, True],
    )

    unique_events = (
        candidates[candidates["event_id"] != ""]
        .drop_duplicates("event_id")
        .head(n)
    )

    cols = [
        "event_id",
        "event_size",
        "_parsed_date",
        "unified_id",
        "platform",
        "Document_Contagion_Score",
        "Contagion_State",
        "Risk_Dimensions",
        "Crisis_Validity_Score",
        "BPKH_Exposure_Score",
        "Public_Trust_Score",
        "Response_Score",
        "Mitigation_Score",
        "unified_text",
    ]

    return unique_events[cols].copy()


# =============================================================================
# 10. VISUALIZATION
# =============================================================================
def make_charts(profile, monthly):
    plt.figure(figsize=(10, 5))
    p = profile.sort_values("documents")
    plt.barh(p["dimension"], p["documents"])
    plt.xlabel("Jumlah dokumen")
    plt.ylabel("Dimensi")
    plt.title("Governance Risk Dimensions — Event-Aware Model")
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "01_risk_dimensions.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.plot(
        monthly["year_month"],
        monthly["external_signal"],
        marker="o",
        label="External Crisis",
    )
    plt.plot(
        monthly["year_month"],
        monthly["bpkh_signal"],
        marker="o",
        label="BPKH Exposure",
    )
    plt.plot(
        monthly["year_month"],
        monthly["trust_signal"],
        marker="o",
        label="Public Trust",
    )
    plt.plot(
        monthly["year_month"],
        monthly["GCI"],
        marker="o",
        linewidth=3,
        label="GCI",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Percentile signal")
    plt.xlabel("Periode")
    plt.title("Governance Contagion Index — Event-Aware")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "02_gci_timeline.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(
        monthly["year_month"],
        monthly["SEI"],
        marker="o",
        label="SEI",
    )
    yellow_threshold = monthly["SEI"].quantile(YELLOW_Q)
    red_threshold = monthly["SEI"].quantile(RED_Q)

    plt.axhline(
        yellow_threshold,
        linestyle="--",
        color="yellow",
        linewidth=2,
        label=f"Yellow threshold ({yellow_threshold:.3f})",
    )
    plt.axhline(
        red_threshold,
        linestyle="--",
        color="red",
        linewidth=2,
        label=f"Red threshold ({red_threshold:.3f})",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("SEI")
    plt.xlabel("Periode")
    plt.title("Governance Contagion Early-Warning Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "03_ews_signal.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# 11. MAIN
# =============================================================================
def main():
    print("=" * 88)
    print("GOVERNANCE CONTAGION INTELLIGENCE v4.0 — FINAL")
    print("Event-Aware GCI | BPKH CFP 2026")
    print("=" * 88)

    # 1. Load
    df = load_corpus()

    # 2. Semantic mapping
    print("\n[1/7] Semantic mapping...")
    df, raw_similarity, adjusted_similarity = semantic_mapping(df)

    # 3. Event-aware adjustment
    print("[2/7] Building crisis episodes...")
    df = assign_event_ids(df)

    # 4. Monthly
    print("[3/7] Building monthly GCI...")
    monthly = build_monthly(df)

    # 5. EWS
    print("[4/7] Anomaly detection + EWS...")
    monthly = anomaly_detection(monthly)
    monthly, yellow, red = classify_ews(monthly)

    # 6. Lag
    print("[5/7] Lead/lag analysis...")
    lag = build_lag_analysis(monthly)

    # 7. Profiles
    print("[6/7] Building profiles...")
    profile = build_profile(df)
    top_events = build_top_events(df, n=20)

    # Main document output
    document_cols = [
        "unified_id",
        "platform_source",
        "platform",
        "_parsed_date",
        "unified_text",
        "Risk_Dimensions",
        "Top_Risk_Dimension",
        "Top_Risk_Score",
        "Risk_1",
        "Risk_1_Score",
        "Risk_2",
        "Risk_2_Score",
        "Risk_3",
        "Risk_3_Score",
        "External_Crisis_Score",
        "BPKH_Exposure_Score",
        "Public_Trust_Score",
        "Response_Score",
        "Mitigation_Score",
        "Crisis_Validity_Score",
        "Crisis_Event_Flag",
        "Strong_Crisis_Event_Flag",
        "Mitigation_Dominant_Flag",
        "Document_Contagion_Score",
        "Contagion_State",
        "event_id",
        "event_size",
    ]

    df[document_cols].to_csv(
        OUT_DIR / "01_document_risk_signals.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    profile.to_csv(
        OUT_DIR / "02_risk_dimension_profile.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    monthly.to_csv(
        OUT_DIR / "03_monthly_gci_ews.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    lag.to_csv(
        OUT_DIR / "04_lag_analysis.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    if not top_events.empty:
        top_events.to_csv(
            OUT_DIR / "05_top_contagion_events.csv",
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )

    manifest = {
        "pipeline": "Governance Contagion Intelligence v4.0 FINAL",
        "embedding_model": EMBEDDING_MODEL,
        "documents_analyzed": int(len(df)),
        "date_start": str(START_DATE),
        "date_end": str(END_DATE),
        "taxonomy_dimensions": DIMENSIONS,
        "external_dimensions": EXTERNAL_DIMS,
        "bpkh_dimensions": BPKH_DIMS,
        "trust_dimensions": TRUST_DIMS,
        "response_dimension": RESPONSE_DIM,
        "semantic_threshold": SIM_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "crisis_gate": CRISIS_GATE,
        "strong_crisis_gate": STRONG_CRISIS_GATE,
        "bpkh_gate": BPKH_GATE,
        "trust_gate": TRUST_GATE,
        "event_window_days": EVENT_WINDOW_DAYS,
        "event_similarity_threshold": EVENT_SIM_THRESHOLD,
        "ews_yellow_quantile": YELLOW_Q,
        "ews_red_quantile": RED_Q,
        "yellow_threshold": yellow,
        "red_threshold": red,
        "methodological_note": (
            "GCI adalah analytical proxy untuk mendeteksi potential governance "
            "contagion. Bukan ukuran risiko resmi BPKH dan bukan bukti kausal. "
            "Clarification/mitigation dipisahkan dari crisis evidence. "
            "Response_Coverage dihitung sebagai response_mean / "
            "(response_mean + external_mean + epsilon), dibatasi 0–1, "
            "dan tidak digunakan dalam formula GCI."
        ),
    }

    with open(
        OUT_DIR / "06_run_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    print("[7/7] Creating charts...")
    make_charts(profile, monthly)

    # Console summary
    print("\n" + "=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print(f"Documents analyzed : {len(df):,}")
    print(f"News               : {(df['platform_source'] == 'News').sum():,}")
    print(f"Social             : {(df['platform_source'] == 'Social').sum():,}")
    print(f"Crisis documents   : {df['Crisis_Event_Flag'].sum():,}")
    print(
        f"Contagion candidates: "
        f"{(df['Contagion_State'].isin(['S3_Potential_Contagion', 'S2_Exposure_With_Response'])).sum():,}"
    )
    print(f"Unique crisis events: {df.loc[df['event_id'] != '', 'event_id'].nunique():,}")

    print("\n[STATE DISTRIBUTION]")
    print(df["Contagion_State"].value_counts().to_string())

    print("\n[TOP GCI EVENTS]")
    if not top_events.empty:
        print(
            top_events[
                [
                    "event_id",
                    "_parsed_date",
                    "Document_Contagion_Score",
                    "Contagion_State",
                    "event_size",
                ]
            ].head(10).to_string(index=False)
        )

    print("\n[MONTHLY EWS]")
    print(
        monthly[
            [
                "year_month",
                "external_signal",
                "bpkh_signal",
                "trust_signal",
                "GCI",
                "SEI",
                "EWS_Level",
            ]
        ].tail(12).to_string(index=False)
    )

    print("\n[LAG ANALYSIS]")
    print(lag.to_string(index=False))

    print("\n" + "=" * 88)
    print(f"SELESAI. Output: {OUT_DIR.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
