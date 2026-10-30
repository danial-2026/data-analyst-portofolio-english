# =============================================================================
# 7. NETWORK & CO-OCCURRENCE ANALYSIS — NCI + TEMPORAL + ANOMALY + EWS
# BPKH CFP 2026
#
# Purpose
# -------
# Complement Script 6 with a network-based governance contagion analysis.
#
# Pipeline:
#   Risk Signals
#        ↓
#   Network / Co-occurrence Analysis
#        ↓
#   NCI (Network Contagion Index)
#        ↓
#   Temporal Analysis
#        ↓
#   Anomaly Detection
#        ↓
#   AI-Based EWS
#        ↓
#   Governance Contagion Risk
#
# IMPORTANT:
# - This script does NOT run LDA/BERTopic.
# - It does NOT repeat semantic mapping from Script 6.
# - It consumes Script 6 outputs as the analytical input.
# - NCI/EWS/GCR are model-based analytical proxies, not official BPKH risk
#   ratings and not causal evidence.
# =============================================================================

import json
import re
import warnings
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError(
        "Package 'networkx' is required. Install with: pip install networkx"
    ) from exc

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Script 6 output:
INPUT_DOCUMENT_SIGNALS = Path(
    r"C:\Users\Danial MW\Documents\KAKANG MKS\LOMBA\CFP BPKH 2026\Data\7. governance_contagion_v4_results\01_document_risk_signals.csv"
)

# Script 6 monthly output, used only as a cross-check/reference.
INPUT_MONTHLY_GCI = Path(
    r"C:\Users\Danial MW\Documents\KAKANG MKS\LOMBA\CFP BPKH 2026\Data\7. governance_contagion_v4_results\03_monthly_gci_ews.csv"
)

OUT_DIR = Path("network_contagion_FINAL_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2026-09-03 23:59:59")

# Risk-dimension activation threshold.
# This is intentionally aligned with the crisis/exposure gates used by Script 6.
DIMENSION_THRESHOLD = 0.12

# Network filters.
MIN_EDGE_WEIGHT = 0.02
TOP_EDGES = 40
TOP_NODES = 30

# NCI weights.
# NCI captures network position + crisis linkage + trust linkage.
NCI_W_CENTRALITY = 0.30
NCI_W_CRISIS_EDGE = 0.30
NCI_W_TRUST_EDGE = 0.20
NCI_W_BETWEENNESS = 0.20

# Temporal features.
ROLLING_WINDOW = 3

# Isolation Forest.
MIN_MONTHS_FOR_ANOMALY = 8
ANOMALY_QUANTILE = 0.90

# EWS thresholds.
YELLOW_Q = 0.75
RED_Q = 0.90


# =============================================================================
# 2. RESEARCH NETWORK TAXONOMY
# =============================================================================
#
# These dimensions are inherited from Script 6.
# Clarification_Response is deliberately excluded from the risk network:
# response/mitigation is a management response, not crisis evidence.
# =============================================================================

NETWORK_DIMS = [
    "Quota_Governance",
    "Corruption_Investigation",
    "Governance_Oversight",
    "BPKH_Governance",
    "Hajj_Fund_Management",
    "Public_Trust_Reputation",
]

EXTERNAL_DIMS = {
    "Quota_Governance",
    "Corruption_Investigation",
    "Governance_Oversight",
}

BPKH_DIMS = {
    "BPKH_Governance",
    "Hajj_Fund_Management",
}

TRUST_DIMS = {
    "Public_Trust_Reputation",
}

# Human-readable labels for tables/figures.
LABELS = {
    "Quota_Governance": "Quota Governance",
    "Corruption_Investigation": "Corruption / Investigation",
    "Governance_Oversight": "Governance Oversight",
    "BPKH_Governance": "BPKH Governance",
    "Hajj_Fund_Management": "Hajj Fund Management",
    "Public_Trust_Reputation": "Public Trust / Reputation",
}


# =============================================================================
# 3. HELPERS
# =============================================================================

def safe_float(x, default=0.0):
    try:
        value = float(x)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def minmax_series(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def percentile_rank(s):
    """
    Convert input to a pandas Series and calculate percentile rank.

    Works safely with:
    - pandas Series
    - numpy arrays
    - lists
    """

    s = pd.Series(s).copy()

    s = pd.to_numeric(
        s,
        errors="coerce"
    ).fillna(0.0)

    if len(s) == 0:
        return pd.Series(
            dtype=float,
            index=s.index
        )

    # Percentile rank using empirical CDF.
    # If all values are identical, return 0.0
    # rather than producing NaN.
    if s.nunique() <= 1:
        return pd.Series(
            np.zeros(len(s)),
            index=s.index,
            dtype=float
        )

    return s.rank(
        method="average",
        pct=True
    )


def parse_date_column(df):
    candidates = [
        "_parsed_date",
        "published_date",
        "date",
        "datetime",
        "created_at",
        "retrieved_at",
    ]
    for col in candidates:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce", utc=True)
            if dt.notna().sum() > 0:
                return dt.dt.tz_convert(None)

    raise ValueError(
        "No usable date column found. Expected one of: "
        + ", ".join(candidates)
    )


def activate_dimensions(row):
    """
    Prefer explicit dimension score columns from Script 6.
    If unavailable, parse Risk_Dimensions as a fallback.
    """
    active = []

    for dim in NETWORK_DIMS:
        score_col_candidates = [
            f"{dim}_Score",
            dim,
        ]
        score = None
        for col in score_col_candidates:
            if col in row.index:
                score = safe_float(row[col], 0.0)
                break

        if score is not None and score >= DIMENSION_THRESHOLD:
            active.append(dim)

    if active:
        return active

    raw = str(row.get("Risk_Dimensions", ""))
    if raw and raw.lower() not in {"nan", "none", ""}:
        for dim in NETWORK_DIMS:
            if re.search(rf"\b{re.escape(dim)}\b", raw):
                active.append(dim)

    return list(dict.fromkeys(active))


def safe_corr(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return np.nan
    if a[mask].nunique() <= 1 or b[mask].nunique() <= 1:
        return np.nan
    return float(a[mask].corr(b[mask]))


# =============================================================================
# 4. LOAD SCRIPT 6 OUTPUT
# =============================================================================

def load_document_signals():
    print("[1/9] Loading Script 6 document-risk output...")

    if not os.path.exists(INPUT_DOCUMENT_SIGNALS):
        raise FileNotFoundError(
            f"Script 6 document-risk output not found: {INPUT_DOCUMENT_SIGNALS}"
        )

    df = pd.read_csv(
        INPUT_DOCUMENT_SIGNALS,
        sep=";",
        encoding="utf-8-sig"
    )

    print(f"  Loaded {len(df):,} document records.")
    return df


# =============================================================================
# 5. DOCUMENT-LEVEL CO-OCCURRENCE NETWORK
# =============================================================================

def build_cooccurrence_network(df):
    """
    Build document-level co-occurrence network from Script 6 Risk_Dimensions.

    Script 6 provides:
        Risk_Dimensions

    Script 7 converts this into:
        active_dimensions

    Only NETWORK_DIMS are retained for the network analysis.
    """

    # ------------------------------------------------------------------
    # 1. Construct active_dimensions from Script 6 Risk_Dimensions
    # ------------------------------------------------------------------
    if "active_dimensions" not in df.columns:

        if "Risk_Dimensions" not in df.columns:
            raise KeyError(
                "Column 'Risk_Dimensions' is not available in Script 6 output."
            )

        def parse_active_dimensions(value):
            if pd.isna(value):
                return []

            # Script 6 stores multiple dimensions in one field.
            dims = [
                x.strip()
                for x in str(value).split(";")
                if x.strip()
            ]

            # Keep only dimensions explicitly defined for
            # the Network / Co-occurrence Analysis.
            dims = [
                x for x in dims
                if x in NETWORK_DIMS
            ]

            # Remove duplicates while preserving order.
            return list(dict.fromkeys(dims))

        df["active_dimensions"] = df["Risk_Dimensions"].apply(
            parse_active_dimensions
        )

    # ------------------------------------------------------------------
    # 2. Build pairwise co-occurrence counts
    # ------------------------------------------------------------------
    edge_counter = {}

    for dims in df["active_dimensions"]:
        dims = sorted(set(dims))

        if len(dims) < 2:
            continue

        for a, b in combinations(dims, 2):
            key = (a, b)
            edge_counter[key] = edge_counter.get(key, 0) + 1

    # ------------------------------------------------------------------
    # 3. Convert edges to DataFrame
    # ------------------------------------------------------------------
    edge_rows = []

    total_docs = max(len(df), 1)

    for (a, b), count in edge_counter.items():
        edge_rows.append(
            {
                "source": a,
                "target": b,
                "cooccurrence_count": int(count),
                "cooccurrence_rate": count / total_docs,
            }
        )

    edges = pd.DataFrame(edge_rows)

    # ------------------------------------------------------------------
    # 4. Handle empty network
    # ------------------------------------------------------------------
    if edges.empty:
        edges = pd.DataFrame(
            columns=[
                "source",
                "target",
                "cooccurrence_count",
                "cooccurrence_rate",
                "association_strength",
            ]
        )

        G = nx.Graph()

        for dim in NETWORK_DIMS:
            G.add_node(dim)

        return edges, G

    # ------------------------------------------------------------------
    # 5. Calculate marginal document frequencies
    # ------------------------------------------------------------------
    dim_doc_counts = {}

    for dim in NETWORK_DIMS:
        dim_doc_counts[dim] = int(
            df["active_dimensions"]
            .apply(lambda x: dim in x)
            .sum()
        )

    # ------------------------------------------------------------------
    # 6. Association strength
    #
    # observed co-occurrence /
    # expected co-occurrence under marginal independence
    # ------------------------------------------------------------------
    associations = []

    for _, r in edges.iterrows():

        a = r["source"]
        b = r["target"]

        p_a = dim_doc_counts.get(a, 0) / total_docs
        p_b = dim_doc_counts.get(b, 0) / total_docs
        p_ab = r["cooccurrence_count"] / total_docs

        expected = p_a * p_b

        association = (
            p_ab / expected
            if expected > 0
            else 0.0
        )

        associations.append(float(association))

    edges["association_strength"] = associations

    # ------------------------------------------------------------------
    # 7. Keep sufficiently supported edges
    # ------------------------------------------------------------------
    edges = edges[
        edges["cooccurrence_rate"] >= MIN_EDGE_WEIGHT
    ].copy()

    # ------------------------------------------------------------------
    # 8. Build NetworkX graph
    # ------------------------------------------------------------------
    G = nx.Graph()

    # Add all predefined risk dimensions as nodes,
    # including isolated nodes.
    for dim in NETWORK_DIMS:
        G.add_node(dim)

    # Add supported edges.
    for _, r in edges.iterrows():

        G.add_edge(
            r["source"],
            r["target"],
            weight=float(r["association_strength"]),
            count=int(r["cooccurrence_count"]),
            rate=float(r["cooccurrence_rate"]),
        )

    # ------------------------------------------------------------------
    # 9. Sort edges by analytical importance
    # ------------------------------------------------------------------
    edges = edges.sort_values(
        ["association_strength", "cooccurrence_count"],
        ascending=False,
    ).reset_index(drop=True)

    return edges, G

# =============================================================================
# 6. NETWORK CENTRALITY
# =============================================================================

def calculate_network_metrics(G):
    if len(G.nodes) == 0:
        return pd.DataFrame()

    degree = nx.degree_centrality(G)

    if G.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(
            G,
            weight=None,
            normalized=True,
        )
    else:
        betweenness = {n: 0.0 for n in G.nodes}

    # Eigenvector centrality may fail on a disconnected graph.
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes}

    rows = []

    for node in G.nodes:
        rows.append(
            {
                "node": node,
                "label": LABELS.get(node, node),
                "degree_centrality": degree.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "eigenvector_centrality": eigenvector.get(node, 0.0),
            }
        )

    metrics = pd.DataFrame(rows)

    # Centrality composite.
    metrics["centrality_score"] = (
        0.40 * metrics["degree_centrality"]
        + 0.40 * metrics["betweenness_centrality"]
        + 0.20 * metrics["eigenvector_centrality"]
    )

    return metrics.sort_values(
        "centrality_score",
        ascending=False,
    )


# =============================================================================
# 7. NCI — NETWORK CONTAGION INDEX
# =============================================================================

def calculate_nci(G, edges, metrics):
    """
    NCI is calculated at the network level and then assigned to BPKH-related
    dimensions.

    Components:
      1. Centrality
      2. Linkage to external crisis dimensions
      3. Linkage to public trust/reputation
      4. Betweenness / bridge position

    This is an analytical proxy for network-based contagion exposure.
    """

    if metrics.empty:
        return pd.DataFrame()

    edge_lookup = {}

    for _, r in edges.iterrows():
        key = tuple(sorted([r["source"], r["target"]]))
        edge_lookup[key] = float(r["association_strength"])

    def edge_strength(a, b):
        return edge_lookup.get(tuple(sorted([a, b])), 0.0)

    rows = []

    for _, r in metrics.iterrows():
        node = r["node"]

        crisis_edges = [
            edge_strength(node, crisis)
            for crisis in EXTERNAL_DIMS
            if crisis != node
        ]

        trust_edges = [
            edge_strength(node, trust)
            for trust in TRUST_DIMS
            if trust != node
        ]

        crisis_edge = max(crisis_edges) if crisis_edges else 0.0
        trust_edge = max(trust_edges) if trust_edges else 0.0

        rows.append(
            {
                "node": node,
                "label": r["label"],
                "centrality_score": r["centrality_score"],
                "crisis_edge_strength": crisis_edge,
                "trust_edge_strength": trust_edge,
                "betweenness_centrality": r["betweenness_centrality"],
            }
        )

    nci = pd.DataFrame(rows)

    # Normalize edge strengths so very large association ratios do not dominate.
    nci["crisis_edge_norm"] = minmax_series(nci["crisis_edge_strength"])
    nci["trust_edge_norm"] = minmax_series(nci["trust_edge_strength"])
    nci["betweenness_norm"] = minmax_series(
        nci["betweenness_centrality"]
    )

    nci["NCI"] = (
        NCI_W_CENTRALITY * minmax_series(nci["centrality_score"])
        + NCI_W_CRISIS_EDGE * nci["crisis_edge_norm"]
        + NCI_W_TRUST_EDGE * nci["trust_edge_norm"]
        + NCI_W_BETWEENNESS * nci["betweenness_norm"]
    )

    # Focus on the BPKH side of the contagion pathway.
    nci["BPKH_Network_Node"] = nci["node"].isin(BPKH_DIMS).astype(int)

    # A node is a direct bridge candidate when it is BPKH-related.
    nci["Bridge_Candidate"] = (
        (nci["BPKH_Network_Node"] == 1)
        & (nci["betweenness_centrality"] > 0)
    ).astype(int)

    return nci.sort_values("NCI", ascending=False)


# =============================================================================
# 8. MONTHLY NCI / TEMPORAL ANALYSIS
# =============================================================================

def build_monthly_nci(df, G, edges, node_nci):
    """
    Build monthly Network Contagion / temporal signals.

    Date source:
        Script 6 -> _parsed_date

    Monthly signals:
        - external-BPKH co-occurrence
        - BPKH-trust co-occurrence
        - network document intensity
        - unique edge intensity
        - NCI
        - NCI momentum
    """

    # ---------------------------------------------------------------
    # 1. Make sure date information exists
    # ---------------------------------------------------------------
    if "_parsed_date" not in df.columns:
        raise KeyError(
            "Column '_parsed_date' is not available in Script 6 output."
        )

    df = df.copy()

    df["_parsed_date"] = pd.to_datetime(
        df["_parsed_date"],
        errors="coerce"
    )

    # Remove records without valid dates
    df = df.dropna(subset=["_parsed_date"]).copy()

    if df.empty:
        raise ValueError(
            "No valid dates were found in '_parsed_date'."
        )

    # ---------------------------------------------------------------
    # 2. Create year_month locally
    # ---------------------------------------------------------------
    df["year_month"] = (
        df["_parsed_date"]
        .dt.to_period("M")
        .astype(str)
    )

    # ---------------------------------------------------------------
    # 3. Make sure active_dimensions exists
    # ---------------------------------------------------------------
    if "active_dimensions" not in df.columns:

        if "Risk_Dimensions" not in df.columns:
            raise KeyError(
                "Column 'Risk_Dimensions' is not available."
            )

        def parse_active_dimensions(value):

            if pd.isna(value):
                return []

            dims = [
                x.strip()
                for x in str(value).split(";")
                if x.strip()
            ]

            dims = [
                x for x in dims
                if x in NETWORK_DIMS
            ]

            return list(dict.fromkeys(dims))

        df["active_dimensions"] = (
            df["Risk_Dimensions"]
            .apply(parse_active_dimensions)
        )

    # ---------------------------------------------------------------
    # 4. Network pair definitions
    # ---------------------------------------------------------------
    external_dims = {
        "Quota_Governance",
        "Corruption_Investigation",
        "Governance_Oversight",
    }

    bpkh_dims = {
        "BPKH_Governance",
        "Hajj_Fund_Management",
    }

    trust_dims = {
        "Public_Trust_Reputation",
    }

    # ---------------------------------------------------------------
    # 5. Calculate monthly network signals
    # ---------------------------------------------------------------
    monthly_rows = []

    for month, g in df.groupby("year_month", sort=True):

        n_docs = len(g)

        external_bpkh_edges = 0
        bpkh_trust_edges = 0
        network_docs = 0
        unique_pairs = set()

        for dims in g["active_dimensions"]:

            dims = set(dims)

            if len(dims) >= 2:
                network_docs += 1

            # External -> BPKH relationship
            if (
                len(dims & external_dims) > 0
                and len(dims & bpkh_dims) > 0
            ):
                external_bpkh_edges += 1

            # BPKH -> Public Trust relationship
            if (
                len(dims & bpkh_dims) > 0
                and len(dims & trust_dims) > 0
            ):
                bpkh_trust_edges += 1

            # Unique co-occurrence pairs
            if len(dims) >= 2:

                for a, b in combinations(
                    sorted(dims),
                    2
                ):
                    unique_pairs.add((a, b))

        # -----------------------------------------------------------
        # Raw monthly rates
        # -----------------------------------------------------------
        external_bpkh_rate = (
            external_bpkh_edges / n_docs
            if n_docs > 0
            else 0.0
        )

        bpkh_trust_rate = (
            bpkh_trust_edges / n_docs
            if n_docs > 0
            else 0.0
        )

        network_doc_rate = (
            network_docs / n_docs
            if n_docs > 0
            else 0.0
        )

        unique_edge_count = len(unique_pairs)

        monthly_rows.append(
            {
                "year_month": month,
                "Documents": int(n_docs),
                "External_BPKH_Edge_Rate": external_bpkh_rate,
                "BPKH_Trust_Edge_Rate": bpkh_trust_rate,
                "Network_Document_Rate": network_doc_rate,
                "Unique_Edge_Count": int(unique_edge_count),
            }
        )

    monthly = pd.DataFrame(monthly_rows)

    # ---------------------------------------------------------------
    # 6. Normalize network indicators
    # ---------------------------------------------------------------
    if monthly.empty:
        return monthly

    def safe_minmax(series):

        s = pd.to_numeric(
            series,
            errors="coerce"
        ).fillna(0.0)

        s_min = s.min()
        s_max = s.max()

        if s_max == s_min:
            return pd.Series(
                np.zeros(len(s)),
                index=s.index
            )

        return (s - s_min) / (s_max - s_min)

    monthly["External_BPKH_Signal"] = safe_minmax(
        monthly["External_BPKH_Edge_Rate"]
    )

    monthly["BPKH_Trust_Signal"] = safe_minmax(
        monthly["BPKH_Trust_Edge_Rate"]
    )

    monthly["Network_Document_Signal"] = safe_minmax(
        monthly["Network_Document_Rate"]
    )

    monthly["Unique_Edge_Signal"] = safe_minmax(
        monthly["Unique_Edge_Count"]
    )

        # ---------------------------------------------------------------
    # 7. Network Contagion Index (NCI)
    # ---------------------------------------------------------------
    monthly["NCI"] = (
        0.40 * monthly["External_BPKH_Signal"]
        + 0.30 * monthly["BPKH_Trust_Signal"]
        + 0.20 * monthly["Network_Document_Signal"]
        + 0.10 * monthly["Unique_Edge_Signal"]
    )

    # ---------------------------------------------------------------
    # 8. Standardized signal names for downstream analysis
    # ---------------------------------------------------------------
    monthly["external_bpkh_edge_rate_signal"] = (
        monthly["External_BPKH_Signal"]
    )

    monthly["bpkh_trust_edge_rate_signal"] = (
        monthly["BPKH_Trust_Signal"]
    )

    monthly["network_documents_signal"] = (
        monthly["Network_Document_Signal"]
    )

    monthly["unique_edges_signal"] = (
        monthly["Unique_Edge_Signal"]
    )

    # ---------------------------------------------------------------
    # 9. Temporal momentum
    # ---------------------------------------------------------------
    monthly["NCI_momentum"] = (
        monthly["NCI"]
        .diff()
        .fillna(0.0)
    )

    # ---------------------------------------------------------------
    # 10. Rolling temporal baseline
    # ---------------------------------------------------------------
    monthly["NCI_3M_Mean"] = (
        monthly["NCI"]
        .rolling(
            window=3,
            min_periods=1
        )
        .mean()
    )

    monthly["NCI_3M_Change"] = (
        monthly["NCI"]
        - monthly["NCI_3M_Mean"]
    )

    # ---------------------------------------------------------------
    # 11. Network structure metadata
    # ---------------------------------------------------------------
    monthly["Network_Nodes"] = int(len(G.nodes))
    monthly["Network_Edges"] = int(len(G.edges))

    return monthly

# =============================================================================
# 9. MERGE WITH SCRIPT 6 TEMPORAL RISK SIGNALS
# =============================================================================

def merge_script6_monthly(monthly):
    if not INPUT_MONTHLY_GCI.exists():
        print(
            "[WARN] Script 6 monthly output not found. "
            "Proceeding with network-only temporal signals."
        )
        return monthly

    base = pd.read_csv(INPUT_MONTHLY_GCI)

    if "year_month" not in base.columns:
        print(
            "[WARN] Script 6 monthly file has no year_month column. "
            "Proceeding without merge."
        )
        return monthly

    keep = [
        c for c in [
            "year_month",
            "GCI",
            "SEI",
            "external_signal",
            "bpkh_signal",
            "trust_signal",
            "volume_signal",
            "crisis_rate_signal",
            "event_density",
            "GCI_momentum",
        ]
        if c in base.columns
    ]

    base = base[keep].drop_duplicates("year_month")

    # Avoid collisions with NCI's own momentum.
    if "GCI_momentum" in base.columns:
        base = base.rename(columns={"GCI_momentum": "Script6_GCI_momentum"})

    merged = monthly.merge(base, on="year_month", how="left")

    return merged


# =============================================================================
# 10. ANOMALY DETECTION
# =============================================================================

def anomaly_detection(monthly):
    monthly = monthly.copy()

    feature_candidates = [
        "NCI",
        "NCI_momentum",
        "external_bpkh_edge_rate_signal",
        "bpkh_trust_edge_rate_signal",
        "network_documents_signal",
        "unique_edges_signal",
    ]

    # GCI/SEI are included only if Script 6 successfully supplied them.
    for c in ["GCI", "SEI"]:
        if c in monthly.columns:
            feature_candidates.append(c)

    features = list(dict.fromkeys(feature_candidates))

    monthly["Anomaly_Score"] = 0.0
    monthly["Anomaly_Flag"] = 0

    if len(monthly) < MIN_MONTHS_FOR_ANOMALY:
        print(
            "[WARN] Fewer than 8 months. Isolation Forest is not "
            "considered sufficiently informative."
        )
        return monthly

    X = monthly[features].fillna(0.0).to_numpy()

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

    cutoff = monthly["Anomaly_Score"].quantile(ANOMALY_QUANTILE)

    monthly["Anomaly_Flag"] = (
        monthly["Anomaly_Score"] >= cutoff
    ).astype(int)

    return monthly


# =============================================================================
# 11. AI-BASED EWS
# =============================================================================

def build_ews(monthly):
    monthly = monthly.copy()

    # Temporal / risk inputs.
    # If Script 6 is available, SEI carries the existing GCI-based temporal
    # escalation signal. NCI is then added as an independent structural signal.
    sei = (
        monthly["SEI"]
        if "SEI" in monthly.columns
        else monthly["NCI"]
    )

    monthly["SEI_reference"] = sei.fillna(0.0)

    # Integrated Network-Temporal Early Warning Score.
    #
    # NCI = network structure
    # SEI = temporal escalation from Script 6
    # Anomaly = unusual multivariate monthly configuration
    #
    # We do NOT treat anomaly alone as governance risk.
    monthly["EWS_Score"] = (
        0.45 * monthly["SEI_reference"]
        + 0.35 * monthly["NCI"]
        + 0.20 * monthly["Anomaly_Score"]
    ).clip(0, 1)

    yellow = monthly["EWS_Score"].quantile(YELLOW_Q)
    red = monthly["EWS_Score"].quantile(RED_Q)

    def level(x):
        if x >= red:
            return "RED"
        if x >= yellow:
            return "YELLOW"
        return "GREEN"

    monthly["EWS_Level"] = monthly["EWS_Score"].map(level)

    monthly["EWS_Yellow_Threshold"] = float(yellow)
    monthly["EWS_Red_Threshold"] = float(red)

    return monthly


# =============================================================================
# 12. INTEGRATED GOVERNANCE CONTAGION RISK
# =============================================================================

def build_governance_contagion_risk(monthly):
    monthly = monthly.copy()

    # GCI comes from Script 6. If unavailable, NCI is used as a transparent
    # fallback rather than silently fabricating a GCI.
    gci = (
        monthly["GCI"]
        if "GCI" in monthly.columns
        else monthly["NCI"]
    ).fillna(0.0)

    monthly["GCI_reference"] = gci

    # Integrated Governance Contagion Risk:
    #   55% existing GCI pathway
    #   30% network contagion structure
    #   15% anomaly signal
    #
    # Anomaly is intentionally given the smallest weight because it is a
    # detector of unusual configurations, not substantive risk by itself.
    monthly["Governance_Contagion_Risk"] = (
        0.55 * monthly["GCI_reference"]
        + 0.30 * monthly["NCI"]
        + 0.15 * monthly["Anomaly_Score"]
    ).clip(0, 1)

    yellow = monthly["Governance_Contagion_Risk"].quantile(YELLOW_Q)
    red = monthly["Governance_Contagion_Risk"].quantile(RED_Q)

    def risk_level(x):
        if x >= red:
            return "RED"
        if x >= yellow:
            return "YELLOW"
        return "GREEN"

    monthly["Governance_Contagion_Level"] = (
        monthly["Governance_Contagion_Risk"].map(risk_level)
    )

    monthly["GCR_Yellow_Threshold"] = float(yellow)
    monthly["GCR_Red_Threshold"] = float(red)

    return monthly


# =============================================================================
# 13. TOP NETWORK BRIDGES / EDGES
# =============================================================================

def build_top_edges(edges):
    if edges.empty:
        return edges.copy()

    out = edges.copy()

    out["source_label"] = out["source"].map(LABELS).fillna(out["source"])
    out["target_label"] = out["target"].map(LABELS).fillna(out["target"])

    out["edge_type"] = np.select(
        [
            out.apply(
                lambda r: (
                    (r["source"] in EXTERNAL_DIMS and r["target"] in BPKH_DIMS)
                    or
                    (r["target"] in EXTERNAL_DIMS and r["source"] in BPKH_DIMS)
                ),
                axis=1,
            ),
            out.apply(
                lambda r: (
                    (r["source"] in BPKH_DIMS and r["target"] in TRUST_DIMS)
                    or
                    (r["target"] in BPKH_DIMS and r["source"] in TRUST_DIMS)
                ),
                axis=1,
            ),
        ],
        [
            "External_Crisis_to_BPKH",
            "BPKH_to_Public_Trust",
        ],
        default="Within_Other_Risk_Dimensions",
    )

    return out.sort_values(
        ["association_strength", "cooccurrence_count"],
        ascending=False,
    ).head(TOP_EDGES)


# =============================================================================
# 14. VISUALIZATION
# =============================================================================

def make_network_chart(G, metrics):
    """
    Publication-ready governance risk co-occurrence network.

    Interpretation:
        - Node size  -> Network Contagion Index (NCI)
        - Edge width -> Association strength
        - Edge label -> Co-occurrence count
        - Node position is manually controlled to improve readability
          because the network contains only six dimensions and one
          isolated node.
    """

    fig, ax = plt.subplots(
        figsize=(15, 10)
    )

    if G.number_of_nodes() == 0:

        ax.text(
            0.5,
            0.5,
            "No network nodes available",
            ha="center",
            va="center",
            fontsize=14,
        )

        ax.axis("off")

    else:

        # =============================================================
        # 1. MANUAL NETWORK POSITION
        # =============================================================
        #
        # Layout is intentionally structured:
        #
        # External governance risks
        #        ↓
        # BPKH governance / fund management
        #        ↓
        # Public trust / corruption exposure
        #
        # Governance Oversight is isolated in the observed network,
        # therefore it is placed separately but still inside the figure.
        #
        pos = {

            "Quota_Governance": (-0.65, 0.55),

            "BPKH_Governance": (-0.25, 0.05),

            "Public_Trust_Reputation": (0.25, 0.40),

            "Hajj_Fund_Management": (0.70, 0.60),

            "Corruption_Investigation": (0.55, -0.35),

            "Governance_Oversight": (-0.75, -0.55),
        }

        # Fallback position for unexpected nodes
        missing_nodes = [
            n for n in G.nodes
            if n not in pos
        ]

        if missing_nodes:

            fallback = nx.spring_layout(
                G.subgraph(missing_nodes),
                seed=42,
            )

            for node, xy in fallback.items():
                pos[node] = xy

        # =============================================================
        # 2. NODE SIZE = NCI
        # =============================================================

        node_sizes = []

        for node in G.nodes:

            row = metrics[
                metrics["node"] == node
            ]

            score = (
                float(row["NCI"].iloc[0])
                if not row.empty
                else 0.0
            )

            node_sizes.append(
                900 + 2800 * score
            )

        # =============================================================
        # 3. DRAW EDGES
        # =============================================================

        edge_widths = []

        for u, v in G.edges:

            association = float(
                G[u][v].get(
                    "weight",
                    0.0
                )
            )

            edge_widths.append(
                max(
                    1.2,
                    min(
                        6.0,
                        1.2 + association
                    ),
                )
            )

        nx.draw_networkx_edges(
            G,
            pos,
            width=edge_widths,
            alpha=0.55,
            edge_color="dimgray",
            connectionstyle="arc3,rad=0.03",
            ax=ax,
        )

        # =============================================================
        # 4. DRAW NODES
        # =============================================================

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color="steelblue",
            alpha=0.90,
            linewidths=1.8,
            edgecolors="black",
            ax=ax,
        )

        # =============================================================
        # 5. NODE LABELS
        # =============================================================

        node_labels = {}

        for node in G.nodes:

            row = metrics[
                metrics["node"] == node
            ]

            nci = (
                float(row["NCI"].iloc[0])
                if not row.empty
                else 0.0
            )

            label = LABELS.get(
                node,
                node
            )

            node_labels[node] = (
                f"{label}\n"
                f"NCI = {nci:.3f}"
            )

        nx.draw_networkx_labels(
            G,
            pos,
            labels=node_labels,
            font_size=9,
            font_weight="normal",
            ax=ax,
        )

        # =============================================================
        # 6. EDGE LABELS
        # =============================================================
        #
        # To avoid excessive visual clutter, display only
        # co-occurrence count (n).
        #
        # Association strength remains encoded by edge width and
        # is available in the exported edge CSV.
        #

        edge_labels = {}

        for u, v in G.edges:

            count = int(
                G[u][v].get(
                    "count",
                    0
                )
            )

            edge_labels[(u, v)] = (
                f"n={count}"
            )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,
            rotate=False,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
                pad=0.20,
            ),
            ax=ax,
        )

        # =============================================================
        # 7. TITLE
        # =============================================================

        ax.set_title(
            "Governance Risk Co-occurrence Network",
            fontsize=19,
            pad=20,
        )

        # =============================================================
        # 8. ANALYTICAL LEGEND / INTERPRETATION
        # =============================================================

        ax.text(
            0.01,
            0.025,
            "Node size ∝ NCI | "
            "Edge width ∝ association strength | "
            "n = number of documents showing co-occurrence",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
        )

        # =============================================================
        # 9. NETWORK SUMMARY
        # =============================================================

        ax.text(
            0.99,
            0.025,
            f"Nodes: {G.number_of_nodes()} | "
            f"Edges: {G.number_of_edges()}",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
        )

        # =============================================================
        # 10. IMPORTANT INTERPRETIVE NOTE
        # =============================================================

        isolated_nodes = [
            n for n in G.nodes
            if G.degree(n) == 0
        ]

        if isolated_nodes:

            isolated_labels = ", ".join(
                LABELS.get(n, n)
                for n in isolated_nodes
            )

            ax.text(
                0.01,
                0.075,
                f"Isolated observed dimension: "
                f"{isolated_labels}",
                transform=ax.transAxes,
                fontsize=8.5,
                ha="left",
                va="bottom",
            )

        # =============================================================
        # 11. CLEAN AXIS
        # =============================================================

        ax.set_xlim(
            -1.05,
            1.05
        )

        ax.set_ylim(
            -0.85,
            0.90
        )

        ax.axis("off")

    fig.tight_layout(
        pad=3
    )

    fig.savefig(
        OUT_DIR / "01_governance_risk_network.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_nci_chart(metrics):
    """
    Rank risk dimensions by Network Contagion Index.
    """

    fig, ax = plt.subplots(figsize=(11, 7))

    p = (
        metrics
        .head(TOP_NODES)
        .sort_values("NCI")
        .copy()
    )

    bars = ax.barh(
        p["label"],
        p["NCI"],
    )

    # -------------------------------------------------------------
    # Value labels
    # -------------------------------------------------------------
    for bar, value in zip(
        bars,
        p["NCI"]
    ):
        ax.text(
            value + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlim(0, 1.08)

    ax.set_xlabel(
        "Network Contagion Index (NCI)"
    )

    ax.set_ylabel(
        "Risk Dimension"
    )

    ax.set_title(
        "Network Contagion Index by Risk Dimension",
        fontsize=17,
        pad=15,
    )

    ax.text(
        0.0,
        -0.13,
        "Higher NCI indicates greater structural connectedness "
        "within the governance-risk co-occurrence network.",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
    )

    ax.grid(
        axis="x",
        alpha=0.25,
        linestyle="--",
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "02_nci_by_dimension.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_temporal_chart(monthly):
    """
    Compare network contagion, Script 6 GCI,
    AI-based EWS, and integrated Governance Contagion Risk.
    """

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    x = monthly["year_month"]

    # -------------------------------------------------------------
    # NCI
    # -------------------------------------------------------------
    ax.plot(
        x,
        monthly["NCI"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="NCI",
    )

    # -------------------------------------------------------------
    # Script 6 GCI
    # -------------------------------------------------------------
    if "GCI" in monthly.columns:

        ax.plot(
            x,
            monthly["GCI"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="GCI",
        )

    # -------------------------------------------------------------
    # AI EWS
    # -------------------------------------------------------------
    ax.plot(
        x,
        monthly["EWS_Score"],
        marker="o",
        markersize=4,
        linewidth=2.5,
        label="AI-Based EWS",
    )

    # -------------------------------------------------------------
    # Governance Contagion Risk
    # -------------------------------------------------------------
    ax.plot(
        x,
        monthly["Governance_Contagion_Risk"],
        marker="o",
        markersize=4,
        linewidth=2.5,
        label="Governance Contagion Risk",
    )

    # -------------------------------------------------------------
    # EWS thresholds
    # -------------------------------------------------------------
    if (
        "EWS_Yellow_Threshold" in monthly.columns
        and "EWS_Red_Threshold" in monthly.columns
    ):

        yellow = float(
            monthly[
                "EWS_Yellow_Threshold"
            ].iloc[0]
        )

        red = float(
            monthly[
                "EWS_Red_Threshold"
            ].iloc[0]
        )

        ax.axhline(
            yellow,
            color="gold",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=f"Yellow threshold ({yellow:.3f})",
        )

        ax.axhline(
            red,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=f"Red threshold ({red:.3f})",
        )

    # -------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------
    ax.set_ylim(
        0,
        1.05
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Score (0–1)"
    )

    ax.set_title(
        "Temporal Network Contagion, AI Early Warning, "
        "and Governance Contagion Risk",
        fontsize=17,
        pad=15,
    )

    ax.tick_params(
        axis="x",
        rotation=60,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--",
    )

    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
    )

    # -------------------------------------------------------------
    # Analytical flow note
    # -------------------------------------------------------------
    ax.text(
        0.99,
        0.02,
        "Temporal network signal → AI-based EWS "
        "→ integrated Governance Contagion Risk",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "03_temporal_nci_ews_gcr.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_ews_chart(monthly):
    """
    AI-based Early Warning Signal.

    Yellow and red thresholds are explicitly colored
    to correspond with their warning levels.
    """

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    x = monthly["year_month"]

    ews = monthly["EWS_Score"]

    # -------------------------------------------------------------
    # Main EWS signal
    # -------------------------------------------------------------
    ax.plot(
        x,
        ews,
        marker="o",
        markersize=5,
        linewidth=2.5,
        label="AI-Based EWS",
    )

    # -------------------------------------------------------------
    # Thresholds
    # -------------------------------------------------------------
    yellow = float(
        monthly[
            "EWS_Yellow_Threshold"
        ].iloc[0]
    )

    red = float(
        monthly[
            "EWS_Red_Threshold"
        ].iloc[0]
    )

    ax.axhline(
        yellow,
        color="gold",
        linestyle="--",
        linewidth=2.2,
        label=f"Yellow threshold ({yellow:.3f})",
    )

    ax.axhline(
        red,
        color="red",
        linestyle="--",
        linewidth=2.2,
        label=f"Red threshold ({red:.3f})",
    )

    # -------------------------------------------------------------
    # Highlight observations crossing thresholds
    # -------------------------------------------------------------
    yellow_mask = (
        (ews >= yellow)
        & (ews < red)
    )

    red_mask = (
        ews >= red
    )

    if yellow_mask.any():

        ax.scatter(
            x[yellow_mask],
            ews[yellow_mask],
            s=75,
            facecolors="gold",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Yellow-level observation",
        )

    if red_mask.any():

        ax.scatter(
            x[red_mask],
            ews[red_mask],
            s=85,
            facecolors="red",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label="Red-level observation",
        )

    # -------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------
    ax.set_ylim(
        0,
        1.05
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "AI-Based EWS Score (0–1)"
    )

    ax.set_title(
        "AI-Based Governance Contagion Early Warning Signal",
        fontsize=17,
        pad=15,
    )

    ax.tick_params(
        axis="x",
        rotation=60,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linestyle="--",
    )

    ax.legend(
        loc="upper left",
        frameon=True,
    )

    # -------------------------------------------------------------
    # Threshold interpretation
    # -------------------------------------------------------------
    ax.text(
        0.99,
        0.02,
        "Yellow = elevated warning signal | "
        "Red = high warning signal",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "04_ai_ews_signal.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
# =============================================================================
# 15. MANIFEST
# =============================================================================

def write_manifest(df, edges, metrics, monthly):
    manifest = {
        "pipeline": "Network & Co-occurrence Governance Contagion Intelligence",
        "research_role": (
            "Analysis 7 — network structure complementing guided semantic "
            "risk mapping, GCI, event, temporal and EWS analyses."
        ),
        "observation_period": {
            "start": "2024-01-01",
            "end": "2026-09-03",
        },
        "input": {
            "document_signals": str(INPUT_DOCUMENT_SIGNALS),
            "monthly_script6_output": str(INPUT_MONTHLY_GCI),
        },
        "documents": int(len(df)),
        "network_dimensions": NETWORK_DIMS,
        "network_nodes": int(len(metrics)),
        "network_edges": int(len(edges)),
        "dimension_threshold": DIMENSION_THRESHOLD,
        "nci_weights": {
            "centrality": NCI_W_CENTRALITY,
            "crisis_edge": NCI_W_CRISIS_EDGE,
            "trust_edge": NCI_W_TRUST_EDGE,
            "betweenness": NCI_W_BETWEENNESS,
        },
        "monthly_periods": int(len(monthly)),
        "anomaly_method": "IsolationForest",
        "anomaly_min_months": MIN_MONTHS_FOR_ANOMALY,
        "ews": {
            "yellow_quantile": YELLOW_Q,
            "red_quantile": RED_Q,
            "formula": (
                "0.45*SEI_reference + 0.35*NCI + 0.20*Anomaly_Score"
            ),
        },
        "governance_contagion_risk": {
            "formula": (
                "0.55*GCI_reference + 0.30*NCI + 0.15*Anomaly_Score"
            ),
            "interpretation": (
                "Analytical proxy for governance contagion exposure; "
                "not an official BPKH risk rating and not causal evidence."
            ),
        },
        "methodological_note": (
            "Co-occurrence is document-level. Network centrality describes "
            "structural association, not causal influence. Clarification/"
            "response is excluded from the risk network because it represents "
            "response/mitigation rather than crisis evidence."
        ),
    }

    with open(
        OUT_DIR / "06_network_run_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# =============================================================================
# 16. MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("ANALYSIS 7 — NETWORK & CO-OCCURRENCE GOVERNANCE CONTAGION")
    print("=" * 78)

    # A. Load Script 6 document-level risk signals.
    print("\n[1/9] Loading Script 6 document-risk output...")
    df = load_document_signals()
    print(f"      Documents: {len(df):,}")

    # B. Build network.
    print("\n[2/9] Building document-level co-occurrence network...")
    edges, G = build_cooccurrence_network(df)
    print(f"      Nodes: {len(G.nodes):,}")
    print(f"      Edges: {len(G.edges):,}")

    # C. Centrality.
    print("\n[3/9] Calculating network centrality...")
    metrics = calculate_network_metrics(G)

    # D. NCI.
    print("\n[4/9] Calculating Network Contagion Index (NCI)...")
    node_nci = calculate_nci(G, edges, metrics)

    # E. Monthly network temporal analysis.
    print("\n[5/9] Building monthly NCI / temporal signals...")
    monthly = build_monthly_nci(df, G, edges, node_nci)

    # F. Merge Script 6 GCI/SEI.
    print("\n[6/9] Merging Script 6 temporal signals...")
    monthly = merge_script6_monthly(monthly)

    # G. Anomaly.
    print("\n[7/9] Running unsupervised anomaly detection...")
    monthly = anomaly_detection(monthly)

    # H. EWS and integrated contagion risk.
    print("\n[8/9] Building AI-Based EWS and Governance Contagion Risk...")
    monthly = build_ews(monthly)
    monthly = build_governance_contagion_risk(monthly)

    # I. Outputs.
    print("\n[9/9] Writing outputs...")

    top_edges = build_top_edges(edges)

    edges.to_csv(
        OUT_DIR / "01_network_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.to_csv(
        OUT_DIR / "02_network_centrality.csv",
        index=False,
        encoding="utf-8-sig",
    )

    node_nci.to_csv(
        OUT_DIR / "03_node_nci.csv",
        index=False,
        encoding="utf-8-sig",
    )

    monthly.to_csv(
        OUT_DIR / "04_monthly_nci_temporal_ews_gcr.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_edges.to_csv(
        OUT_DIR / "05_top_network_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_network_chart(G, node_nci)
    make_nci_chart(node_nci)
    make_temporal_chart(monthly)
    make_ews_chart(monthly)

    write_manifest(df, edges, node_nci, monthly)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)

    print("\nMAIN OUTPUTS:")
    for p in sorted(OUT_DIR.glob("*")):
        print(f"  - {p.name}")

    print("\nLATEST MONTHS:")
    display_cols = [
        c for c in [
            "year_month",
            "NCI",
            "GCI",
            "SEI",
            "Anomaly_Score",
            "EWS_Score",
            "EWS_Level",
            "Governance_Contagion_Risk",
            "Governance_Contagion_Level",
        ]
        if c in monthly.columns
    ]
    print(monthly[display_cols].tail(12).to_string(index=False))

    print("\nTOP NETWORK EDGES:")
    edge_display = [
        c for c in [
            "source_label",
            "target_label",
            "edge_type",
            "cooccurrence_count",
            "association_strength",
        ]
        if c in top_edges.columns
    ]
    print(top_edges[edge_display].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
