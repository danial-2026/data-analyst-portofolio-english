# -*- coding: utf-8 -*-
"""Complete, fail-fast analysis pipeline for the Asia–Africa green-transition panel.

Run 01_prepare_data.py first. This script does not treat a single calendar year as a common policy shock. It audits a
country-specific policy registry and reports policy timing descriptively, but does not
estimate causal policy effects from the incomplete 12-country shortlist. It omits GSynth
unless a valid generalized synthetic-control estimator is specified, and omits MARCOS
pending an audited decision matrix and formula.

Recommended environment (Python 3.10/3.11):
  pip install pandas numpy scipy openpyxl matplotlib seaborn statsmodels scikit-learn
  pip install linearmodels xgboost shap libpysal esda spreg joblib
"""
from __future__ import annotations

import os
import sys
import json
import math
import warnings
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.neighbors import BallTree
import joblib
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# CONFIGURATION — edit only these items unless you know why a specification changes
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get("RESEARCH_READY_FILE", str(BASE_DIR / "Final_Analysis" / "Analysis_Ready_Dataset.xlsx")))
DATA_SHEET = "AnalysisData"
POLICY_REGISTRY_FILE = Path(os.environ.get("RESEARCH_POLICY_REGISTRY_FILE", str(DATA_FILE.parent / "Policy_Registry_To_Complete.xlsx")))
OUTPUT_DIR = Path(os.environ.get("RESEARCH_RESULTS_DIR", str(BASE_DIR / "Final_Analysis" / "Results")))

YEAR_MIN, YEAR_MAX = 2010, 2024
RANDOM_STATE = 42
TEST_COUNTRY_SHARE = 0.20
# Bound parallel workers to avoid severe oversubscription on high-core-count machines.
N_JOBS = max(1, min(4, os.cpu_count() or 1))
XGB_ESTIMATORS = 300
RF_ESTIMATORS = 300
K_NEIGHBORS = 4
MORAN_PERMUTATIONS = 999
MIN_SPATIAL_COUNTRIES = 30
MIN_SPATIAL_YEARS = 8
SHAP_MAX_ROWS = 500
TREE_MAX_DEPTH = 4
TREE_MIN_LEAF_MINIMUM = 10

Y = "Y"
CORE_FEATURES = ["X1", "X2", "X3", "Institution"]
OPTIONAL_CONTROLS = ["ED_log", "HC", "FDI_ihs", "ASI", "GI", "FD"]

# Policy causal identification is intentionally disabled for this paper's current shortlist.
# Policy events differ in type and year, and only about nine candidate AI-strategy countries
# are identified so far; this is too small/selective for a defensible causal panel event study.
# No GSynth is run. A shifted control mean is not a generalized synthetic-control estimator.

# =============================================================================
# Utilities
# =============================================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs() -> None:
    for sub in ["Tables", "Figures", "Models", "Logs", "Data"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def result_to_table(result, model_name: str) -> pd.DataFrame:
    params = result.params
    ses = result.std_errors if hasattr(result, "std_errors") else result.bse
    pvalues = result.pvalues
    try:
        ci = result.conf_int()
        lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
    except Exception:
        lower, upper = pd.Series(np.nan, index=params.index), pd.Series(np.nan, index=params.index)
    out = pd.DataFrame({
        "Model": model_name,
        "Term": params.index.astype(str),
        "Coefficient": np.asarray(params, dtype=float),
        "Std_Error": np.asarray(ses, dtype=float),
        "P_Value": np.asarray(pvalues, dtype=float),
        "CI_Lower_95": np.asarray(lower, dtype=float),
        "CI_Upper_95": np.asarray(upper, dtype=float),
    })
    return out


def concise_statsmodels_report(result, model_label: str, n_clusters: Optional[int] = None) -> str:
    """Create a report without statsmodels' high-dimensional joint-F summary.

    With many fixed-effect dummies and country clustering, statsmodels can warn that the
    covariance matrix for the model-wide joint F test is rank-deficient. We avoid presenting
    that test as a result; coefficient-level estimates, standard errors, p-values, and CIs
    are retained in the result tables.
    """
    try:
        ci = result.conf_int()
        ci_lo = np.asarray(ci.iloc[:, 0], dtype=float)
        ci_hi = np.asarray(ci.iloc[:, 1], dtype=float)
    except Exception:
        ci_lo = np.repeat(np.nan, len(result.params))
        ci_hi = np.repeat(np.nan, len(result.params))
    params = result.params
    bse = result.bse
    pvals = result.pvalues
    table = pd.DataFrame({
        "Coefficient": np.asarray(params, dtype=float),
        "Std_Error": np.asarray(bse, dtype=float),
        "P_Value": np.asarray(pvals, dtype=float),
        "CI_Lower_95": ci_lo,
        "CI_Upper_95": ci_hi,
    }, index=[str(x) for x in params.index])
    header = [
        f"Model: {model_label}",
        f"N observations: {int(result.nobs)}",
        f"R-squared: {float(result.rsquared):.8f}",
        f"Country clusters: {n_clusters if n_clusters is not None else 'not specified'}",
        "Covariance: country-clustered robust; coefficient-level inference reported.",
        "Model-wide joint F test intentionally omitted where high-dimensional FE restrictions can make its covariance rank-deficient.",
        "",
    ]
    return "\n".join(header) + table.to_string(float_format=lambda x: f"{x:.8g}")


def cluster_ols(formula: str, data: pd.DataFrame, model_label: str):
    if data["ID"].nunique() < 20:
        raise ValueError(f"{model_label}: terlalu sedikit cluster negara ({data['ID'].nunique()}).")
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["ID"], "use_correction": True})
    save_text(OUTPUT_DIR / "Logs" / f"{safe_slug(model_label)}.txt",
              concise_statsmodels_report(model, model_label, n_clusters=data["ID"].nunique()))
    return model


def choose_available_controls(df: pd.DataFrame) -> list[str]:
    out = []
    for col in OPTIONAL_CONTROLS:
        if col in df.columns and df[col].notna().sum() > 0 and df[col].nunique(dropna=True) > 1:
            out.append(col)
    return out


def find_coord_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    lat = next((c for c in ["Latitude", "latitude", "Lat"] if c in df.columns), None)
    lon = next((c for c in ["Longitude", "longitude", "Lon", "Long"] if c in df.columns), None)
    return lat, lon


def build_knn_weights(ids: list[str], lat_lon: np.ndarray, k: int):
    """Build binary k-nearest-neighbour W using great-circle neighbour ranking (haversine)."""
    try:
        from libpysal.weights import W
    except ImportError as exc:
        raise ImportError("Spatial analysis requires libpysal: pip install libpysal esda spreg") from exc
    if len(ids) <= k:
        raise ValueError(f"Need more than k={k} countries to create KNN weights; have {len(ids)}.")
    coords_rad = np.radians(lat_lon)  # [latitude, longitude]
    tree = BallTree(coords_rad, metric="haversine")
    inds = tree.query(coords_rad, k=k + 1, return_distance=False)
    neighbours = {ids[i]: [ids[j] for j in inds[i] if j != i][:k] for i in range(len(ids))}
    w = W(neighbours, id_order=ids)
    w.transform = "r"
    return w


def panel_fe_models(df: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    """Two-way country/year FE with country-clustered SEs.

    Preferred estimator: linearmodels.PanelOLS. If unavailable, use the equivalent
    dummy-variable OLS specification with statsmodels and country-clustered SEs.
    The estimator used is written into the output table/log.
    """
    try:
        from linearmodels.panel import PanelOLS
        use_panelols = True
    except ImportError:
        PanelOLS = None
        use_panelols = False
        log("WARNING: linearmodels not installed; using statsmodels OLS with country/year dummies and country-clustered standard errors for two-way FE.")

    model_rows = []
    data_specs = [
        ("FE_Baseline", CORE_FEATURES),
        ("FE_With_Controls", CORE_FEATURES + controls),
    ]
    for name, rhs in data_specs:
        needed = [Y, "ID", "Year"] + rhs
        sample = df[needed].dropna().copy()
        if sample["ID"].nunique() < 20 or sample["Y"].nunique() < 3:
            log(f"SKIP {name}: insufficient usable sample.")
            continue
        try:
            if use_panelols:
                pdata = sample.set_index(["ID", "Year"]).sort_index()
                if pdata.index.has_duplicates:
                    raise ValueError(f"{name}: duplicate country-year index after sample selection.")
                formula = f"Y ~ 1 + {' + '.join(rhs)} + EntityEffects + TimeEffects"
                mod = PanelOLS.from_formula(formula, data=pdata, drop_absorbed=True, check_rank=True)
                res = mod.fit(cov_type="clustered", cluster_entity=True, debiased=True, group_debias=True)
                estimator = "linearmodels.PanelOLS; entity + time FE; cluster by country"
                within_r2 = float(res.rsquared_within)
                summary_text = str(res.summary)
                residuals = res.resids.rename("Residual").reset_index()
                residuals.columns = ["ID", "Year", "Residual"]
            else:
                # Explicit country/year dummies are an algebraically equivalent TWFE specification.
                formula = f"Y ~ 1 + {' + '.join(rhs)} + C(ID) + C(Year)"
                res = smf.ols(formula, data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample["ID"], "use_correction": True})
                estimator = "statsmodels OLS with country + year dummies; cluster by country"
                within_r2 = np.nan
                summary_text = concise_statsmodels_report(res, name, n_clusters=sample["ID"].nunique())
                residuals = sample[["ID", "Year"]].copy()
                residuals["Residual"] = np.asarray(res.resid, dtype=float)
            save_text(OUTPUT_DIR / "Logs" / f"{name}_summary.txt", summary_text)
            table = result_to_table(res, name)
            table["N_Observations"] = int(res.nobs)
            table["N_Countries"] = int(sample["ID"].nunique())
            table["R2_Within"] = within_r2
            table["Estimator"] = estimator
            model_rows.append(table)
            try:
                joblib.dump(res, OUTPUT_DIR / "Models" / f"{name}.joblib")
            except Exception:
                pass
            if name == "FE_With_Controls":
                residuals["ID"] = residuals["ID"].astype(str)
                residuals["Year"] = pd.to_numeric(residuals["Year"], errors="coerce").astype(int)
                residuals.to_excel(OUTPUT_DIR / "Data" / "Panel_FE_With_Controls_Residuals.xlsx", index=False)
        except Exception as exc:
            save_text(OUTPUT_DIR / "Logs" / f"{name}_ERROR.txt", traceback.format_exc())
            log(f"WARNING {name} failed: {exc}")
    return pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()


def calculate_vif(df: pd.DataFrame, regressors: list[str]) -> pd.DataFrame:
    if len(regressors) < 2:
        return pd.DataFrame(columns=["Variable", "VIF"])
    tmp = df[regressors].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < max(20, len(regressors) + 5):
        return pd.DataFrame(columns=["Variable", "VIF"])
    # VIF is a screening diagnostic only; not itself a model-selection test.
    x_raw = tmp.to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(tmp)), x_raw])
    rows = []
    for i, col in enumerate(regressors, start=1):
        try:
            val = float(variance_inflation_factor(x, i))
        except Exception:
            val = np.nan
        rows.append({"Variable": col, "VIF": val, "N": len(tmp)})
    return pd.DataFrame(rows)


def run_path_analysis(df: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    rhs_ctrl = " + ".join(controls)
    ctrl_tail = f" + {rhs_ctrl}" if rhs_ctrl else ""
    fe = " + C(ID) + C(Year)"
    specs = [
        ("Path_A_X1_to_X2", f"X2 ~ X1 + Institution{ctrl_tail}{fe}", ["X2", "X1", "Institution", *controls]),
        ("Path_B_X2_to_X3", f"X3 ~ X2 + X1 + Institution{ctrl_tail}{fe}", ["X3", "X2", "X1", "Institution", *controls]),
        ("Outcome_All_Core", f"Y ~ X1 + X2 + X3 + Institution{ctrl_tail}{fe}", ["Y", "X1", "X2", "X3", "Institution", *controls]),
    ]
    if "X3_innovation_only" in df.columns:
        specs.append(("Outcome_InnovationOnly_Sensitivity", f"Y ~ X1 + X2 + X3_innovation_only + Institution{ctrl_tail}{fe}", ["Y", "X1", "X2", "X3_innovation_only", "Institution", *controls]))
    rows = []
    for label, formula, variables in specs:
        variables = list(dict.fromkeys([*variables, "ID", "Year"]))
        sample = df[variables].dropna().copy()
        if sample["ID"].nunique() < 20:
            log(f"SKIP path model {label}: too few countries.")
            continue
        try:
            res = cluster_ols(formula, sample, label)
            tab = pd.DataFrame({
                "Model": label, "Term": res.params.index.astype(str),
                "Coefficient": res.params.values, "Std_Error": res.bse.values,
                "P_Value": res.pvalues.values, "CI_Lower_95": res.conf_int()[0].values,
                "CI_Upper_95": res.conf_int()[1].values, "N_Observations": int(res.nobs),
                "N_Countries": int(sample["ID"].nunique()), "R2": float(res.rsquared),
            })
            rows.append(tab)
        except Exception as exc:
            log(f"WARNING path model {label} failed: {exc}")
            save_text(OUTPUT_DIR / "Logs" / f"{safe_slug(label)}_ERROR.txt", traceback.format_exc())

    # Moderation: interaction is explicitly mean-centred, and interpretation remains conditional.
    mod_cols = [Y, "X3", "Institution", "ID", "Year", *controls]
    sample = df[mod_cols].dropna().copy()
    if sample["ID"].nunique() >= 20:
        sample["X3_c"] = sample["X3"] - sample["X3"].mean()
        sample["Institution_c"] = sample["Institution"] - sample["Institution"].mean()
        sample["X3_Inst"] = sample["X3_c"] * sample["Institution_c"]
        formula = f"Y ~ X3_c + Institution_c + X3_Inst{ctrl_tail}{fe}"
        try:
            res = cluster_ols(formula, sample, "Moderation_X3_by_Institution")
            rows.append(pd.DataFrame({
                "Model": "Moderation_X3_by_Institution", "Term": res.params.index.astype(str),
                "Coefficient": res.params.values, "Std_Error": res.bse.values,
                "P_Value": res.pvalues.values, "CI_Lower_95": res.conf_int()[0].values,
                "CI_Upper_95": res.conf_int()[1].values, "N_Observations": int(res.nobs),
                "N_Countries": int(sample["ID"].nunique()), "R2": float(res.rsquared),
            }))
        except Exception as exc:
            log(f"WARNING moderation model failed: {exc}")
            save_text(OUTPUT_DIR / "Logs" / "Moderation_ERROR.txt", traceback.format_exc())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_spatial_diagnostics_and_models(df: pd.DataFrame, controls: list[str]) -> None:
    """Moran's I by year plus optional fixed-effects SAR, SEM, and SDM via PySAL."""
    lat_col, lon_col = find_coord_columns(df)
    if not lat_col or not lon_col:
        save_text(OUTPUT_DIR / "Logs" / "Spatial_Status.txt", "SKIPPED: latitude/longitude columns not found. Add verified coordinates or a defensible polygon-based W matrix.")
        log("Spatial analyses skipped: coordinates are unavailable.")
        return
    try:
        from esda import Moran
    except ImportError:
        save_text(OUTPUT_DIR / "Logs" / "Spatial_Status.txt", "SKIPPED: install libpysal, esda, and spreg to run spatial diagnostics/models.")
        log("Spatial diagnostics skipped: install libpysal, esda, spreg.")
        return

    coords_by_id = df[["ID", lat_col, lon_col]].dropna().drop_duplicates()
    if coords_by_id["ID"].duplicated().any():
        raise ValueError("Coordinates vary across observations for one ID. Resolve conflicting coordinates before spatial analysis.")
    coords_by_id = coords_by_id[(coords_by_id[lat_col].between(-90, 90)) & (coords_by_id[lon_col].between(-180, 180))].copy()
    all_years_for_moran = sorted(int(y) for y in df["Year"].dropna().unique())
    residual_path = OUTPUT_DIR / "Data" / "Panel_FE_With_Controls_Residuals.xlsx"
    residual_data = pd.read_excel(residual_path) if residual_path.exists() else None

    # Use a common set of countries and a fixed W across years for each Moran series.
    # Rebuilding a different KNN matrix every year would make cross-year I comparisons
    # confound changes in Y with changes in the sample/weights matrix.
    y_complete_ids = []
    for country_id, group in df.groupby("ID"):
        vals = group.loc[group["Year"].isin(all_years_for_moran), ["Year", Y]].dropna()
        if set(vals["Year"].astype(int)) == set(all_years_for_moran) and vals["Year"].nunique() == len(all_years_for_moran):
            y_complete_ids.append(str(country_id))
    y_common_ids = sorted(set(y_complete_ids).intersection(set(coords_by_id["ID"].astype(str))))
    w_y_common = None
    if len(y_common_ids) >= MIN_SPATIAL_COUNTRIES:
        coords_y = coords_by_id.set_index("ID").loc[y_common_ids, [lat_col, lon_col]].to_numpy(float)
        w_y_common = build_knn_weights(y_common_ids, coords_y, min(K_NEIGHBORS, len(y_common_ids) - 1))

    residual_common_ids = []
    w_residual_common = None
    if residual_data is not None and {"ID", "Year", "Residual"}.issubset(residual_data.columns):
        residual_data["ID"] = residual_data["ID"].astype(str).str.strip().str.upper()
        residual_data["Year"] = pd.to_numeric(residual_data["Year"], errors="coerce")
        residual_data["Residual"] = pd.to_numeric(residual_data["Residual"], errors="coerce")
        for country_id, group in residual_data.groupby("ID"):
            vals = group.loc[group["Year"].isin(all_years_for_moran), ["Year", "Residual"]].dropna()
            if set(vals["Year"].astype(int)) == set(all_years_for_moran) and vals["Year"].nunique() == len(all_years_for_moran):
                residual_common_ids.append(str(country_id))
        residual_common_ids = sorted(set(residual_common_ids).intersection(set(coords_by_id["ID"].astype(str))))
        if len(residual_common_ids) >= MIN_SPATIAL_COUNTRIES:
            coords_r = coords_by_id.set_index("ID").loc[residual_common_ids, [lat_col, lon_col]].to_numpy(float)
            w_residual_common = build_knn_weights(residual_common_ids, coords_r, min(K_NEIGHBORS, len(residual_common_ids) - 1))

    moran_rows = []
    for year in all_years_for_moran:
        row = {"Year": int(year), "N_Countries_Y": len(y_common_ids), "Moran_I_Y": np.nan,
               "P_Y_Permutation": np.nan, "N_Countries_FE_Residual": len(residual_common_ids),
               "Moran_I_FE_Residual": np.nan, "P_Residual_Permutation": np.nan, "Status": "OK"}
        if w_y_common is not None:
            y_lookup = df.loc[df["Year"].eq(year), ["ID", Y]].dropna().set_index("ID")[Y]
            y_values = y_lookup.reindex(y_common_ids)
            if y_values.notna().all():
                try:
                    np.random.seed(RANDOM_STATE + int(year))
                    mi_y = Moran(y_values.to_numpy(float), w_y_common, permutations=MORAN_PERMUTATIONS)
                    row["Moran_I_Y"] = float(mi_y.I)
                    row["P_Y_Permutation"] = float(mi_y.p_sim)
                except Exception as exc:
                    row["Status"] = f"Y Moran failed: {exc}"
            else:
                row["Status"] = "Y Moran skipped: common-sample outcome missing in this year"
        else:
            row["Status"] = f"Y Moran skipped: fewer than {MIN_SPATIAL_COUNTRIES} countries with Y and coordinates in every year"

        if w_residual_common is not None and residual_data is not None:
            r_lookup = residual_data.loc[residual_data["Year"].eq(year), ["ID", "Residual"]].dropna().set_index("ID")["Residual"]
            r_values = r_lookup.reindex(residual_common_ids)
            if r_values.notna().all():
                try:
                    np.random.seed(RANDOM_STATE + 10000 + int(year))
                    mi_r = Moran(r_values.to_numpy(float), w_residual_common, permutations=MORAN_PERMUTATIONS)
                    row["Moran_I_FE_Residual"] = float(mi_r.I)
                    row["P_Residual_Permutation"] = float(mi_r.p_sim)
                except Exception as exc:
                    row["Status"] += f"; residual Moran failed: {exc}"
            else:
                row["Status"] += "; residual Moran skipped: common-sample residual missing in this year"
        elif residual_data is not None:
            row["Status"] += f"; residual Moran skipped: fewer than {MIN_SPATIAL_COUNTRIES} countries with FE residuals and coordinates in every year"
        moran_rows.append(row)
    pd.DataFrame(moran_rows).to_excel(OUTPUT_DIR / "Tables" / "Spatial_Morans_I_by_Year.xlsx", index=False)

    # Models require a balanced complete-case country x year panel and verified coordinates.
    spatial_vars = [Y, *CORE_FEATURES, *controls]
    spatial_data = df[["Country", "ID", "Year", lat_col, lon_col, *spatial_vars]].copy()
    for col in spatial_vars + [lat_col, lon_col]:
        spatial_data[col] = pd.to_numeric(spatial_data[col], errors="coerce") if col not in ["Country", "ID"] else spatial_data[col]
    spatial_data = spatial_data.dropna(subset=[*spatial_vars, lat_col, lon_col]).copy()
    all_years = sorted(int(y) for y in df["Year"].dropna().unique())
    valid_ids = []
    for country_id, g in spatial_data.groupby("ID"):
        if sorted(g["Year"].unique().astype(int).tolist()) == all_years and len(g) == len(all_years):
            valid_ids.append(country_id)
    spatial_data = spatial_data[spatial_data["ID"].isin(valid_ids)].copy()
    spatial_data = spatial_data.sort_values(["ID", "Year"])
    if len(valid_ids) < MIN_SPATIAL_COUNTRIES or len(all_years) < MIN_SPATIAL_YEARS:
        save_text(OUTPUT_DIR / "Logs" / "Spatial_Status.txt",
                  f"Moran's I attempted by year. Spatial panel models SKIPPED: only {len(valid_ids)} countries have verified coordinates and complete data for all {len(all_years)} years. Thresholds: {MIN_SPATIAL_COUNTRIES} countries and {MIN_SPATIAL_YEARS} years.")
        log(f"Spatial panel models skipped: {len(valid_ids)} complete geocoded countries.")
        return

    try:
        import spreg
    except ImportError:
        save_text(OUTPUT_DIR / "Logs" / "Spatial_Status.txt", "Moran's I completed. Spatial panel models skipped: install spreg (pip install spreg).")
        return

    ids = sorted(valid_ids)
    years = all_years
    geo = spatial_data.drop_duplicates("ID").set_index("ID").loc[ids]
    coords = geo[[lat_col, lon_col]].to_numpy(float)
    w = build_knn_weights(ids, coords, K_NEIGHBORS)
    w_matrix = w.sparse.toarray()
    n, t = len(ids), len(years)

    def wide_matrix(variable: str) -> np.ndarray:
        p = spatial_data.pivot(index="ID", columns="Year", values=variable).reindex(index=ids, columns=years)
        arr = p.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"Spatial panel has missing/infinite values in {variable} after balanced-sample filtering.")
        return arr

    ywide = wide_matrix(Y)
    base_vars = [*CORE_FEATURES, *controls]
    x_matrices = {v: wide_matrix(v) for v in base_vars}
    year_matrices = {}
    for yr in years[1:]:
        year_vec = np.array([1.0 if x == yr else 0.0 for x in years], dtype=float)
        year_matrices[f"YearFE_{yr}"] = np.tile(year_vec, (n, 1))

    model_specs = {
        "SAR_FE": [(v, x_matrices[v]) for v in base_vars] + list(year_matrices.items()),
        "SDM_FE": ([(v, x_matrices[v]) for v in base_vars]
                   + [(f"W_{v}", w_matrix @ x_matrices[v]) for v in base_vars]
                   + list(year_matrices.items())),
        "SEM_FE": [(v, x_matrices[v]) for v in base_vars] + list(year_matrices.items()),
    }
    spatial_summary_rows = []
    status_lines = [f"KNN W: k={K_NEIGHBORS}; row-standardized; haversine nearest-neighbour ranking.",
                    f"Model sample: {n} countries x {t} years = {n*t} observations.",
                    "Spatial FE estimator absorbs country fixed effects; explicit year-dummy blocks are included."]
    for model_name, blocks in model_specs.items():
        try:
            # Each exogenous variable is represented by its T annual values, block by block.
            xwide = np.hstack([matrix for _, matrix in blocks])
            xnames = [f"{name}_{yr}" for name, _ in blocks for yr in years]
            if model_name == "SEM_FE":
                model = spreg.Panel_FE_Error(ywide, xwide, w, name_y=[f"Y_{yr}" for yr in years], name_x=xnames, name_w="KNN4", vm=True)
            else:
                # SAR: spatial lag of Y. SDM: spatial lag of Y plus spatially lagged X blocks.
                model = spreg.Panel_FE_Lag(ywide, xwide, w, name_y=[f"Y_{yr}" for yr in years], name_x=xnames, name_w="KNN4", vm=True)
            save_text(OUTPUT_DIR / "Logs" / f"{model_name}_summary.txt", str(getattr(model, "summary", model)))
            spatial_summary_rows.append({
                "Model": model_name, "N_Countries": int(n), "N_Years": int(t), "N_Observations": int(n*t),
                "Spatial_Rho_or_Lambda": float(getattr(model, "rho", getattr(model, "lam", np.nan))),
                "LogLik": float(getattr(model, "logll", np.nan)), "AIC": float(getattr(model, "aic", np.nan)),
                "BIC": float(getattr(model, "schwarz", np.nan)), "Status": "Estimated; inspect full summary",
            })
            # Coefficient extraction is guarded because PySAL versions differ in parameter naming.
            betas = np.asarray(getattr(model, "betas", []), dtype=float).reshape(-1)
            zstats = getattr(model, "z_stat", [])
            names = list(getattr(model, "name_x", []))
            if len(names) == len(betas):
                param_names = names
            elif len(names) == len(betas) - 1:
                param_names = names + ["Spatial_parameter"]
            else:
                param_names = [f"parameter_{i+1}" for i in range(len(betas))]
            coef_rows = []
            for i, value in enumerate(betas):
                zval, pval = (np.nan, np.nan)
                if isinstance(zstats, (list, tuple)) and i < len(zstats) and len(zstats[i]) >= 2:
                    zval, pval = zstats[i][0], zstats[i][1]
                coef_rows.append({"Model": model_name, "Parameter": param_names[i] if i < len(param_names) else f"parameter_{i+1}",
                                  "Estimate": float(value), "Z_Statistic": float(zval) if np.isfinite(zval) else np.nan,
                                  "P_Value": float(pval) if np.isfinite(pval) else np.nan})
            pd.DataFrame(coef_rows).to_excel(OUTPUT_DIR / "Tables" / f"{model_name}_coefficients.xlsx", index=False)

            # SDM direct/indirect/total impacts are calculated only if model parameters can be mapped
            # unambiguously to beta_X, theta_WX and rho. This avoids assigning the wrong coefficient.
            if model_name == "SDM_FE":
                param_map = {str(nm): float(val) for nm, val in zip(param_names, betas)}
                def find_param(target: str) -> Optional[float]:
                    exact = [v for k, v in param_map.items() if k == target or k.startswith(target + "_")]
                    return float(exact[0]) if len(exact) == 1 else None
                rho = float(getattr(model, "rho", np.nan))
                impact_rows = []
                if np.isfinite(rho) and abs(rho) < 0.98:
                    identity = np.eye(n)
                    try:
                        multiplier = np.linalg.inv(identity - rho * w_matrix)
                        for variable in base_vars:
                            beta = find_param(variable)
                            theta = find_param(f"W_{variable}")
                            if beta is None or theta is None:
                                continue
                            impact = multiplier @ (beta * identity + theta * w_matrix)
                            direct = float(np.trace(impact) / n)
                            total = float(impact.sum(axis=1).mean())
                            impact_rows.append({"Variable": variable, "Direct_Effect_Point": direct,
                                                "Indirect_Effect_Point": total - direct, "Total_Effect_Point": total,
                                                "Inference": "Point estimates only; confidence intervals not computed"})
                        if impact_rows:
                            pd.DataFrame(impact_rows).to_excel(OUTPUT_DIR / "Tables" / "SDM_FE_Point_Impacts_NO_SE.xlsx", index=False)
                        else:
                            status_lines.append("SDM impacts not calculated: parameter labels could not be mapped uniquely to beta_X/theta_WX.")
                    except np.linalg.LinAlgError:
                        status_lines.append("SDM impacts not calculated: (I - rho W) not invertible for estimated rho.")
                else:
                    status_lines.append("SDM impacts not calculated: estimated spatial rho missing or outside conservative invertibility guard.")
        except Exception as exc:
            status_lines.append(f"{model_name} FAILED: {type(exc).__name__}: {exc}")
            save_text(OUTPUT_DIR / "Logs" / f"{model_name}_ERROR.txt", traceback.format_exc())
            log(f"WARNING {model_name} failed: {exc}")
    pd.DataFrame(spatial_summary_rows).to_excel(OUTPUT_DIR / "Tables" / "Spatial_Panel_Model_Comparison.xlsx", index=False)
    save_text(OUTPUT_DIR / "Logs" / "Spatial_Status.txt", "\n".join(status_lines))


def run_policy_registry_diagnostics(df: pd.DataFrame) -> None:
    """Audit policy timing and provide a descriptive event-time profile, never a causal ATT.

    The supplied list is selective and mixes several types of interventions. Candidate dates
    are audit entries, not automatically verified treatment dates. Until the full country
    roster has verified treated and never-treated status and consistent adoption dates, a
    causal staggered DiD is not estimated. The chart/table are descriptive only.
    """
    if not POLICY_REGISTRY_FILE.exists():
        save_text(OUTPUT_DIR / "Logs" / "Policy_Registry_Status.txt",
                  f"Policy registry not found at {POLICY_REGISTRY_FILE}. Run 01_prepare_data.py first.")
        return
    try:
        registry = pd.read_excel(POLICY_REGISTRY_FILE, sheet_name="CountryRegistry")
        inventory = pd.read_excel(POLICY_REGISTRY_FILE, sheet_name="EventInventory")
    except Exception as exc:
        save_text(OUTPUT_DIR / "Logs" / "Policy_Registry_Status.txt", f"Could not read registry: {exc}")
        return
    if "ID" not in registry.columns:
        save_text(OUTPUT_DIR / "Logs" / "Policy_Registry_Status.txt", "Policy registry lacks ID column; no policy analysis performed.")
        return
    registry["ID"] = registry["ID"].astype(str).str.strip().str.upper()
    if registry["ID"].duplicated().any():
        raise ValueError("Policy registry contains duplicate ID values; resolve manually before analysis.")
    panel_ids = set(df["ID"].astype(str).str.strip().str.upper())
    registry_ids = set(registry["ID"])
    missing_ids = sorted(panel_ids - registry_ids)
    extra_ids = sorted(registry_ids - panel_ids)
    status_col = "AI_Treatment_Status"
    reviewed_col = "AI_Treatment_Reviewed"
    year_col = "AI_Treatment_Year"
    required_cols = [status_col, reviewed_col, year_col]
    missing_cols = [c for c in required_cols if c not in registry.columns]
    if missing_cols:
        save_text(OUTPUT_DIR / "Logs" / "Policy_Registry_Status.txt", f"Registry missing required columns {missing_cols}; descriptive-only status unavailable.")
        return

    matched = registry[registry["ID"].isin(panel_ids)].copy()
    matched[year_col] = pd.to_numeric(matched[year_col], errors="coerce")
    matched[reviewed_col] = pd.to_numeric(matched[reviewed_col], errors="coerce").fillna(0).astype(int)
    status_counts = matched[status_col].fillna("missing_status").astype(str).value_counts(dropna=False).rename_axis("AI_Treatment_Status").reset_index(name="Countries")
    status_counts.to_excel(OUTPUT_DIR / "Tables" / "Policy_Registry_Status_Counts.xlsx", index=False)
    matched.to_excel(OUTPUT_DIR / "Data" / "Policy_Registry_Matched_To_Sample.xlsx", index=False)

    status_values = matched[status_col].fillna("missing_status").astype(str)
    never_with_year = int((status_values.eq("never_treated_verified") & matched[year_col].notna()).sum())
    treated_without_year = int((status_values.eq("treated_verified") & matched[year_col].isna()).sum())
    if never_with_year or treated_without_year:
        log(f"WARNING policy registry has inconsistent status/year rows: never-treated with a treatment year={never_with_year}; treated with no year={treated_without_year}. Fix CountryRegistry before any causal evaluation.")

    descriptive_candidate_statuses = {"candidate_treated_review", "treated_verified"}
    candidate_mask = matched[year_col].notna() & status_values.isin(descriptive_candidate_statuses)
    cohort = (matched[candidate_mask]
              .groupby(year_col).agg(Candidate_Countries=("ID", "nunique"), Reviewed_Countries=(reviewed_col, "sum"))
              .reset_index().rename(columns={year_col: "Candidate_AI_Strategy_Year"}))
    if not cohort.empty:
        cohort.to_excel(OUTPUT_DIR / "Tables" / "Candidate_AI_Strategy_Cohort_Counts.xlsx", index=False)

    # Descriptive event-time averages around candidate AI-strategy years. Country baseline
    # is k=-1; values are mean within-country changes from that baseline, not treatment effects.
    candidates = matched[candidate_mask][["ID", year_col, status_col, reviewed_col]].copy()
    if not candidates.empty:
        candidates[year_col] = pd.to_numeric(candidates[year_col], errors="coerce")
        outcome = df[["ID", "Year", Y]].copy()
        outcome["ID"] = outcome["ID"].astype(str).str.strip().str.upper()
        outcome["Year"] = pd.to_numeric(outcome["Year"], errors="coerce")
        outcome[Y] = pd.to_numeric(outcome[Y], errors="coerce")
        outcome = outcome.dropna(subset=["ID", "Year", Y])
        records = []
        for _, row in candidates.dropna(subset=[year_col]).iterrows():
            country_id, g = str(row["ID"]), int(row[year_col])
            cdat = outcome[outcome["ID"].eq(country_id)].set_index("Year")[Y]
            baseline = cdat.get(g - 1, np.nan)
            if pd.isna(baseline):
                continue
            for rel_t in range(-5, 6):
                year = g + rel_t
                value = cdat.get(year, np.nan)
                if pd.notna(value):
                    records.append({"ID": country_id, "Candidate_Policy_Year": g, "Event_Time": rel_t,
                                    "Year": year, "Y": float(value), "Change_from_t_minus_1": float(value - baseline),
                                    "Registry_Status": str(row[status_col]), "Reviewed": int(row[reviewed_col])})
        profile = pd.DataFrame(records)
        if not profile.empty:
            agg = profile.groupby("Event_Time").agg(
                N_Countries=("ID", "nunique"), Mean_Y=("Y", "mean"), SD_Y=("Y", "std"),
                Mean_Change_from_t_minus_1=("Change_from_t_minus_1", "mean"),
                SD_Change_from_t_minus_1=("Change_from_t_minus_1", "std"),
            ).reset_index()
            agg["SE_Descriptive"] = agg["SD_Change_from_t_minus_1"] / np.sqrt(agg["N_Countries"].clip(lower=1))
            agg["CI_Lower_95_Descriptive"] = agg["Mean_Change_from_t_minus_1"] - 1.96 * agg["SE_Descriptive"]
            agg["CI_Upper_95_Descriptive"] = agg["Mean_Change_from_t_minus_1"] + 1.96 * agg["SE_Descriptive"]
            agg["Interpretation"] = "Descriptive within-country change around candidate policy year; not causal ATT; CI is not cluster-robust causal inference."
            agg.to_excel(OUTPUT_DIR / "Tables" / "AI_Policy_EventTime_Descriptive_Not_Causal.xlsx", index=False)
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.errorbar(agg["Event_Time"], agg["Mean_Change_from_t_minus_1"],
                        yerr=1.96 * agg["SE_Descriptive"], fmt="o-", capsize=3)
            ax.axhline(0, color="gray", linestyle="--")
            ax.axvline(0, color="red", linestyle=":")
            ax.set(title="Descriptive outcome profile around candidate AI strategy years",
                   xlabel="Years relative to country-specific candidate year", ylabel="Mean change in Y from t = -1")
            fig.text(0.5, 0.01, "DESCRIPTIVE ONLY — no control group; not a causal treatment effect.", ha="center", fontsize=9)
            fig.tight_layout(rect=[0, 0.04, 1, 1])
            fig.savefig(OUTPUT_DIR / "Figures" / "AI_Policy_EventTime_Descriptive_Not_Causal.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    acceptable_status = {"treated_verified", "never_treated_verified"}
    status_clean = matched[status_col].fillna("missing_status").astype(str)
    status_year_consistent = (
        (status_clean.eq("treated_verified") & matched[year_col].notna())
        | (status_clean.eq("never_treated_verified") & matched[year_col].isna())
    )
    all_reviewed = (not missing_ids and status_clean.isin(acceptable_status).all()
                    and matched[reviewed_col].eq(1).all() and status_year_consistent.all())
    n_treated = int((status_clean.eq("treated_verified") & matched[year_col].notna()).sum())
    n_never = int((status_clean.eq("never_treated_verified") & matched[year_col].isna()).sum())
    lines = [
        f"Registry file: {POLICY_REGISTRY_FILE}",
        f"Panel countries: {len(panel_ids)}; registry countries matched: {matched['ID'].nunique()}; missing IDs: {missing_ids}; extra IDs: {extra_ids}",
        f"Candidate AI-strategy years in seed: {int(matched[year_col].notna().sum())}",
        f"Fully reviewed treated countries: {n_treated}; verified never-treated countries: {n_never}",
        f"All panel-country treatment statuses reviewed and year-consistent: {all_reviewed}",
        f"Rows inconsistent with status/year rules: never-treated with year={never_with_year}; treated without year={treated_without_year}",
        "Causal staggered DiD: NOT ESTIMATED. The initial candidate list is selective, policy events differ by type/year, and the treated-country count is currently too small for reliable country-clustered inference. A full country-by-country policy search and a pre-specified identification plan are prerequisites.",
        "The event-time plot/table, if produced, summarize within-country changes from t=-1 and are descriptive only; do not report them as an ATT or policy effect.",
    ]
    save_text(OUTPUT_DIR / "Logs" / "Policy_Registry_Status.txt", "\n".join(lines))
    log("Policy registry audited. Causal policy effects were not estimated; see Policy_Registry_Status.txt.")


def train_and_explain_ml(df: pd.DataFrame, controls: list[str]) -> None:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("ML section requires xgboost: pip install xgboost") from exc

    features = [c for c in CORE_FEATURES + controls if c in df.columns]
    required = [Y, "ID", "Country", "Year", *features]
    model_data = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if model_data["ID"].nunique() < 30 or len(model_data) < 200:
        raise ValueError(f"Insufficient complete cases for ML: {len(model_data)} rows, {model_data['ID'].nunique()} countries.")
    model_data = model_data.sort_values(["ID", "Year"]).reset_index(drop=True)
    # Outer test countries are untouched until final evaluation. Model selection is made
    # only on an inner country-held-out validation split within the outer training set.
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_COUNTRY_SHARE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(model_data[features], model_data[Y], groups=model_data["ID"]))
    train = model_data.iloc[train_idx].copy()
    test = model_data.iloc[test_idx].copy()
    if train["ID"].nunique() < 10 or test["ID"].nunique() < 5:
        raise ValueError("Country-held-out ML split yielded too few train/test countries. Review sample size and split ratio.")
    X_train, y_train = train[features], train[Y]
    X_test, y_test = test[features], test[Y]

    def make_models():
        return {
            "XGBoost": XGBRegressor(n_estimators=XGB_ESTIMATORS, max_depth=4, learning_rate=0.03,
                                    subsample=0.80, colsample_bytree=0.80,
                                    reg_lambda=1.0, objective="reg:squarederror",
                                    random_state=RANDOM_STATE, n_jobs=N_JOBS),
            "RandomForest": RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=8,
                                                  min_samples_leaf=3, max_features=1.0,
                                                  random_state=RANDOM_STATE, n_jobs=N_JOBS),
        }

    inner_splitter = GroupShuffleSplit(n_splits=5, test_size=0.20, random_state=RANDOM_STATE + 1)
    inner_scores = []
    for fold_id, (inner_tr_idx, inner_va_idx) in enumerate(inner_splitter.split(X_train, y_train, groups=train["ID"]), start=1):
        inner_train = train.iloc[inner_tr_idx]
        inner_valid = train.iloc[inner_va_idx]
        for name, candidate in make_models().items():
            candidate.fit(inner_train[features], inner_train[Y])
            val_pred = candidate.predict(inner_valid[features])
            inner_scores.append({"Model": name, "Inner_Fold": fold_id,
                                 "Inner_Validation_RMSE": float(np.sqrt(mean_squared_error(inner_valid[Y], val_pred))),
                                 "Inner_Validation_MAE": float(mean_absolute_error(inner_valid[Y], val_pred)),
                                 "N_Inner_Train_Countries": int(inner_train["ID"].nunique()),
                                 "N_Inner_Validation_Countries": int(inner_valid["ID"].nunique())})
    inner_scores_df = pd.DataFrame(inner_scores)
    selection = (inner_scores_df.groupby("Model", as_index=False)
                 .agg(Inner_Validation_RMSE_Mean=("Inner_Validation_RMSE", "mean"),
                      Inner_Validation_RMSE_SD=("Inner_Validation_RMSE", "std"),
                      Inner_Validation_MAE_Mean=("Inner_Validation_MAE", "mean"),
                      Inner_Folds=("Inner_Fold", "nunique"))
                 .sort_values(["Inner_Validation_RMSE_Mean", "Inner_Validation_MAE_Mean"]).reset_index(drop=True))
    inner_scores_df.to_excel(OUTPUT_DIR / "Tables" / "ML_Inner_Validation_Fold_Scores.xlsx", index=False)
    best_name = str(selection.iloc[0]["Model"])
    selection.to_excel(OUTPUT_DIR / "Tables" / "ML_Inner_Model_Selection.xlsx", index=False)

    # Refit both models on all outer training countries and evaluate once on the untouched
    # outer test countries. The outer test metrics do not select the winning model.
    models = make_models()
    metric_rows = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        inner_row = selection[selection["Model"].eq(name)].iloc[0]
        metric_rows.append({
            "Model": name, "Validation": "Outer hold-out of entire countries",
            "Selected_By_Inner_Validation": name == best_name,
            "Inner_Validation_RMSE_Mean": float(inner_row["Inner_Validation_RMSE_Mean"]),
            "Inner_Validation_RMSE_SD": float(inner_row["Inner_Validation_RMSE_SD"]) if pd.notna(inner_row["Inner_Validation_RMSE_SD"]) else np.nan,
            "Inner_Validation_MAE_Mean": float(inner_row["Inner_Validation_MAE_Mean"]),
            "N_Train_Rows": len(train), "N_Test_Rows": len(test),
            "N_Train_Countries": train["ID"].nunique(), "N_Test_Countries": test["ID"].nunique(),
            "Outer_Test_RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "Outer_Test_MAE": float(mean_absolute_error(y_test, pred)),
            "Outer_Test_Median_AE": float(median_absolute_error(y_test, pred)),
            "Outer_Test_R2": float(r2_score(y_test, pred)),
        })
        joblib.dump(model, OUTPUT_DIR / "Models" / f"{safe_slug(name)}_country_holdout.joblib")
    comparison = pd.DataFrame(metric_rows).sort_values(["Inner_Validation_RMSE_Mean", "Inner_Validation_MAE_Mean"], ascending=True)
    comparison.to_excel(OUTPUT_DIR / "Tables" / "ML_Model_Comparison.xlsx", index=False)
    best_model = models[best_name]
    best_pred = predictions[best_name]
    pred_out = test[["Country", "ID", "Year", Y]].rename(columns={Y: "Actual_Y"}).copy()
    pred_out["Predicted_Y"] = best_pred
    pred_out["Residual"] = pred_out["Actual_Y"] - pred_out["Predicted_Y"]
    pred_out.to_excel(OUTPUT_DIR / "Data" / "Best_Model_HeldOut_Country_Predictions.xlsx", index=False)
    save_text(OUTPUT_DIR / "Logs" / "ML_Model_Selection.txt",
              f"Selected by mean RMSE across five inner country-held-out validation splits: {best_name}.\nOuter test countries were not used for model selection.\nFeatures: {features}\nTrain countries: {train['ID'].nunique()}\nOuter test countries: {test['ID'].nunique()}\n\nOuter test metrics are a single held-out evaluation. Metrics are regression metrics; F1/precision/recall are not primary because Y is continuous.\n")

    # Temporal holdout robustness: train on earlier years, test on the last three years.
    years = sorted(model_data["Year"].unique())
    temporal_rows = []
    if len(years) >= 8:
        cutoff = int(years[-3])  # test the last three distinct years
        tr = model_data[model_data["Year"] < cutoff]
        te = model_data[model_data["Year"] >= cutoff]
        if len(tr) > 100 and len(te) > 30:
            for name, model_cls in [
                ("XGBoost", lambda: XGBRegressor(n_estimators=XGB_ESTIMATORS, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=N_JOBS)),
                ("RandomForest", lambda: RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=8, min_samples_leaf=3, max_features=1.0, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
            ]:
                tm = model_cls()
                tm.fit(tr[features], tr[Y])
                tp = tm.predict(te[features])
                temporal_rows.append({"Model": name, "Train_Through_Year": int(tr["Year"].max()), "Test_From_Year": int(te["Year"].min()), "Test_To_Year": int(te["Year"].max()), "N_Test": len(te), "RMSE": float(np.sqrt(mean_squared_error(te[Y], tp))), "MAE": float(mean_absolute_error(te[Y], tp)), "R2": float(r2_score(te[Y], tp))})
    if temporal_rows:
        pd.DataFrame(temporal_rows).to_excel(OUTPUT_DIR / "Tables" / "ML_Temporal_Holdout_Robustness.xlsx", index=False)

    # Do not score X3_innovation_only on the ML holdout here: Script 1 estimates its
    # entropy weights and min/max from the full panel for construct-validity sensitivity.
    # That index is suitable for the FE sensitivity model, but a strict unseen-country
    # prediction comparison would need the raw indicators, weights, and scaling fitted only
    # within each training fold. The existing core composite indices were supplied by the
    # researcher and their construction provenance is documented in AuditNotes.

    # SHAP: values explain prediction contributions, not causal impacts.
    try:
        import shap
        rng = np.random.default_rng(RANDOM_STATE)
        n_explain = min(SHAP_MAX_ROWS, len(X_test))
        X_explain = X_test.sample(n=n_explain, random_state=RANDOM_STATE) if len(X_test) > n_explain else X_test.copy()
        explainer = shap.TreeExplainer(best_model)
        sv = explainer(X_explain)
        shap_array = np.asarray(sv.values)
        importance = pd.DataFrame({"Variable": features, "Mean_Abs_SHAP": np.abs(shap_array).mean(axis=0)})
        importance = importance.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        importance["Relative_Importance_Percent"] = importance["Mean_Abs_SHAP"] / importance["Mean_Abs_SHAP"].sum() * 100
        importance["Rank"] = np.arange(1, len(importance) + 1)
        importance.to_excel(OUTPUT_DIR / "Tables" / "SHAP_Global_Importance.xlsx", index=False)

        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, X_explain, show=False, plot_type="dot")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Figures" / "SHAP_Beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(8, 5.5))
        shap.summary_plot(sv, X_explain, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Figures" / "SHAP_Global_Bar.png", dpi=300, bbox_inches="tight")
        plt.close()
        # Dependence figures are limited to the four theoretically central constructs;
        # control variables remain visible in global SHAP importance/beeswarm plots.
        for feature in [f for f in CORE_FEATURES if f in features]:
            plt.figure(figsize=(7.5, 5.5))
            shap.dependence_plot(feature, shap_array, X_explain, interaction_index="auto", show=False)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "Figures" / f"SHAP_Dependence_{safe_slug(feature)}.png", dpi=300, bbox_inches="tight")
            plt.close()
    except Exception as exc:
        save_text(OUTPUT_DIR / "Logs" / "SHAP_ERROR.txt", traceback.format_exc())
        log(f"WARNING: SHAP section failed; model metrics are still saved. Reason: {exc}")

    # Interpretable, deliberately shallow regression tree: predictive rules only.
    tree = DecisionTreeRegressor(max_depth=TREE_MAX_DEPTH,
                                 min_samples_leaf=max(TREE_MIN_LEAF_MINIMUM, int(np.ceil(0.02 * len(X_train)))),
                                 random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    joblib.dump(tree, OUTPUT_DIR / "Models" / "Decision_Tree_Regressor.joblib")
    tree_imp = pd.DataFrame({"Variable": features, "Importance": tree.feature_importances_}).sort_values("Importance", ascending=False)
    tree_imp.to_excel(OUTPUT_DIR / "Tables" / "Decision_Tree_Importance.xlsx", index=False)
    save_text(OUTPUT_DIR / "Logs" / "Decision_Tree_Rules.txt", export_text(tree, feature_names=features, decimals=4))
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(tree, feature_names=features, filled=True, rounded=True, proportion=False, impurity=True, ax=ax, fontsize=8)
    ax.set_title("Shallow Decision Tree for Predicting Green Industrial Leapfrogging\n(Predictive partitions, not causal thresholds)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figures" / "Decision_Tree.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # One-at-a-time, in-range model-implied scenarios on held-out-country observations.
    base_pred = best_model.predict(X_test)
    base_mean = float(np.mean(base_pred))
    scenario_rows = [{"Scenario": "Baseline observed test features", "Changed_Variable": "None", "Set_Value": np.nan,
                      "Mean_Predicted_Y": base_mean, "Change_from_Baseline": 0.0, "Median_Change_Individual": 0.0,
                      "P10_Change_Individual": 0.0, "P90_Change_Individual": 0.0,
                      "Interpretation": "Model prediction baseline; not an observed policy outcome."}]
    for feature in features:
        new_value = float(X_train[feature].quantile(0.75))
        changed = X_test.copy()
        changed[feature] = new_value
        altered_pred = best_model.predict(changed)
        diff = altered_pred - base_pred
        scenario_rows.append({"Scenario": f"Set {feature} to training P75", "Changed_Variable": feature, "Set_Value": new_value,
                              "Mean_Predicted_Y": float(np.mean(altered_pred)), "Change_from_Baseline": float(np.mean(diff)),
                              "Median_Change_Individual": float(np.median(diff)), "P10_Change_Individual": float(np.quantile(diff, 0.10)),
                              "P90_Change_Individual": float(np.quantile(diff, 0.90)),
                              "Interpretation": "Model-implied prediction scenario; not a causal policy effect."})
    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_excel(OUTPUT_DIR / "Tables" / "Model_Implied_Scenarios.xlsx", index=False)

    # Save data split provenance to allow exact reproducibility.
    split_prov = model_data[["ID", "Country"]].drop_duplicates().copy()
    split_prov["Split"] = np.where(split_prov["ID"].isin(set(train["ID"])), "Train", "Test")
    split_prov.to_excel(OUTPUT_DIR / "Data" / "Country_Holdout_Split.xlsx", index=False)
    log(f"ML complete. Selected by mean inner-validation RMSE: {best_name}; outer train/test countries={train['ID'].nunique()}/{test['ID'].nunique()}.")


def main() -> None:
    ensure_dirs()
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Analysis-ready data not found: {DATA_FILE}. Run 01_prepare_data.py first.")
    df = pd.read_excel(DATA_FILE, sheet_name=DATA_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Country", "ID", "Year", *CORE_FEATURES, Y]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")

    df["Country"] = df["Country"].astype(str).str.strip()
    df["ID"] = df["ID"].astype(str).str.strip().str.upper()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
    df["Year"] = df["Year"].astype(int)
    numeric_columns = [Y, *CORE_FEATURES, *OPTIONAL_CONTROLS, "X3_innovation_only", "Y_no_overlap"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.duplicated(["ID", "Year"]).any():
        example = df.loc[df.duplicated(["ID", "Year"], keep=False), ["Country", "ID", "Year"]].head(20)
        raise ValueError(f"Duplicate country-year rows detected. Resolve before analysis.\n{example.to_string(index=False)}")
    if df["ID"].nunique() < 20:
        raise ValueError(f"Only {df['ID'].nunique()} countries available; expected a broader panel. Check Script 1 filtering.")

    controls = choose_available_controls(df)
    data_audit = {
        "rows": int(len(df)), "countries": int(df["ID"].nunique()), "years": sorted(int(y) for y in df["Year"].unique()),
        "core_variables": CORE_FEATURES + [Y], "controls_used": controls,
        "controls_missing_from_dataset": [c for c in OPTIONAL_CONTROLS if c not in df.columns],
        "core_missingness": {c: int(df[c].isna().sum()) for c in [Y, *CORE_FEATURES]},
        "data_file": str(DATA_FILE), "policy_registry_file": str(POLICY_REGISTRY_FILE), "causal_policy_DID_estimated": False,
    }
    with open(OUTPUT_DIR / "Logs" / "Analysis_Audit.json", "w", encoding="utf-8") as f:
        json.dump(data_audit, f, indent=2, ensure_ascii=False)
    log("=" * 78)
    log("FINAL ANALYSIS PIPELINE")
    log(json.dumps(data_audit, indent=2, ensure_ascii=False))
    log("=" * 78)

    # 1. Descriptive statistics and missingness.
    describe_vars = [c for c in [Y, *CORE_FEATURES, *controls, "X3_innovation_only", "Y_no_overlap"] if c in df.columns]
    desc = df[describe_vars].describe(percentiles=[.25, .5, .75]).T
    desc["N_missing"] = df[describe_vars].isna().sum()
    desc["Missing_percent"] = df[describe_vars].isna().mean() * 100
    desc.to_excel(OUTPUT_DIR / "Tables" / "Descriptive_Statistics.xlsx")
    df.groupby(["Region"] if "Region" in df.columns else ["Year"])[Y].agg(["count", "mean", "std", "median"]).to_excel(OUTPUT_DIR / "Tables" / "Y_Summary_by_Region.xlsx")
    miss = pd.DataFrame({"Variable": describe_vars, "N": len(df), "Missing": [df[c].isna().sum() for c in describe_vars], "Missing_percent": [df[c].isna().mean() * 100 for c in describe_vars]})
    miss.to_excel(OUTPUT_DIR / "Tables" / "Missingness_Report.xlsx", index=False)

    # 2. Correlations.
    corr_vars = [c for c in [Y, *CORE_FEATURES, *controls] if c in df.columns]
    corr = df[corr_vars].corr(method="pearson", min_periods=20)
    corr.to_excel(OUTPUT_DIR / "Tables" / "Correlation_Matrix_Pearson.xlsx")
    corr_s = df[corr_vars].corr(method="spearman", min_periods=20)
    corr_s.to_excel(OUTPUT_DIR / "Tables" / "Correlation_Matrix_Spearman.xlsx")
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(corr_vars)), max(6, 0.85 * len(corr_vars))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, linewidths=.4, ax=ax)
    ax.set_title("Pearson Correlation Matrix (descriptive, not causal)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figures" / "Correlation_Heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Multicollinearity screen.
    vif = calculate_vif(df, [*CORE_FEATURES, *controls])
    vif.to_excel(OUTPUT_DIR / "Tables" / "VIF_Screening.xlsx", index=False)

    # 4. Two-way fixed-effects panel models.
    log("Running panel fixed-effects models...")
    fe_tables = panel_fe_models(df, controls)
    if not fe_tables.empty:
        fe_tables.to_excel(OUTPUT_DIR / "Tables" / "Panel_FE_Results.xlsx", index=False)
    else:
        save_text(OUTPUT_DIR / "Logs" / "Panel_FE_Status.txt", "No panel FE model estimated. Inspect error logs.")

    # 5. Construct-validity / alternative-index sensitivity.
    sensitivity_rows = []
    if "X3_innovation_only" in df.columns:
        alt_features = ["X1", "X2", "X3_innovation_only", "Institution", *controls]
        alt_sample = df[[Y, "ID", "Year", *alt_features]].dropna().copy()
        try:
            try:
                from linearmodels.panel import PanelOLS
                pdata = alt_sample.set_index(["ID", "Year"]).sort_index()
                formula = "Y ~ 1 + " + " + ".join(alt_features) + " + EntityEffects + TimeEffects"
                alt_res = PanelOLS.from_formula(formula, data=pdata, drop_absorbed=True, check_rank=True).fit(cov_type="clustered", cluster_entity=True, debiased=True, group_debias=True)
                alt_summary = str(alt_res.summary)
            except ImportError:
                formula = "Y ~ 1 + " + " + ".join(alt_features) + " + C(ID) + C(Year)"
                alt_res = smf.ols(formula, data=alt_sample).fit(cov_type="cluster", cov_kwds={"groups": alt_sample["ID"], "use_correction": True})
                alt_summary = concise_statsmodels_report(alt_res, "FE_Sensitivity_X3_Innovation_Only", n_clusters=alt_sample["ID"].nunique())
            save_text(OUTPUT_DIR / "Logs" / "FE_Sensitivity_X3_Innovation_Only_summary.txt", alt_summary)
            sensitivity_rows.append(result_to_table(alt_res, "FE_Sensitivity_X3_Innovation_Only"))
        except Exception as exc:
            save_text(OUTPUT_DIR / "Logs" / "FE_Sensitivity_X3_Innovation_Only_ERROR.txt", traceback.format_exc())
            log(f"WARNING alternative X3 FE sensitivity failed: {exc}")
    if "Y_no_overlap" in df.columns:
        alt_sample = df[["Y_no_overlap", "ID", "Year", *CORE_FEATURES, *controls]].dropna().copy()
        try:
            try:
                from linearmodels.panel import PanelOLS
                pdata = alt_sample.set_index(["ID", "Year"]).sort_index()
                formula = "Y_no_overlap ~ 1 + " + " + ".join(CORE_FEATURES + controls) + " + EntityEffects + TimeEffects"
                alt_res = PanelOLS.from_formula(formula, data=pdata, drop_absorbed=True, check_rank=True).fit(cov_type="clustered", cluster_entity=True, debiased=True, group_debias=True)
                alt_summary = str(alt_res.summary)
            except ImportError:
                formula = "Y_no_overlap ~ 1 + " + " + ".join(CORE_FEATURES + controls) + " + C(ID) + C(Year)"
                alt_res = smf.ols(formula, data=alt_sample).fit(cov_type="cluster", cov_kwds={"groups": alt_sample["ID"], "use_correction": True})
                alt_summary = concise_statsmodels_report(alt_res, "FE_Sensitivity_Y_No_Overlap", n_clusters=alt_sample["ID"].nunique())
            save_text(OUTPUT_DIR / "Logs" / "FE_Sensitivity_Y_No_Overlap_summary.txt", alt_summary)
            sensitivity_rows.append(result_to_table(alt_res, "FE_Sensitivity_Y_No_Overlap"))
        except Exception as exc:
            save_text(OUTPUT_DIR / "Logs" / "FE_Sensitivity_Y_No_Overlap_ERROR.txt", traceback.format_exc())
            log(f"WARNING alternative Y FE sensitivity failed: {exc}")
    if sensitivity_rows:
        pd.concat(sensitivity_rows, ignore_index=True).to_excel(OUTPUT_DIR / "Tables" / "Index_Validity_Sensitivity.xlsx", index=False)

    # 6. Path analysis and institutional moderation (conditional associations, not causal mediation proof).
    log("Running path/mechanism regressions...")
    path_results = run_path_analysis(df, controls)
    if not path_results.empty:
        path_results.to_excel(OUTPUT_DIR / "Tables" / "Path_and_Moderation_Results.xlsx", index=False)

    # 7. Spatial diagnostics and panel SAR/SDM/SEM if valid coordinates and complete sample exist.
    log("Running spatial diagnostics/models when data support them...")
    try:
        run_spatial_diagnostics_and_models(df, controls)
    except Exception as exc:
        save_text(OUTPUT_DIR / "Logs" / "Spatial_FATAL_ERROR.txt", traceback.format_exc())
        log(f"WARNING spatial module stopped safely: {exc}")

    # 8. Audit country-specific policy events; descriptive only, not a causal DID/ATT.
    run_policy_registry_diagnostics(df)

    # 9. ML, SHAP, tree rules, model-implied scenarios.
    log("Training ML models with country-held-out validation...")
    train_and_explain_ml(df, controls)

    # 10. Final reproducibility manifest and interpretation guardrails.
    guardrails = """INTERPRETATION GUARDRAILS
- Fixed-effects coefficients are conditional associations unless a separate identification strategy justifies causality.
- Moran's I diagnoses spatial autocorrelation; it does not identify a causal spillover.
- Spatial panel estimates require scrutiny of W, sample coverage, specification, and estimator diagnostics.
- SDM point impacts, if generated, are point estimates only; no standard errors are reported by that output.
- SHAP values represent model-attributed predictive contribution, not causal effects.
- Decision-tree splits are predictive partitions, not proven policy thresholds.
- Model-implied P75 scenarios are not causal policy impacts.
- ML model selection uses an inner country-held-out validation set; the outer test set is reserved for final evaluation.
- Country-specific AI policy events are inventoried separately from Industry 4.0 and institution-creation events.
- Causal policy DID is not estimated from the incomplete, selective 12-country policy shortlist.
- The policy event-time profile is descriptive only and must not be reported as ATT.
- This script does not estimate GSynth. A shifted control-group mean is not GSynth.
- MARCOS is intentionally omitted until its decision matrix, criterion directions, weights, normalization, and utility formula are independently validated.
"""
    save_text(OUTPUT_DIR / "Logs" / "READ_ME_BEFORE_INTERPRETING.txt", guardrails)
    log("\nANALYSIS PIPELINE FINISHED. Inspect Logs for skipped/failed modules and read READ_ME_BEFORE_INTERPRETING.txt.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
