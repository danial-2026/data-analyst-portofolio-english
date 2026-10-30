# -*- coding: utf-8 -*-
"""Prepare and audit the Asia–Africa country-year panel.

Run this first. Carbon-intensity missing values are handled transparently before Y is built: internal within-country interpolation, then region-year median, then year median. Every imputed value is flagged and audited. Other variables are not imputed.
X1 is interpreted as AI-enabling digital capability (not direct AI capability). X3 is rebuilt
as the MAIN sustainable-innovation index from R&D expenditure and R&D researchers only, avoiding
renewable-energy overlap with Y. Y is rebuilt as an entropy-weighted index from manufacturing value added (positive),
GDP carbon intensity (negative), and renewable electricity generation share (positive).

Dependencies: pandas, numpy, openpyxl
"""
from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# =============================================================================
# USER CONFIGURATION — edit these paths if needed
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = Path(os.environ.get("RESEARCH_INPUT_FILE", str(BASE_DIR / "Master_Asia_Africa_Final_Dataset.xlsx")))
CARBON_FILE = Path(os.environ.get("RESEARCH_CARBON_FILE", str(BASE_DIR / "Carbon_Intensity_GDP_Asia_Africa_2010_2024.xlsx")))
CARBON_SHEET = "Aligned_Data"
CARBON_VALUE_COL = "Carbon_Intensity_GDP_kg_CO2e_per_2021_PPP_USD"
OUTPUT_DIR = Path(os.environ.get("RESEARCH_OUTPUT_DIR", str(BASE_DIR / "Final_Analysis")))
SHEET_PREFERENCE = ["Ready_to_Use_Data", "AnalysisData", "Raw_Data"]
YEAR_MIN = 2010
YEAR_MAX = 2024
KEEP_ONLY_ASIA_AFRICA = True

# Keep these ISO3 sets consistent with the study's stated country sample.
ASIA_CODES = set("AFG ARM AZE BHR BGD BTN BRN KHM CHN CYP GEO IND IDN IRN IRQ JPN JOR KAZ PRK KOR KWT KGZ LAO LBN MYS MDV MNG MMR NPL OMN PAK PHL QAT SAU SGP LKA SYR TJK THA TLS TUR TKM ARE UZB VNM YEM AFG".split())
AFRICA_CODES = set("AGO BEN BWA BFA BDI CPV CMR CAF TCD COM COD COG CIV DJI EGY GNQ ERI ETH GAB GMB GHA GIN GNB KEN LSO LBR LBY MDG MWI MLI MRT MUS MAR MOZ NAM NER NGA RWA STP SEN SYC SLE SOM SSD SDN TZA TGO TUN UGA ZMB ZWE".split())

# Canonical variable aliases.
ALIASES = {
    "Country": ["Country", "Country Name", "country", "Country_Name", "Region/country/area"],
    "ID": ["ID", "Country Code", "CountryCode", "ISO3", "ISO Code", "country_code"],
    "Year": ["Year", "year", "YEAR"],
    "Region": ["Region", "region", "Continent"],
    "Latitude": ["Latitude", "latitude", "Lat"],
    "Longitude": ["Longitude", "longitude", "Lon", "Long"],
    "Y": ["Y", "Green Industrial Leapfrogging", "Green_Industrial_Leapfrogging"],
    "X1": ["X1", "AI Capability", "AI_Capability", "AI Technology Spillover"],
    "X2": ["X2", "Digital Transformation", "Digital_Transformation"],
    "X3": ["X3", "Sustainable Innovation", "Sustainable_Innovation"],
    "Institution": ["Institution", "Institutional Quality", "Institutional_Quality"],
    "ED": ["ED", "GDP per capita (current US$)", "Economic Development"],
    "HC": ["HC", "School enrollment, tertiary (% gross)", "Human Capital"],
    "FDI": ["FDI", "Foreign direct investment, net inflows (% of GDP)"],
    "FDI_ihs": ["FDI_ihs"],
    "ASI": ["ASI", "Advanced Industrial Structure"],
    "GI": ["GI", "General government final consumption expenditure (% of GDP)", "Government Intervention"],
    "FD": ["FD", "Domestic credit to private sector (% of GDP)", "Financial Development"],
}

CORE_VARS = ["Y", "X1", "X2", "X3", "Institution"]
CONTROL_VARS = ["ED_log", "HC", "FDI_ihs", "ASI", "GI", "FD"]
COORD_VARS = ["Latitude", "Longitude"]

# Exact source-indicator candidates for optional, conceptually narrower robustness indices.
RD_CANDIDATES = ["Research and development expenditure (% of GDP)", "R&D expenditure (% of GDP)"]
RESEARCHER_CANDIDATES = ["Researchers in R&D (per million people)", "Researchers in R&D (per million people) (WDI)"]
MANUFACTURING_CANDIDATES = ["Manufacturing, value added (% of GDP)"]
RENEWABLE_GENERATION_CANDIDATES = ["RE_Gen_Share", "RE share of electricity generation (%)"]
CARBON_INTENSITY_CANDIDATES = [CARBON_VALUE_COL, "Carbon intensity of GDP", "CO2 intensity (kg CO2 per PPP $ GDP)"]

MISSING_TOKENS = {"", "..", "-", "—", "na", "n/a", "nan", "null", "none"}


# Policy events supplied by the researcher and checked against government/official
# sources on 15 October 2026. This is an initial inventory, NOT a complete treatment
# roster for all countries in the panel. Only the AI-strategy events listed as
# candidate primary treatment are eligible for further review; the policy year must
# be verified consistently (announcement vs approval vs effective date) before DID.
POLICY_EVENTS = [
    {"ID":"CHN", "Country":"China", "Policy_Year":2017, "Policy_Name":"New Generation Artificial Intelligence Development Plan", "Policy_Type":"AI_strategy", "Primary_AI_Treatment_Candidate":1, "Evidence_Status":"Official national plan; verify coding convention for adoption date", "Source_URL":"https://app.www.gov.cn/govdata/gov/201707/20/408540/article.html", "Researcher_Note":"State Council plan; published July 2017."},
    {"ID":"IND", "Country":"India", "Policy_Year":2018, "Policy_Name":"National Strategy for Artificial Intelligence (#AIforAll)", "Policy_Type":"AI_strategy", "Primary_AI_Treatment_Candidate":1, "Evidence_Status":"Official strategy document; date shown as June 2018", "Source_URL":"https://www.niti.gov.in/sites/default/files/2023-03/National-Strategy-for-Artificial-Intelligence.pdf", "Researcher_Note":"NITI Aayog strategy."},
    {"ID":"SGP", "Country":"Singapore", "Policy_Year":2019, "Policy_Name":"National AI Strategy", "Policy_Type":"AI_strategy", "Primary_AI_Treatment_Candidate":1, "Evidence_Status":"Official government strategy page", "Source_URL":"https://www.smartnation.gov.sg/initiatives/national-ai-strategy/", "Researcher_Note":"Launch year 2019; distinguish from NAIS 2.0 (2023)."},
    {"ID":"KOR", "Country":"Korea, Rep.", "Policy_Year":2019, "Policy_Name":"National Strategy for Artificial Intelligence", "Policy_Type":"AI_strategy", "Primary_AI_Treatment_Candidate":1, "Evidence_Status":"Official government press release; 17 December 2019", "Source_URL":"https://m.korea.kr/briefing/pressReleaseView.do?newsId=156366736", "Researcher_Note":"Official policy briefing dated 17 December 2019."},
    {"ID":"JPN", "Country":"Japan", "Policy_Year":2019, "Policy_Name":"AI Strategy 2019", "Policy_Type":"AI_strategy", "Primary_AI_Treatment_Candidate":1, "Evidence_Status":"Official Cabinet Office document dated 11 June 2019", "Source_URL":"https://www8.cao.go.jp/cstp/ai/aistratagy2019en.pdf", "Researcher_Note":"Integrated Innovation Strategy Promotion Council decision."},
    {"ID":"IDN", "Country":"Indonesia", "Policy_Year":2020, "Policy_Name":"Strategi Nasional Kecerdasan Artifisial Republik Indonesia 2020–2045", "Policy_Type":"AI_strategy", "Primary_AI_Treatment_Candidate":1, "Evidence_Status":"Strategy document dated July 2020; formal adoption convention should be recorded", "Source_URL":"https://karya.brin.go.id/id/eprint/13918/1/Artikel_Michael_PR%20Elektronika_2020.pdf", "Researcher_Note":"Separate from Making Indonesia 4.0 (2018), which is a broader Industry 4.0 roadmap."},
    {"ID":"SAU", "Country":"Saudi Arabia", "Policy_Year":2020, "Policy_Name":"National Strategy for Data & AI (NSDAI)", "Policy_Type":"AI_strategy", "Primary_AI_Treatment_Candidate":1, "Evidence_Status":"Official SDAIA page says strategy approved 17 July 2020", "Source_URL":"https://sdaia.gov.sa/en/SDAIA/SdaiaStrategies/Pages/NationalStrategyForDataAndAI.aspx", "Researcher_Note":"SDAIA establishment is a separate institutional event in 2019."},
    {"ID":"EGY", "Country":"Egypt, Arab Rep.", "Policy_Year":2020, "Policy_Name":"Egypt National AI Strategy (first edition)", "Policy_Type":"AI_strategy", "Primary_AI_Treatment_Candidate":1, "Evidence_Status":"Official 2025 second-edition strategy says Egypt released the first edition in 2020; verify the exact formal approval/effective date before causal coding", "Source_URL":"https://ai.gov.eg/SynchedFiles/en/Resources/AIstrategy%20English%2016-1-2025-1.pdf", "Researcher_Note":"A different official source has sometimes been described as a 2021 launch; record and resolve the date convention. National AI Council was approved in 2019 and must remain a separate institutional event."},
    {"ID":"RWA", "Country":"Rwanda", "Policy_Year":2023, "Policy_Name":"National Artificial Intelligence Policy", "Policy_Type":"AI_strategy", "Primary_AI_Treatment_Candidate":1, "Evidence_Status":"Official government policy page updated in April 2023", "Source_URL":"https://www.minict.gov.rw/ai-policy", "Researcher_Note":"Policy year candidate 2023; use formal adoption/effective date if a more precise date is available."},
    {"ID":"IDN", "Country":"Indonesia", "Policy_Year":2018, "Policy_Name":"Making Indonesia 4.0", "Policy_Type":"digital_industry_policy", "Primary_AI_Treatment_Candidate":0, "Evidence_Status":"Official government announcement dated 4 April 2018", "Source_URL":"https://www.setneg.go.id/baca/index/presiden_jadikan_making_indonesia_40_sebagai_agenda_nasional", "Researcher_Note":"Broad Industry 4.0/digital-industry policy; do not merge with AI-strategy treatment."},
    {"ID":"MYS", "Country":"Malaysia", "Policy_Year":2018, "Policy_Name":"Industry4WRD: National Policy on Industry 4.0", "Policy_Type":"digital_industry_policy", "Primary_AI_Treatment_Candidate":0, "Evidence_Status":"Official MITI page gives launch date 31 October 2018", "Source_URL":"https://www.miti.gov.my/index.php/pages/view/industry4WRD", "Researcher_Note":"Broad Industry 4.0 policy; separate treatment from national AI strategy."},
    {"ID":"SAU", "Country":"Saudi Arabia", "Policy_Year":2019, "Policy_Name":"Saudi Data and Artificial Intelligence Authority (SDAIA) established", "Policy_Type":"AI_governance_institution", "Primary_AI_Treatment_Candidate":0, "Evidence_Status":"Official SDAIA page confirms establishment by Royal Order", "Source_URL":"https://sdaia.gov.sa/en/SDAIA/about/Pages/About.aspx", "Researcher_Note":"Institutional-capacity event, not the 2020 national strategy."},
    {"ID":"EGY", "Country":"Egypt, Arab Rep.", "Policy_Year":2019, "Policy_Name":"National Council for Artificial Intelligence approved", "Policy_Type":"AI_governance_institution", "Primary_AI_Treatment_Candidate":0, "Evidence_Status":"Official government news dated 22 November 2019", "Source_URL":"https://sis.gov.eg/en/media-center/news/cabinet-approves-establishing-national-artificial-intelligence-council/", "Researcher_Note":"Institution formation precedes the national AI strategy; code separately."},
    {"ID":"KEN", "Country":"Kenya", "Policy_Year":2019, "Policy_Name":"Distributed Ledgers and Artificial Intelligence Taskforce/report", "Policy_Type":"AI_taskforce_report", "Primary_AI_Treatment_Candidate":0, "Evidence_Status":"Official later government strategy document describes this as a previous taskforce/report", "Source_URL":"https://ict.go.ke/sites/default/files/2025-06/National-AI-Strategy_IMPLEMENTATION.pdf", "Researcher_Note":"Not coded as national AI strategy adoption; avoid treating a taskforce report as an implemented policy."},
    {"ID":"MAR", "Country":"Morocco", "Policy_Year":2024, "Policy_Name":"Digital Morocco 2030 strategy", "Policy_Type":"broad_digital_strategy", "Primary_AI_Treatment_Candidate":0, "Evidence_Status":"Official digital-transition ministry page dated 25 September 2024", "Source_URL":"https://mmsp.gov.ma/fr/actualites/cliquez-ici-pour-d%C3%A9couvrir-la-strat%C3%A9gie-digital-morocco-2030", "Researcher_Note":"Broad digital strategy, not equivalent to an AI-specific national strategy."},
]



def log(message: str) -> None:
    print(message, flush=True)


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    colmap = {str(c).strip().casefold(): c for c in columns}
    for candidate in candidates:
        if candidate.casefold() in colmap:
            return colmap[candidate.casefold()]
    return None


def rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for canonical, candidates in ALIASES.items():
        col = pick_column(df.columns, candidates)
        if col is not None and col != canonical and canonical not in df.columns:
            rename[col] = canonical
    return df.rename(columns=rename)


def classify_region(code: object, existing: object = None) -> object:
    if isinstance(existing, str) and existing.strip().casefold() in {"asia", "africa"}:
        return existing.strip().title()
    code = str(code).strip().upper() if pd.notna(code) else ""
    if code in ASIA_CODES:
        return "Asia"
    if code in AFRICA_CODES:
        return "Africa"
    return existing if pd.notna(existing) else np.nan


def locate_input_sheet(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input workbook tidak ditemukan: {path}\n"
            "Letakkan workbook sumber di folder yang sama dengan skrip atau atur environment variable RESEARCH_INPUT_FILE."
        )
    xls = pd.ExcelFile(path)
    for sheet in SHEET_PREFERENCE:
        if sheet in xls.sheet_names:
            return pd.read_excel(path, sheet_name=sheet), sheet
    # One-sheet workbook created by an older enrichment script.
    return pd.read_excel(path, sheet_name=xls.sheet_names[0]), xls.sheet_names[0]


def numeric_clean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.mask(cleaned.str.casefold().isin(MISSING_TOKENS))
    # Do not guess locale-specific separators: comma decimals must be resolved at source.
    return pd.to_numeric(cleaned, errors="coerce")


def find_indicator(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    return pick_column(df.columns, candidates)


def entropy_index(df: pd.DataFrame, columns: list[str], directions: list[int], out_name: str) -> tuple[pd.Series, pd.DataFrame]:
    """Entropy-weighted min-max index using one common complete-case sample.

    No imputation is used. Min/max and entropy weights are estimated from rows with
    every specified indicator observed, so indicator-specific missingness cannot create
    weights based on different row populations. Scores for incomplete rows remain NaN.
    """
    if len(columns) != len(directions) or not columns:
        raise ValueError(f"Invalid indicator specification for {out_name}.")
    if any(direction not in (-1, 1) for direction in directions):
        raise ValueError(f"Directions for {out_name} must contain only +1 or -1.")
    raw = pd.DataFrame({col: numeric_clean(df[col]) for col in columns}, index=df.index)
    complete = raw.notna().all(axis=1)
    n_complete = int(complete.sum())
    if n_complete < 2:
        raise ValueError(f"{out_name}: only {n_complete} rows have all indicators. Need at least 2 complete rows; index not created.")

    z = pd.DataFrame(index=df.index, columns=columns, dtype=float)
    weight_rows = []
    entropies = []
    for col, direction in zip(columns, directions):
        x_fit = raw.loc[complete, col].astype(float)
        lo, hi = float(x_fit.min()), float(x_fit.max())
        if np.isclose(hi, lo):
            scaled = pd.Series(np.nan, index=df.index, dtype=float)
            scaled.loc[raw[col].notna()] = 0.5
            entropy = 1.0  # constant criterion carries no relative information
        else:
            scaled = ((raw[col] - lo) / (hi - lo)).clip(0.0, 1.0)
            if direction == -1:
                # Reverse a negative-oriented indicator BEFORE calculating entropy weights.
                # Higher raw values must contribute lower normalized scores.
                scaled = 1.0 - scaled
            elif direction != 1:
                raise ValueError(f"Direction for {col!r} must be +1 or -1, got {direction!r}.")
            x_complete = scaled.loc[complete].astype(float)
            col_sum = float(x_complete.sum())
            if col_sum <= 0:
                entropy = 1.0
            else:
                p = (x_complete / col_sum).clip(lower=0)
                p = p[p > 0]
                entropy = float(-(p * np.log(p)).sum() / np.log(n_complete)) if n_complete > 1 else 1.0
                entropy = float(np.clip(entropy, 0.0, 1.0))
        z[col] = scaled
        entropies.append(entropy)
        weight_rows.append({
            "Index": out_name, "Indicator": col,
            "Direction": "positive" if direction == 1 else "negative",
            "N_Complete_For_Weights": n_complete, "Min_Complete_Sample": lo,
            "Max_Complete_Sample": hi, "Entropy": entropy,
        })
    divergences = 1.0 - np.asarray(entropies, dtype=float)
    if divergences.sum() <= 1e-12:
        weights = np.repeat(1.0 / len(columns), len(columns))
    else:
        weights = divergences / divergences.sum()
    score = pd.Series(np.nan, index=df.index, name=out_name, dtype=float)
    score.loc[complete] = z.loc[complete, columns].to_numpy(dtype=float) @ weights
    for i, row in enumerate(weight_rows):
        row["Weight"] = float(weights[i])
    return score, pd.DataFrame(weight_rows)


def build_policy_registry(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reviewable policy-event inventory and a complete country roster.

    Existing reviewed country-registry entries are preserved on re-runs. The 12-country
    seed list is NOT assumed to be exhaustive: countries outside it remain unknown, not
    untreated. No causal treatment variable is merged into the analysis data by default.
    """
    events = pd.DataFrame(POLICY_EVENTS)
    events.insert(0, "Event_ID", [f"EV{i:02d}" for i in range(1, len(events) + 1)])

    country_roster = df[["ID", "Country"] + (["Region"] if "Region" in df.columns else [])].drop_duplicates().sort_values("ID")
    if country_roster["ID"].duplicated().any():
        raise ValueError("Cannot create policy registry: one ID maps to multiple country rows.")
    country_roster["AI_Treatment_Year"] = pd.Series(pd.NA, index=country_roster.index, dtype="Int64")
    country_roster["AI_Treatment_Policy"] = ""
    country_roster["AI_Treatment_Source_URL"] = ""
    country_roster["AI_Treatment_Status"] = "unknown_not_researched"
    country_roster["AI_Treatment_Reviewed"] = 0
    country_roster["AI_Treatment_Verification_Note"] = "No AI strategy event seeded from initial shortlist; this does NOT mean no policy existed. Research country policy history through 2024."
    country_roster["Digital_Industry_Policy_Year"] = pd.Series(pd.NA, index=country_roster.index, dtype="Int64")
    country_roster["Digital_Industry_Policy_Name"] = ""
    country_roster["Digital_Industry_Source_URL"] = ""
    country_roster["Other_AI_Institutional_Event"] = ""
    country_roster["Other_AI_Institutional_Year"] = pd.Series(pd.NA, index=country_roster.index, dtype="Int64")
    country_roster["Other_AI_Event_Source_URL"] = ""

    primary = events[events["Primary_AI_Treatment_Candidate"].eq(1)]
    for _, ev in primary.iterrows():
        mask = country_roster["ID"].eq(ev["ID"])
        if mask.any():
            country_roster.loc[mask, "AI_Treatment_Year"] = int(ev["Policy_Year"])
            country_roster.loc[mask, "AI_Treatment_Policy"] = ev["Policy_Name"]
            country_roster.loc[mask, "AI_Treatment_Source_URL"] = ev["Source_URL"]
            country_roster.loc[mask, "AI_Treatment_Status"] = "candidate_treated_review"
            country_roster.loc[mask, "AI_Treatment_Verification_Note"] = ev["Evidence_Status"] + ". Review whether the recorded date is announcement, approval, or effective date."
    digital = events[events["Policy_Type"].eq("digital_industry_policy") | events["Policy_Type"].eq("broad_digital_strategy")]
    for _, ev in digital.iterrows():
        mask = country_roster["ID"].eq(ev["ID"])
        if mask.any():
            country_roster.loc[mask, "Digital_Industry_Policy_Year"] = int(ev["Policy_Year"])
            country_roster.loc[mask, "Digital_Industry_Policy_Name"] = ev["Policy_Name"]
            country_roster.loc[mask, "Digital_Industry_Source_URL"] = ev["Source_URL"]
    institutional = events[events["Policy_Type"].eq("AI_governance_institution")]
    for _, ev in institutional.iterrows():
        mask = country_roster["ID"].eq(ev["ID"])
        if mask.any():
            country_roster.loc[mask, "Other_AI_Institutional_Event"] = ev["Policy_Name"]
            country_roster.loc[mask, "Other_AI_Institutional_Year"] = int(ev["Policy_Year"])
            country_roster.loc[mask, "Other_AI_Event_Source_URL"] = ev["Source_URL"]

    registry_path = output_dir / "Policy_Registry_To_Complete.xlsx"
    prior_sheets = {}
    if registry_path.exists():
        try:
            with pd.ExcelFile(registry_path) as xls:
                prior_sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}
            prior = prior_sheets.get("CountryRegistry", pd.DataFrame()).copy()
            if not prior.empty and "ID" in prior.columns:
                prior["ID"] = prior["ID"].astype(str).str.strip().str.upper()
                prior = prior.drop_duplicates("ID", keep="last").set_index("ID")
                current = country_roster.set_index("ID")

                # Preserve all researcher-added columns for the same country. This covers
                # date conventions, source notes, review fields, and other custom audit columns.
                for col in prior.columns:
                    if col not in current.columns:
                        current[col] = prior[col].reindex(current.index)

                # Preserve treatment/policy fields only for reviewed rows or rows explicitly
                # verified as treated/never-treated. Unreviewed candidates are refreshed from
                # the current seed inventory, including changes to candidate dates.
                shared_ids = current.index.intersection(prior.index)
                if "AI_Treatment_Reviewed" in prior.columns:
                    prior_reviewed = pd.to_numeric(prior["AI_Treatment_Reviewed"], errors="coerce").fillna(0).eq(1)
                else:
                    prior_reviewed = pd.Series(False, index=prior.index)
                if "AI_Treatment_Status" in prior.columns:
                    prior_verified = prior["AI_Treatment_Status"].astype(str).isin({"treated_verified", "never_treated_verified"})
                else:
                    prior_verified = pd.Series(False, index=prior.index)
                locked_ids = shared_ids.intersection(prior.index[prior_reviewed | prior_verified])
                treatment_cols = [c for c in [
                    "AI_Treatment_Year", "AI_Treatment_Policy", "AI_Treatment_Source_URL",
                    "AI_Treatment_Status", "AI_Treatment_Reviewed", "AI_Treatment_Verification_Note",
                    "Digital_Industry_Policy_Year", "Digital_Industry_Policy_Name", "Digital_Industry_Source_URL",
                    "Other_AI_Institutional_Event", "Other_AI_Institutional_Year", "Other_AI_Event_Source_URL",
                ] if c in prior.columns and c in current.columns]
                for col in treatment_cols:
                    if len(locked_ids):
                        current.loc[locked_ids, col] = prior.loc[locked_ids, col]
                country_roster = current.reset_index()
        except Exception as exc:
            raise RuntimeError(f"Could not preserve existing policy registry {registry_path}: {exc}") from exc

    instructions = pd.DataFrame([{
        "Instructions": "Review every country in CountryRegistry. For AI_Treatment_Status use candidate_treated_review, treated_verified, never_treated_verified, or unknown_not_researched.",
        "Rule_1": "Do not mark a country never_treated_verified without documenting a systematic search of official sources through 2024.",
        "Rule_2": "AI_Treatment_Year must consistently use the official approval/adoption year, not a mixture of proposal, taskforce, institution-creation, or strategy-launch dates.",
        "Rule_3": "Digital/Industry 4.0 policies and AI-governance institutions are separate exposures, not interchangeable AI strategy treatments.",
        "Rule_4": "This shortlist is not exhaustive; policy analysis is descriptive-only until the whole country roster is reviewed.",
    }])
    if registry_path.exists():
        # Replace CountryRegistry with the refreshed roster while retaining all custom
        # columns already read above and preserving the existing Instructions sheet.
        with pd.ExcelWriter(registry_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            country_roster.to_excel(writer, sheet_name="CountryRegistry", index=False)
            events.to_excel(writer, sheet_name="EventInventory", index=False)
            if "Instructions" not in prior_sheets:
                instructions.to_excel(writer, sheet_name="Instructions", index=False)
    else:
        with pd.ExcelWriter(registry_path, engine="openpyxl") as writer:
            country_roster.to_excel(writer, sheet_name="CountryRegistry", index=False)
            events.to_excel(writer, sheet_name="EventInventory", index=False)
            instructions.to_excel(writer, sheet_name="Instructions", index=False)
    return events, country_roster


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_df, source_sheet = locate_input_sheet(INPUT_FILE)
    log(f"[1/7] Loaded {INPUT_FILE.name!r}, sheet={source_sheet!r}, shape={input_df.shape}")

    df = rename_aliases(input_df)
    required_keys = ["Country", "Year"]
    missing_keys = [c for c in required_keys if c not in df.columns]
    if missing_keys:
        raise ValueError(f"Kolom kunci belum ditemukan: {missing_keys}. Kolom sumber: {list(df.columns)}")
    if "ID" not in df.columns:
        log("WARNING: ID/kode negara tidak ditemukan; Country akan dipakai sebagai identifier. Spatial analysis kemungkinan tidak tersedia.")
        df["ID"] = df["Country"].astype(str).str.strip()

    df["Country"] = df["Country"].astype(str).str.strip()
    df["ID"] = df["ID"].astype(str).str.strip().str.upper()
    df["Year"] = numeric_clean(df["Year"])
    df = df[df["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
    df["Year"] = df["Year"].astype(int)
    df = df[df["Country"].ne("") & df["ID"].ne("")].copy()

    # If the master workbook has a Ready_to_Use_Data sheet plus Raw_Data, merge only
    # the source indicator columns that are absent from Ready_to_Use_Data. This preserves
    # the supplied composites while allowing the sensitivity indices to be rebuilt from
    # additional indicator/source columns. The sheet named Raw_Data may already have been
    # imputed upstream; merging it cannot restore original missing values. Never merge duplicated keys or overwrite supplied variables.
    try:
        workbook_sheets = pd.ExcelFile(INPUT_FILE).sheet_names
    except Exception:
        workbook_sheets = []
    if source_sheet != "Raw_Data" and "Raw_Data" in workbook_sheets:
        raw = rename_aliases(pd.read_excel(INPUT_FILE, sheet_name="Raw_Data"))
        if {"ID", "Year"}.issubset(raw.columns):
            raw["ID"] = raw["ID"].astype(str).str.strip().str.upper()
            raw["Year"] = numeric_clean(raw["Year"])
            raw = raw[raw["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
            raw["Year"] = raw["Year"].astype(int)
            raw = raw[raw["ID"].ne("")].copy()
            raw_dups = raw.duplicated(["ID", "Year"], keep=False)
            if raw_dups.any():
                examples = raw.loc[raw_dups, ["ID", "Year"]].head(10).to_dict("records")
                raise ValueError(f"Raw_Data sheet has duplicate ID-Year keys; cannot safely merge indicators: {examples}")
            raw_extras = [c for c in raw.columns if c not in set(df.columns) | {"ID", "Year", "Country"}]
            if raw_extras:
                raw_join = raw[["ID", "Year", *raw_extras]]
                merged = df.merge(raw_join, on=["ID", "Year"], how="left", validate="one_to_one", indicator="_raw_merge_status")
                match_rate = float(merged["_raw_merge_status"].eq("both").mean() * 100)
                df = merged.drop(columns="_raw_merge_status")
                log(f"[2b/7] Added {len(raw_extras)} additional indicator/source columns from Raw_Data; key match rate={match_rate:.2f}%.")
                if match_rate < 95.0:
                    log("WARNING: fewer than 95% of panel rows matched Raw_Data; inspect source-sheet coverage before using sensitivity indices.")
        else:
            log("WARNING: Raw_Data sheet exists but lacks ID/Year; additional source columns were not merged.")

    if "Region" not in df.columns:
        df["Region"] = np.nan
    df["Region"] = [classify_region(code, reg) for code, reg in zip(df["ID"], df["Region"])]
    if KEEP_ONLY_ASIA_AFRICA:
        before = len(df)
        df = df[df["Region"].isin(["Asia", "Africa"])].copy()
        log(f"[2/7] Asia/Africa filter: {before} -> {len(df)} rows; countries={df['ID'].nunique()}")
        if df.empty:
            raise ValueError("Tidak ada observasi Asia/Africa selepas filter. Periksa ISO3/Region mapping.")

    # Convert modeled variables and coordinates explicitly; retain all other source fields.
    model_numeric = set(CORE_VARS + CONTROL_VARS + ["ED", "FDI", "Latitude", "Longitude"])
    for col in model_numeric:
        if col in df.columns:
            df[col] = numeric_clean(df[col])
    # FDI IHS is defined correctly for positive and negative values: asinh(x), no clipping.
    if "FDI" in df.columns:
        # Recompute from signed raw FDI, correcting older versions that clipped negative values.
        df["FDI_ihs"] = np.arcsinh(df["FDI"])
    elif "FDI_ihs" in df.columns:
        df["FDI_ihs"] = numeric_clean(df["FDI_ihs"])
        log("WARNING: only precomputed FDI_ihs is available; its construction cannot be audited without raw FDI.")
    if "ED" in df.columns:
        # GDP per capita is strongly right-skewed. Log only positive observed values; do not replace nonpositive values.
        ed = numeric_clean(df["ED"])
        ed_log = pd.Series(np.nan, index=df.index, dtype=float)
        positive_ed = ed.gt(0)
        ed_log.loc[positive_ed] = np.log(ed.loc[positive_ed])
        df["ED_log"] = ed_log

    missing_core_cols = [c for c in CORE_VARS if c not in df.columns]
    if missing_core_cols:
        raise ValueError(f"Variabel inti tidak tersedia: {missing_core_cols}. Jangan dihitung dengan proxy tanpa keputusan konseptual.")

    # Hard stop for duplicated panel keys; do not average or silently deduplicate.
    dup = df.duplicated(["ID", "Year"], keep=False)
    if dup.any():
        example = df.loc[dup, ["Country", "ID", "Year"]].head(20).to_dict("records")
        raise ValueError(f"Ada duplicate country-year keys ({int(dup.sum())} rows). Perbaiki merge sebelum analisis. Contoh: {example}")

    # Check ID-to-name and coordinate consistency.
    country_names_per_id = df.groupby("ID")["Country"].nunique()
    inconsistent_names = country_names_per_id[country_names_per_id > 1]
    if not inconsistent_names.empty:
        raise ValueError(f"Satu ID memiliki lebih dari satu nama negara: {inconsistent_names.index.tolist()}")
    for coord in ["Latitude", "Longitude"]:
        if coord in df.columns:
            spread = df.groupby("ID")[coord].nunique(dropna=True)
            bad = spread[spread > 1]
            if not bad.empty:
                raise ValueError(f"Koordinat {coord} tidak konsisten antar-tahun untuk ID: {bad.index.tolist()[:20]}")

    # Primary X3: sustainable innovation measured only by R&D spending and researchers.
    # This replaces the prior X3 that overlapped with Y through renewable-energy consumption.
    # Stop rather than silently fall back to the conceptually overlapping supplied X3.
    weights_tables = []
    rd_col = find_indicator(df, RD_CANDIDATES)
    researcher_col = find_indicator(df, RESEARCHER_CANDIDATES)
    if not (rd_col and researcher_col):
        raise ValueError(
            "Cannot construct primary X3: both R&D expenditure and R&D researchers indicators are required. "
            "Check Raw_Data/source columns. The old X3 is not silently retained because it includes "
            "renewable-energy consumption, which overlaps conceptually with Y."
        )
    df["X3"], wtab = entropy_index(df, [rd_col, researcher_col], [1, 1], "X3")
    weights_tables.append(wtab)
    log(f"[3/7] Rebuilt PRIMARY X3 sustainable-innovation index from: {rd_col}; {researcher_col}")

    # Merge the supplied carbon-intensity workbook by ISO3 ID + Year, then REBUILD Y.
    # Direction: manufacturing value added (+), carbon intensity (-), renewable electricity share (+).
    if not CARBON_FILE.exists():
        raise FileNotFoundError(
            f"Carbon-intensity workbook not found: {CARBON_FILE}. Put it beside this script or set RESEARCH_CARBON_FILE."
        )
    carbon_book = pd.ExcelFile(CARBON_FILE)
    if CARBON_SHEET not in carbon_book.sheet_names:
        raise ValueError(f"Sheet {CARBON_SHEET!r} not found in {CARBON_FILE.name}; sheets={carbon_book.sheet_names}")
    carbon = pd.read_excel(CARBON_FILE, sheet_name=CARBON_SHEET)
    needed_carbon_cols = {"ID", "Year", CARBON_VALUE_COL}
    if not needed_carbon_cols.issubset(carbon.columns):
        raise ValueError(f"Carbon workbook missing required columns: {sorted(needed_carbon_cols - set(carbon.columns))}")
    carbon = carbon[["ID", "Year", CARBON_VALUE_COL]].copy()
    carbon["ID"] = carbon["ID"].astype(str).str.strip().str.upper()
    carbon["Year"] = numeric_clean(carbon["Year"])
    carbon[CARBON_VALUE_COL] = numeric_clean(carbon[CARBON_VALUE_COL])
    carbon = carbon[carbon["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
    carbon["Year"] = carbon["Year"].astype(int)
    if carbon.duplicated(["ID", "Year"]).any():
        raise ValueError("Carbon workbook has duplicate ID-Year keys; cannot safely merge.")
    if CARBON_VALUE_COL in df.columns:
        df = df.drop(columns=[CARBON_VALUE_COL])
    df = df.merge(carbon, on=["ID", "Year"], how="left", validate="one_to_one")
    carbon_col = CARBON_VALUE_COL
    carbon_raw_col = "Carbon_Intensity_GDP_Observed"
    df[carbon_raw_col] = numeric_clean(df[carbon_col])
    carbon_match = int(df[carbon_raw_col].notna().sum())
    df["Carbon_Intensity_Imputed"] = df[carbon_raw_col].isna()
    df["Carbon_Intensity_Imputation_Method"] = np.where(
        df[carbon_raw_col].notna(), "Observed", "Not yet imputed"
    )

    # Transparent, carbon-only imputation hierarchy:
    # (1) interpolate internal gaps within country, without extrapolating endpoints;
    # (2) fill remaining gaps using same-region, same-year median;
    # (3) where that is unavailable, use same-year median across the study sample.
    # No global median, forward-fill, or backward-fill is used.
    df["Carbon_Intensity_GDP"] = df[carbon_raw_col].astype(float)
    for _, idx in df.groupby("ID", sort=False).groups.items():
        idx = list(idx)
        ordered_idx = df.loc[idx].sort_values("Year").index
        vals = df.loc[ordered_idx, "Carbon_Intensity_GDP"]
        interp = vals.interpolate(method="linear", limit_area="inside")
        fill_idx = vals.index[vals.isna() & interp.notna()]
        df.loc[fill_idx, "Carbon_Intensity_GDP"] = interp.loc[fill_idx]
        df.loc[fill_idx, "Carbon_Intensity_Imputation_Method"] = "Within-country linear interpolation (internal gap)"

    remain = df["Carbon_Intensity_GDP"].isna()
    region_year_median = df.groupby(["Region", "Year"])["Carbon_Intensity_GDP"].transform("median")
    fill_idx = df.index[remain & region_year_median.notna()]
    df.loc[fill_idx, "Carbon_Intensity_GDP"] = region_year_median.loc[fill_idx]
    df.loc[fill_idx, "Carbon_Intensity_Imputation_Method"] = "Region-year median"

    remain = df["Carbon_Intensity_GDP"].isna()
    year_median = df.groupby("Year")["Carbon_Intensity_GDP"].transform("median")
    fill_idx = df.index[remain & year_median.notna()]
    df.loc[fill_idx, "Carbon_Intensity_GDP"] = year_median.loc[fill_idx]
    df.loc[fill_idx, "Carbon_Intensity_Imputation_Method"] = "Study-sample year median (fallback)"

    carbon_unresolved = int(df["Carbon_Intensity_GDP"].isna().sum())
    carbon_imputed = int(df["Carbon_Intensity_Imputed"].sum())
    carbon_method_counts = df.loc[df["Carbon_Intensity_Imputed"], "Carbon_Intensity_Imputation_Method"].value_counts().to_dict()
    log(f"[4a/7] Merged GDP carbon intensity: originally observed={carbon_match}/{len(df)} ({carbon_match/len(df)*100:.2f}%).")
    log(f"[4a/7] Carbon missing values={carbon_imputed}; imputation methods={carbon_method_counts}; unresolved after hierarchy={carbon_unresolved}.")

    mfg_col = find_indicator(df, MANUFACTURING_CANDIDATES)
    ren_col = find_indicator(df, RENEWABLE_GENERATION_CANDIDATES)
    if not (mfg_col and ren_col):
        raise ValueError(
            "Cannot rebuild Y: manufacturing value added and renewable electricity generation share are required. "
            f"Detected manufacturing={mfg_col!r}, renewable_generation={ren_col!r}."
        )
    # Use the imputed, audited series (not the raw source column) in Y.
    y_cols = [mfg_col, "Carbon_Intensity_GDP", ren_col]
    df["Y_original_supplied"] = numeric_clean(df["Y"])
    df["Y"], y_weights = entropy_index(df, y_cols, [1, -1, 1], "Y")
    weights_tables.append(y_weights)
    n_y = int(df["Y"].notna().sum())
    log(f"[4b/7] Rebuilt Y entropy index from: {mfg_col} (+); Carbon_Intensity_GDP (-); {ren_col} (+). Valid Y={n_y}/{len(df)}; rows with any missing Y indicator remain missing.")

    # Construct metadata/audit tables.
    df = df.sort_values(["ID", "Year"]).reset_index(drop=True)
    numeric_candidates = [c for c in CORE_VARS + CONTROL_VARS if c in df.columns]
    missingness = pd.DataFrame({
        "Variable": numeric_candidates,
        "N_total": len(df),
        "N_missing": [int(df[c].isna().sum()) for c in numeric_candidates],
        "Missing_percent": [float(df[c].isna().mean() * 100) for c in numeric_candidates],
        "N_observed": [int(df[c].notna().sum()) for c in numeric_candidates],
    })
    coverage = df.groupby(["ID", "Country", "Region"], dropna=False).agg(
        Years_observed=("Year", "nunique"), First_year=("Year", "min"), Last_year=("Year", "max"),
        Rows=("Year", "size"),
    ).reset_index()
    all_years = set(range(YEAR_MIN, YEAR_MAX + 1))
    coverage["Is_balanced_2010_2024"] = coverage["Years_observed"].eq(len(all_years)) & coverage["First_year"].eq(YEAR_MIN) & coverage["Last_year"].eq(YEAR_MAX)

    policy_events, policy_registry = build_policy_registry(df, OUTPUT_DIR)

    ai_terms = [c for c in df.columns if any(term in str(c).casefold() for term in ["artificial intelligence", "ai patent", "ai publication", "machine learning", "ai skill", "ai capacity"])]
    has_ai_specific = len(ai_terms) > 0
    var_dictionary = pd.DataFrame([
        {"Variable": "Y", "Role": "Outcome", "Interpretation": "Rebuilt entropy-weighted Green Industrial Leapfrogging index: manufacturing value added (+), GDP carbon intensity (-), renewable electricity generation share (+). Carbon intensity is imputed only via the documented hierarchy and each imputed row is flagged; other Y components are not imputed."},
        {"Variable": "Y_original_supplied", "Role": "Audit-only", "Interpretation": "Original supplied Y preserved for traceability; not used as the primary outcome after rebuilding Y."},
        {"Variable": "Carbon_Intensity_GDP", "Role": "Y component", "Interpretation": "Carbon intensity of GDP (kg CO2e per 2021 PPP $ GDP); negative direction in Y, higher intensity means worse environmental performance."},
        {"Variable": "X1", "Role": "Main predictor", "Interpretation": "AI-enabling digital capability index. Available indicators (internet use, secure servers, broadband, mobile subscriptions, scientific/technical publications) are enabling infrastructure/capability measures, not direct AI adoption or AI-specific capability."},
        {"Variable": "X2", "Role": "Main predictor / mechanism", "Interpretation": "User-supplied Digital Transformation composite; audit indicator definitions and direction."},
        {"Variable": "X3", "Role": "Main predictor / mechanism", "Interpretation": "Primary sustainable-innovation index rebuilt from R&D expenditure and researchers in R&D only; renewable-energy consumption is excluded to avoid conceptual overlap with Y."},
        {"Variable": "Institution", "Role": "Moderator / predictor", "Interpretation": "User-supplied institutional-quality composite; verify source scale and year coverage."},
        {"Variable": "Y_definition_risk", "Role": "Audit flag", "Interpretation": "Supplied Y is retained. If its emissions component is total CO2 emissions, it is scale-dependent and not equivalent to carbon intensity per GDP."},
        {"Variable": "FDI_ihs", "Role": "Control", "Interpretation": "Recomputed as asinh(raw FDI); retains negative, zero, and positive values."},
        {"Variable": "ED_log", "Role": "Control", "Interpretation": "Natural log of positive GDP per capita; nonpositive observations remain missing."},
    ])
    ai_note = pd.DataFrame([{
        "Audit": "AI-specific measurement",
        "Result": "AI-specific indicator found: " + "; ".join(ai_terms) if has_ai_specific else "No AI-specific indicator detected in source column names.",
        "Action": "If no direct AI indicators are present, describe X1 as AI-enabling digital capability/proxy and avoid claiming direct AI capability measurement."
    }, {
        "Audit": "Carbon-intensity imputation",
        "Result": "Carbon intensity only: internal within-country linear interpolation; remaining gaps filled with region-year median; final fallback is study-sample year median. Flags and method labels are stored per row.",
        "Action": "Report imputed share and methods. Treat imputed values as estimated, not observed. Consider a complete-observed-carbon sensitivity check only if required by reviewers; do not impute other variables automatically."
    }, {
        "Audit": "Upstream imputation provenance — IMPORTANT",
        "Result": "The previously supplied WDI/IRENA cleaning scripts interpolate within countries, apply backfill/forward-fill, and use median fills before merging. If this workbook was produced by those scripts, this preparation script cannot recover which values were originally missing.",
        "Action": "For the strongest paper-ready analysis, rerun data preparation from the original WDI export and original IRENA CSV without those upstream fills. Until then, describe the input as previously processed and do not claim the panel contains only observed values."
    }, {
        "Audit": "Index provenance",
        "Result": "Y, X1, X2, and Institution are preserved from the supplied workbook. X3 is rebuilt from R&D expenditure and R&D researchers only.",
        "Action": "Document original index weights/directions. Verify whether Y uses total CO2 emissions or a valid size-adjusted carbon-intensity measure before strong substantive claims."
    }])

    output_xlsx = OUTPUT_DIR / "Analysis_Ready_Dataset.xlsx"
    output_csv = OUTPUT_DIR / "Analysis_Ready_Dataset.csv"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="AnalysisData", index=False)
        missingness.to_excel(writer, sheet_name="Missingness", index=False)
        coverage.to_excel(writer, sheet_name="PanelCoverage", index=False)
        var_dictionary.to_excel(writer, sheet_name="VariableDictionary", index=False)
        ai_note.to_excel(writer, sheet_name="AuditNotes", index=False)
        if weights_tables:
            pd.concat(weights_tables, ignore_index=True).to_excel(writer, sheet_name="SensitivityIndexWeights", index=False)
        policy_events.to_excel(writer, sheet_name="PolicyEventsSeed", index=False)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    audit_summary = {
        "input_file": str(INPUT_FILE), "source_sheet": source_sheet,
        "output_xlsx": str(output_xlsx), "output_csv": str(output_csv),
        "rows": int(len(df)), "countries": int(df["ID"].nunique()),
        "years": sorted(int(y) for y in df["Year"].unique()),
        "min_year": int(df["Year"].min()), "max_year": int(df["Year"].max()),
        "balanced_countries_2010_2024": int(coverage["Is_balanced_2010_2024"].sum()),
        "core_missingness": {c: int(df[c].isna().sum()) for c in CORE_VARS},
        "controls_available": [c for c in CONTROL_VARS if c in df.columns],
        "ai_specific_columns_found": ai_terms,
        "X3_primary_indicators": [rd_col, researcher_col],
        "Y_primary_indicators": [mfg_col, "Carbon_Intensity_GDP", ren_col],
        "Y_indicator_directions": {mfg_col: "positive", "Carbon_Intensity_GDP": "negative", ren_col: "positive"},
        "Y_valid_observations": int(df["Y"].notna().sum()),
        "carbon_intensity_observed_n": int(df[carbon_raw_col].notna().sum()),
        "carbon_intensity_imputed_n": int(df["Carbon_Intensity_Imputed"].sum()),
        "carbon_intensity_unresolved_n": int(df["Carbon_Intensity_GDP"].isna().sum()),
        "carbon_intensity_imputation_method_counts": {str(k): int(v) for k, v in carbon_method_counts.items()},
        "Y_missing_observations": int(df["Y"].isna().sum()),
        "Y_carbon_intensity_indicator_found": True,
        "Y_carbon_intensity_indicator": carbon_col,
        "carbon_intensity_file": str(CARBON_FILE),
        "policy_registry_file": str(OUTPUT_DIR / "Policy_Registry_To_Complete.xlsx"),
        "policy_candidate_AI_strategy_countries_in_sample": int(policy_registry["AI_Treatment_Year"].notna().sum()),
        "policy_registry_complete": False,
        "upstream_imputation_provenance_requires_review": True,
    }
    with open(OUTPUT_DIR / "Data_Preparation_Audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    log("\n[7/7] DATA PREPARATION COMPLETE")
    log(json.dumps(audit_summary, indent=2, ensure_ascii=False))
    log("Next: run 02_full_analysis.py. Review AuditNotes and VariableDictionary before interpreting results.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        raise
