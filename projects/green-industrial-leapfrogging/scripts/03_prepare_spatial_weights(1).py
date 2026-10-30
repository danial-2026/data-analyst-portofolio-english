#!/usr/bin/env python3
"""
03_prepare_spatial_weights.py
Preparation-only stage for spatial analysis. Script 2 remains responsible for
Moran's I, LISA, SAR/SDM/Spatial Error Model estimation.

Inputs (place in the same folder, or edit CONFIG below):
  1) Analysis_Ready_Dataset.xlsx (sheet AnalysisData; change if needed)
  2) Natural Earth Admin 0 countries shapefile, e.g. ne_110m_admin_0_countries.shp
     plus its companion .dbf/.shx/.prj files in the same directory.

Outputs:
  Spatial_Preparation.xlsx
  spatial_W_queen_row_standardized.csv
  spatial_W_knn5_row_standardized.csv
  spatial_country_lookup.csv
  spatial_preparation_report.txt

Important:
- Queen contiguity is the preferred primary W if valid polygon coverage is adequate.
- Countries with no polygon neighbours are connected using a documented nearest-neighbour
  fallback solely to avoid islands/isolates in W; the report identifies them.
- KNN5 based on representative-point coordinates is a secondary alternative W.
- Coordinates are not regression predictors; they are used only to construct W.
"""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Analysis_Ready_Dataset.xlsx"
INPUT_SHEET = "AnalysisData"
OUTPUT_DIR = BASE_DIR / "Spatial_Preparation"
SHAPEFILE = BASE_DIR / "ne_110m_admin_0_countries.shp"

# Edit these if your workbook uses different names.
COUNTRY_CODE_CANDIDATES = ["ID", "Country Code", "Country_Code", "ISO3", "iso_a3", "CountryCode"]
COUNTRY_NAME_CANDIDATES = ["Country", "Country Name", "Country_Name", "name", "NAME"]
YEAR_CANDIDATES = ["Year", "year"]

def first_existing(columns, candidates, required=True):
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise KeyError(f"Kolom tidak ditemukan. Kandidat: {candidates}; kolom tersedia: {list(columns)}")
    return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input tidak ditemukan: {INPUT_FILE}\n"
            "Letakkan Analysis_Ready_Dataset.xlsx di folder script atau ubah INPUT_FILE."
        )
    if not SHAPEFILE.exists():
        raise FileNotFoundError(
            f"Shapefile tidak ditemukan: {SHAPEFILE}\n"
            "Unduh Natural Earth Admin 0 Countries (1:110m), ekstrak seluruh file pendamping "
            "(.shp, .shx, .dbf, .prj), lalu letakkan di folder script. "
            "Jangan memakai file .shp saja."
        )

    try:
        import geopandas as gpd
        from libpysal.weights import Queen, KNN
    except ImportError as e:
        raise ImportError(
            "Paket spasial belum tersedia. Jalankan: pip install geopandas libpysal shapely pyogrio"
        ) from e

    df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)
    code_col = first_existing(df.columns, COUNTRY_CODE_CANDIDATES)
    name_col = first_existing(df.columns, COUNTRY_NAME_CANDIDATES, required=False)
    year_col = first_existing(df.columns, YEAR_CANDIDATES)

    df[code_col] = df[code_col].astype(str).str.strip().str.upper()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    country_table = df[[code_col] + ([name_col] if name_col else [])].drop_duplicates()
    if country_table[code_col].duplicated().any():
        raise ValueError("Country code tidak unik pada tabel referensi negara.")

    # Read polygons. Different Natural Earth releases can store -99 or non-ISO
    # codes in one ISO field, so test ALL available code fields and then use a
    # conservative country-name fallback for unmatched panel countries.
    world = gpd.read_file(SHAPEFILE)
    iso_candidates = [c for c in ["ADM0_A3", "SOV_A3", "ISO_A3", "GU_A3", "iso_a3"] if c in world.columns]
    if not iso_candidates:
        raise KeyError(f"Tidak ada field kode negara yang dikenali. Kolom shapefile: {list(world.columns)}")
    name_shape_candidates = [c for c in ["ADMIN", "NAME", "NAME_EN", "NAME_LONG", "SOVEREIGNT", "FORMAL_EN", "FORMAL_FR", "BRK_NAME", "NAME_CIAWF", "NAME_SORT", "GEOUNIT", "SUBUNIT"] if c in world.columns]
    world = world[iso_candidates + name_shape_candidates + ["geometry"]].copy()
    world = world[world.geometry.notna() & ~world.geometry.is_empty]

    def norm_name(value):
        import re, unicodedata
        if pd.isna(value):
            return ""
        value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^A-Z0-9]", "", value.upper())

    # Canonical panel ISO3 -> acceptable Natural Earth name variants.
    name_aliases = {
        "BHR": ["Bahrain", "Kingdom of Bahrain"],
        "COM": ["Comoros", "Union of the Comoros", "Comores"],
        "CPV": ["Cabo Verde", "Cape Verde", "Republic of Cabo Verde"],
        "MDV": ["Maldives", "Republic of Maldives"],
        "MUS": ["Mauritius", "Republic of Mauritius"],
        "SGP": ["Singapore", "Republic of Singapore"],
        "SSD": ["South Sudan", "Republic of South Sudan"],
        "STP": ["Sao Tome and Principe", "São Tomé and Príncipe", "Sao Tome & Principe", "Democratic Republic of Sao Tome and Principe"],
        "SYC": ["Seychelles", "Republic of Seychelles"]
    }
    # Create one canonical polygon key, preferring a matching panel ISO code from
    # any available Natural Earth field; otherwise match by verified country name.
    panel_codes_set = set(country_table[code_col].astype(str))
    rows = []
    for _, r in world.iterrows():
        keys = [str(r[c]).strip().upper() for c in iso_candidates if pd.notna(r[c]) and str(r[c]).strip().upper() not in {"-99", "-99.0", "NONE", "NAN"}]
        matched_code = next((k for k in keys if k in panel_codes_set), None)
        shape_names = [norm_name(r[c]) for c in name_shape_candidates if pd.notna(r[c])]
        if matched_code is None:
            for pcode, aliases_for_code in name_aliases.items():
                alias_norms = {norm_name(a) for a in aliases_for_code}
                # First use exact normalized names. Then allow containment only for
                # sufficiently long canonical names to accommodate variants such as
                # "Republic of ..." without fuzzy-matching unrelated countries.
                exact_match = any(n in alias_norms for n in shape_names)
                contained_match = any(
                    len(a) >= 6 and any(a in n or n in a for n in shape_names if len(n) >= 6)
                    for a in alias_norms
                )
                if exact_match or contained_match:
                    matched_code = pcode
                    break
        if matched_code is not None:
            rows.append({"ISO3": matched_code, "geometry": r.geometry})
    matched_world = gpd.GeoDataFrame(rows, geometry="geometry", crs=world.crs)
    matched_world = matched_world.dissolve(by="ISO3", as_index=False)

    # Exact code matching across all available Natural Earth fields above; name fallback
    # is used only for the nine codes previously reported as unmatched.
    shape_codes = set(matched_world["ISO3"])
    country_table["Shape_ISO3"] = country_table[code_col].where(country_table[code_col].isin(shape_codes), np.nan)
    unmatched = country_table[country_table["Shape_ISO3"].isna()][code_col].tolist()
    lookup = country_table.dropna(subset=["Shape_ISO3"]).merge(
        matched_world, left_on="Shape_ISO3", right_on="ISO3", how="left"
    )
    lookup = lookup.drop_duplicates("Shape_ISO3")
    if len(lookup) < 3:
        raise ValueError("Kurang dari 3 negara cocok dengan peta. Periksa kode ISO3 dan shapefile.")

    # Representative points are guaranteed to lie within each polygon; used for KNN and fallback.
    lookup = gpd.GeoDataFrame(lookup, geometry="geometry", crs=world.crs)
    points = lookup.copy()
    points["geometry"] = points.geometry.representative_point()
    points = points.set_geometry("geometry")
    # WGS84 coordinates for the lookup table (descriptive / distance calculations only).
    points_ll = points.to_crs("EPSG:4326")
    points["Longitude"] = points_ll.geometry.x
    points["Latitude"] = points_ll.geometry.y

    # Queen contiguity; report isolates and use nearest neighbour only to connect isolates.
    queen = Queen.from_dataframe(lookup, use_index=False)
    queen_neighbors = {i: list(v) for i, v in queen.neighbors.items()}
    isolates_idx = list(queen.islands)
    # KNN computed in projected CRS to avoid longitude/latitude distortion.
    metric_crs = lookup.estimate_utm_crs()
    if metric_crs is None:
        metric_crs = "EPSG:6933"
    pts_metric = points.to_crs(metric_crs)
    xy = np.column_stack([pts_metric.geometry.x, pts_metric.geometry.y])
    knn = KNN.from_array(xy, k=min(5, len(points)-1))
    # Queen adjacency matrix, symmetrized. Isolates get nearest-neighbour links.
    n = len(lookup)
    Wq = np.zeros((n, n), dtype=float)
    for i, js in queen_neighbors.items():
        for j in js:
            Wq[i, j] = 1.0
            Wq[j, i] = 1.0
    isolate_codes = []
    code_values = lookup[code_col].astype(str).tolist()
    for i in isolates_idx:
        d = np.sqrt(((xy - xy[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        j = int(np.argmin(d))
        Wq[i, j] = Wq[j, i] = 1.0
        isolate_codes.append(code_values[i])

    Wk = np.zeros((n, n), dtype=float)
    for i, js in knn.neighbors.items():
        for j in js:
            Wk[i, j] = 1.0
    # Symmetrize KNN so neighbour relation is not directional, then row-standardize.
    Wk = np.maximum(Wk, Wk.T)
    def row_standardize(W):
        sums = W.sum(axis=1, keepdims=True)
        if np.any(sums == 0):
            raise ValueError("W memiliki baris tanpa tetangga setelah penanganan isolate.")
        return W / sums
    Wq_rs, Wk_rs = row_standardize(Wq), row_standardize(Wk)

    ids = lookup[code_col].astype(str).tolist()
    expected_codes = set(df[code_col].astype(str).unique())
    missing_from_W = sorted(expected_codes - set(ids))
    if missing_from_W:
        # Save diagnostics listing likely name/code fields from the source shapefile.
        diagnostic_cols = [c for c in iso_candidates + name_shape_candidates if c in gpd.read_file(SHAPEFILE).columns]
        try:
            raw_shapes = gpd.read_file(SHAPEFILE)
            raw_shapes[diagnostic_cols].drop_duplicates().to_csv(
                OUTPUT_DIR / "shapefile_country_name_code_diagnostics.csv", index=False
            )
        except Exception:
            pass
        raise ValueError(
            f"Matriks spasial belum mencakup seluruh negara panel ({len(ids)}/{len(expected_codes)}). "
            f"Belum cocok: {', '.join(missing_from_W)}. W tidak ditulis. "
            f"Lihat shapefile_country_name_code_diagnostics.csv di {OUTPUT_DIR} untuk nama/kode asli shapefile."
        )
    pd.DataFrame(Wq_rs, index=ids, columns=ids).to_csv(OUTPUT_DIR / "spatial_W_queen_row_standardized.csv", index_label="Country_ID")
    pd.DataFrame(Wk_rs, index=ids, columns=ids).to_csv(OUTPUT_DIR / "spatial_W_knn5_row_standardized.csv", index_label="Country_ID")

    coords = points[[code_col, "Longitude", "Latitude"]].copy()
    if name_col:
        coords = coords.merge(country_table[[code_col, name_col]].drop_duplicates(), on=code_col, how="left")
    coords["Shape_ISO3"] = lookup["Shape_ISO3"].values
    coords.to_csv(OUTPUT_DIR / "spatial_country_lookup.csv", index=False)

    # Enrich panel with verified-from-polygon representative points; never overwrite existing fields silently.
    enriched = df.merge(coords, on=code_col, how="left", suffixes=("", "_spatial"))
    enriched.to_excel(OUTPUT_DIR / "Spatial_Preparation.xlsx", index=False, sheet_name="SpatialData")

    wq_degree = Wq.sum(axis=1)
    wk_degree = Wk.sum(axis=1)
    report = [
        "SPATIAL PREPARATION REPORT",
        f"Input: {INPUT_FILE.name} [{INPUT_SHEET}]",
        f"Panel rows: {len(df)}",
        f"Panel countries: {df[code_col].nunique()}",
        f"Countries matched to polygons: {len(lookup)}",
        f"Countries unmatched to polygons: {len(unmatched)}",
        f"Unmatched panel ISO3 codes: {', '.join(unmatched) if unmatched else 'None'}",
        f"Queen islands before fallback: {len(isolates_idx)}",
        f"Queen isolate codes linked to nearest polygon representative point: {', '.join(isolate_codes) if isolate_codes else 'None'}",
        f"Queen W mean neighbours after fallback: {wq_degree.mean():.3f}",
        f"KNN W k requested: {min(5, len(points)-1)}",
        f"KNN W mean neighbours after symmetrization: {wk_degree.mean():.3f}",
        "Both matrices are row-standardized; diagonal is zero.",
        "Coordinates are representative points inside country polygons, not necessarily capital coordinates.",
        "Queen W is the primary candidate; KNN W is an alternative specification.",
        "Coverage check passed: every panel country is included in both W matrices.",
        "No Moran's I, LISA, SAR, SDM, or Spatial Error Model is estimated by this preparation script."
    ]
    (OUTPUT_DIR / "spatial_preparation_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nOutput folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
