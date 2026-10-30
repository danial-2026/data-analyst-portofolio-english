# DIFRI dashboard data dictionary

All CSV files in `data/` are derived analytical outputs. They do not contain raw review text.

| File | Grain / contents | Important fields |
|---|---|---|
| `national_monthly.csv` | One row per month; national DIFRI and robustness alternatives | `review_month`, `SF`, `DP`, `TO`, `CP`, `DIFRI`, `month_status` |
| `national_pillar_monthly.csv` | One row per month and pillar | `pillar`, `prevalence_eb`, `severity_burden`, `pillar_score` |
| `sector_monthly.csv` | One row per sector, month, and pillar; includes sector DIFRI | `sector`, `review_month`, `pillar`, `prevalence_eb`, `pillar_score`, `DIFRI_sector` |
| `sector_pillar_monthly.csv` | Sector-month-pillar panel | `sector`, `review_month`, `pillar`, `prevalence_eb`, `pillar_score`, `DIFRI_sector` |
| `app_month_pillar.csv` | App-month-pillar estimates | `sector`, `app_id`, `review_month`, `total_reviews`, `pillar`, `risk_n`, `prevalence_eb`, `mean_severity_conditional`, `pillar_severity_burden` |
| `eb_cap_sensitivity.csv` | Monthly comparison of main and capped EB specification | `DIFRI_main`, `DIFRI_EB_cap1000`, difference |
| `robustness_aggregation.csv` | Summary statistics by aggregation specification | `Specification`, `Mean`, `Median`, `SD`, correlation with reference |
| `monte_carlo_uncertainty.csv` | Monthly Monte Carlo summary for probability-weighted estimator | `mc_mean`, `mc_lower_95`, `mc_upper_95`, `probability_weighted_point_estimate` |
| `partial_month_benchmark.csv` | 24-month versus 23-month benchmark | `Dimension`, `Mean_24_month`, `Mean_23_month_excluding_partial` |
| `annotation_reliability.csv` | Pairwise annotator agreement | `label`, `annotator_1`, `annotator_2`, `agreement`, `kappa` |
| `model_selection.csv` | Candidate and selected classifier performance | `Pillar`, `Model`, `PR-AUC`, `F1`, `Precision`, `Recall`, `Threshold`, `Selected` |
| `risk_prevalence_by_sector.csv` | Predicted prevalence by sector and risk dimension | `Sector`, `Risk dimension`, `Risk reviews`, `Total reviews`, `Prevalence (%)` |
| `severity_reference.csv` | Sector-month severity reference estimates | See CSV headers |
| `difri_monthly_main.csv` | Compact monthly DIFRI and pillar series | `Month`, `SF`, `DP`, `TO`, `CP`, `DIFRI`, `Month status` |

## Pillar codes
- `SF`: Security and Fraud
- `DP`: Data and Privacy
- `TO`: Transaction and Operational
- `CP`: Consumer Protection

## Units and interpretation
- DIFRI/pillar score fields are index scores on a 0–4 scale, unless the field explicitly says `Prevalence (%)` or `prevalence_*`.
- `prevalence_eb` is an Empirical-Bayes adjusted prevalence estimate, stored as a proportion (0–1).
- `Prevalence (%)` is stored in percentage units (0–100).
- `pillar_severity_burden` and `severity_burden` are severity-weighted burden measures on the study's index scale.
- `month_status` flags the partial September 2026 period.

## Limitations
App-month estimates are derived indicators, not individual complaint records. The severity component is an exploratory keyword proxy, not validated realized harm. Consumer-generated app reviews are self-selected and may not represent all digital financial service users.
