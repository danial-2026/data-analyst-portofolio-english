# DIFRI Research Observatory

Interactive public dashboard for consumer-generated digital financial risk signals in Indonesia.

## Run locally
Because the dashboard fetches CSV files with JavaScript, do not open `index.html` directly with `file://`.
Run a local web server from this folder, for example:
- Python: `python -m http.server 8000`
- Open `http://localhost:8000`

## Publish on GitHub Pages
1. Create a GitHub repository (e.g. `difri-dashboard`).
2. Upload the contents of this folder so `index.html` is at the repository root and `data/` is beside it.
3. In **Settings → Pages**, choose **Deploy from a branch**, branch `main`, folder `/ (root)`.
4. Save and wait for GitHub Pages to publish.

## Data
The dashboard loads the CSV files in `data/` directly. Keep the filenames and folder structure unchanged.
The package includes derived/aggregated analytical datasets only; it does not include raw review text or the 2.03 GB SQLite corpus.

## Scope and caveats
- 122 applications across Banking, Payment, and Pindar.
- 4,232,749 valid unique reviews; October 2024–September 2026.
- September 2026 is partial/provisional through 19 September.
- Consumer reviews are self-selected, not a probability sample.
- Severity is an exploratory rule-based proxy, not human-validated realized loss.
- DIFRI is not a population prevalence estimate or direct measure of digital resilience.

## Mobile support
The interface is responsive for phones and tablets:
- On small screens, the left navigation becomes a slide-in drawer with a tap-to-dismiss backdrop.
- KPI cards and content panels reflow into a single-column layout where appropriate.
- Charts resize to the viewport and data tables remain horizontally scrollable.
- Controls have touch-friendly sizing and visible keyboard focus states.
