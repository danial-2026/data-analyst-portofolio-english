# Governance Contagion Intelligence — Public Release

## Files
- `projects/governance-contagion/index.html`: standalone interactive HTML dashboard.

## Suggested GitHub Pages URL
After adding this folder to the root of the existing `danial-2026/data-analyst-portofolio-english` repository and enabling GitHub Pages, the expected path is:
`https://danial-2026.github.io/data-analyst-portofolio-english/projects/governance-contagion/`

The URL becomes public only after the files are committed to the correct repository and Pages finishes publishing.

## Privacy review
This HTML embeds aggregate analytical outputs in JavaScript for client-side visualization. It includes 33 monthly aggregate observations and network-level node/edge metrics. It does not intentionally include the uploaded raw news/article datasets, article headlines, or the local Windows paths from the run manifest.

Important: anything committed to a public repository is visible and downloadable by anyone. Do not add raw news datasets, human annotations, confidential source files, credentials, tokens, personal contact details, or unpublished sensitive material to this public folder.

## Research caveats
The dashboard states that GCI/NCI/EWS/GCR are research-specific analytical proxies and not official BPKH ratings or causal evidence. The model lab is exploratory and does not rerun the original NLP or Isolation Forest pipeline.

## Dependencies
Plotly.js and Cytoscape.js are loaded from public CDNs, so an internet connection is required for those visualizations to render.
