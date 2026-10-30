PUBLICATION DATA FOR PORTFOLIO DASHBOARD

Files:
- publications.json: recommended source for a JavaScript/HTML dashboard (fetch and parse JSON).
- publications.csv: spreadsheet-friendly alternative.

JSON structure:
[
  {
    "id": 1,
    "title": "...",
    "url": "https://...",
    "status": "Published | Accepted",
    "date": "...",
    "journal": "...",
    "indexing": "...",
    "type": "Journal article | Conference proceeding"
  }
]

The records preserve the title order in the supplied CV and pair each title with the URL
in the same order as supplied by the user. Publication status, date, journal, and indexing
are transcribed from the CV. URLs are supplied by the user. Metadata has not been
independently verified against publisher websites. Empty indexing values mean the CV
did not specify an indexing label.

Example dashboard usage:
fetch('./data/publications.json')
  .then(response => response.json())
  .then(publications => {
    // Render publication cards/rows using title, journal, date, status, and url.
  });
