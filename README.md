# DRRM Full-Run Dashboard (Streamlit)

An interactive dashboard for Local DRRM Fund analysis:
- Executive report (entries, totals, utilization rate, QRF proxy, COVID vs non-COVID)
- Per-region rollups + top LGUs per region
- COVID vs non-COVID stacked charts
- Export to PDF

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run DRRM_streamlit_dashboard.py
