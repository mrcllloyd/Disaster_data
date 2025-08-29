
# DRRM Full-Run Dashboard (Unified CSV, Extensive Discussion)
# -----------------------------------------------------------
# Loads a single unified CSV that contains all chapter tables, with a Table_Source column.
# Provides deep, in-app explanations anchored to the chapter's narrative.
#
# Run: streamlit run DRRM_streamlit_dashboard_unified.py

import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="DRRM Dashboard", layout="wide")

# ---------- Helpers ----------
def currency(x):
    try: return f"₱{float(x):,.0f}"
    except: return "—"

def pct(x, d=1):
    try: return f"{100*float(x):.{d}f}%"
    except: return "—"

def to_num(v):
    if v is None: return np.nan
    try: return float(v)
    except:
        try:
            s = str(v).replace(",", "").replace("₱", "")
            return float(s)
        except: return np.nan

def load_unified_default_or_upload(default_path: Path):
    st.sidebar.subheader("Data")
    up = st.sidebar.file_uploader("Upload Unified CSV (with Table_Source)", type=["csv"], key="unified")
    if up is not None:
        df = pd.read_csv(up)
        st.sidebar.success("Loaded uploaded file")
        return df
    if default_path.exists():
        st.sidebar.caption(f"Loaded default: {default_path}")
        return pd.read_csv(default_path)
    st.sidebar.error("No unified CSV available. Please upload one.")
    return None

def export_pdf(pages):
    """pages: list of (title:str, fig:plt.Figure)"""
    bio = io.BytesIO()
    with PdfPages(bio) as pdf:
        for title, fig in pages:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    bio.seek(0)
    st.download_button("Download PDF Report", data=bio, file_name="DRRM_Unified_Report.pdf", mime="application/pdf")

# ---------- Load Data ----------
BASE = Path(__file__).resolve().parent
DEFAULT_PATH = BASE / "csv_from_chapter" / "Unified_DRRRM_Tables.csv"
df = load_unified_default_or_upload(DEFAULT_PATH)

st.title("DRRM Dashboard — Unified CSV")
st.caption("Executive report with in-app discussion & analysis. Source tables consolidated into one file via `Table_Source`.")

if df is None:
    st.stop()

# Normalization
if "Table_Source" not in df.columns:
    st.error("Unified CSV must contain a 'Table_Source' column. Please upload the merged file.")
    st.stop()

table_options = sorted(df["Table_Source"].dropna().unique().tolist())
table_sel = st.sidebar.selectbox("Select table (Table_Source)", table_options, index=0)
filtered = df[df["Table_Source"] == table_sel].copy()

# ---------- Executive Section ----------
with st.expander(" Narrative", expanded=True):
    st.markdown("""
**What this dashboard shows**  
- *Allocations* across LGU levels and regions (*Tables 21-1 to 21-5*), and *utilization* patterns (COVID vs non-COVID, financing gaps, and class-level utilization rates).  
- Use the selector in the left sidebar to switch between tables; each view includes **discussion** tying the data to policy and programming choices.

**Why this matters**  
- **LDRRMF** is the primary local DRM funding source; **30%** is the **QRF** for immediate response; **70%** supports ex-ante DRR PPAs.  
- Many LGUs exhibit **under-utilization**, leading to large **Special Trust Funds (STFs)** and delayed DRR implementation.  
- During **COVID-19 (2020)**, utilization spiked where pandemic expenditures were allowed; this can mask underlying DRR execution gaps.  
- **Financing gaps** persist in high-impact events even after pooling local QRFs; **risk pooling** (PCDIP/parametric) and **inter-local mutual aid** can help close liquidity shortfalls.

**How to read**  
- **City/Province/Region Funding** tables show the **composition** of DRM resources (LDRRMF, STFs, inter-LGU transfers, NDRRMF, other sources).  
- **Utilization** tables contrast **COVID vs non-COVID** spending, highlight **gaps**, and show **utilization rates** by income class.
""")

st.markdown("---")

# ---------- View Logic per Table ----------
pages_for_pdf = []

def show_table(table_name: str, data: pd.DataFrame):
    st.subheader(table_name.replace("_", " "))
    st.dataframe(data)

def add_pdf_page(title, fig):
    pages_for_pdf.append((title, fig))

# Common control: Top-N
topn = st.sidebar.slider("Top N (for charts)", min_value=5, max_value=50, value=10, step=5)

# 1) Table21_1_CityFunding
if table_sel == "Table21_1_CityFunding":
    st.markdown("### City Funding — Composition & Concentration")
    st.markdown("""
**Discussion.** This table shows **city-level funding composition**: **LDRRMF** (core), **STFs** (carry-overs), inter-LGU transfers, **NDRRMF**, and **other sources**.  
Look for **concentration**: a handful of cities may account for a large share of total funding while others contribute small shares — a sign that **fiscal capacity** and **reporting completeness** differ.
**Implications:** Cities with **high resources** but **low utilization** should prioritize **execution** (planning, procurement), and consider **pre-arranged** protection (premium set-asides via STFs) for **liquidity-on-event**.
""")
    show_table("Table21_1_CityFunding", filtered)

    # Chart: Top N by Total Funding
    cols = [c for c in filtered.columns if c.lower().startswith("total")]
    total_col = cols[0] if cols else "Total Funding"
    dfp = filtered.sort_values(total_col, ascending=False).head(topn)
    x = dfp[dfp.columns[0]].astype(str).tolist()
    y = pd.to_numeric(dfp[total_col], errors="coerce").fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(9,4))
    ax.barh(x[::-1], y[::-1])
    ax.set_title(f"Top {topn} Cities by {total_col}")
    ax.set_xlabel("PHP")
    st.pyplot(fig)
    add_pdf_page("Top Cities — Total Funding", fig)

# 2) Table21_2_CityFundingByClass
elif table_sel == "Table21_2_CityFundingByClass":
    st.markdown("### City Funding by Income Class — Distribution & Equity")
    st.markdown("""
**Discussion.** Funding distribution by **income class** reveals whether higher-capacity cities dominate **allocations**.  
If **Class 1** cities hold most of the resources, consider balancing **DRR execution support** for lower-income cities (targeted TA, pooled coverage, or earmarked grants).
""")
    show_table("Table21_2_CityFundingByClass", filtered)

    # Chart: Total Funding by Class
    if "Total Funding" in filtered.columns and "Class" in filtered.columns:
        dfp = filtered.copy()
        dfp["Total Funding"] = pd.to_numeric(dfp["Total Funding"], errors="coerce")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(dfp["Class"].astype(str), dfp["Total Funding"])
        ax.set_title("City Total Funding by Income Class")
        ax.set_ylabel("PHP")
        st.pyplot(fig)
        add_pdf_page("City Funding by Class", fig)

# 3) Table21_3_ProvinceFunding_Consolidated
elif table_sel == "Table21_3_ProvinceFunding_Consolidated":
    st.markdown("### Province Funding (Consolidated City/Municipal Reports)")
    st.markdown("""
**Discussion.** Consolidated figures can **overlap** with standalone provincial reports. Use this alongside **Table 21-4** to see how **provincial** resources compare when **cities/municipalities** are included vs **standalone** provincial data.
**Analytical cue:** Check **STF magnitudes** — high carry-overs indicate **execution lags** or project bundling across fiscal years.
""")
    show_table("Table21_3_ProvinceFunding_Consolidated", filtered)

    # Chart: Top N Provinces by Total Funding
    total_col = "Total Funding" if "Total Funding" in filtered.columns else filtered.columns[-2]
    dfp = filtered.sort_values(total_col, ascending=False).head(topn)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.barh(dfp["Province"].astype(str)[::-1], pd.to_numeric(dfp[total_col], errors="coerce").fillna(0).tolist()[::-1])
    ax.set_title(f"Top {topn} Provinces by {total_col} (Consolidated)")
    ax.set_xlabel("PHP")
    st.pyplot(fig)
    add_pdf_page("Top Provinces — Consolidated", fig)

# 4) Table21_4_ProvinceFunding_Standalone
elif table_sel == "Table21_4_ProvinceFunding_Standalone":
    st.markdown("### Province Funding (Standalone Provincial Reports)")
    st.markdown("""
**Discussion.** Standalone provincial reports exclude cities/municipalities; compare against **consolidated** to understand **provincial vs LGU-level** resource control.
**Policy angle:** If provincial **STFs** are large while **utilization** is low, consider **time-bound STF targets** and **pooled premium** allocations to convert idle balances into **pre-arranged** protection.
""")
    show_table("Table21_4_ProvinceFunding_Standalone", filtered)

    total_col = "Total Funding" if "Total Funding" in filtered.columns else filtered.columns[-2]
    dfp = filtered.sort_values(total_col, ascending=False).head(topn)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.barh(dfp["Province"].astype(str)[::-1], pd.to_numeric(dfp[total_col], errors="coerce").fillna(0).tolist()[::-1])
    ax.set_title(f"Top {topn} Provinces by {total_col} (Standalone)")
    ax.set_xlabel("PHP")
    st.pyplot(fig)
    add_pdf_page("Top Provinces — Standalone", fig)

# 5) Table21_5_RegionFunding
elif table_sel == "Table21_5_RegionFunding":
    st.markdown("### Region Funding — Composition of Sources")
    st.markdown("""
**Discussion.** Regional allocation patterns help identify **concentration** (e.g., NCR) and **heterogeneity** in **STFs** and **other sources** across regions.
**Use case:** Regions with repeated shocks but **modest QRF-equivalents** might prioritize **pooled/parametric** solutions to assure timely liquidity.
""")
    show_table("Table21_5_RegionFunding", filtered)

    # Stacked bar by major components (if present)
    components = [c for c in ["LDRRMF","STF","From other LGUs","NDRRMF","Other Sources"] if c in filtered.columns]
    if "Region" in filtered.columns and components:
        dfp = filtered.copy()
        for c in components: dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0)
        dfp = dfp.sort_values(sum(dfp[c] for c in components), ascending=False).head(topn)
        idx = np.arange(dfp.shape[0])
        bottoms = np.zeros(dfp.shape[0])
        fig, ax = plt.subplots(figsize=(10,5))
        for comp in components:
            vals = dfp[comp].values
            ax.bar(idx, vals, bottom=bottoms, label=comp)
            bottoms += vals
        ax.set_xticks(idx); ax.set_xticklabels(dfp["Region"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("PHP")
        ax.set_title("Regional Funding — Stacked by Source")
        ax.legend()
        st.pyplot(fig)
        add_pdf_page("Regional Funding — Composition", fig)

# 6) Table21_COVID_vs_NonCOVID_Utilization
elif table_sel == "Table21_COVID_vs_NonCOVID_Utilization":
    st.markdown("### Utilization by Region — COVID vs Non-COVID")
    st.markdown("""
**Discussion.** 2020 saw **COVID-19 eligibility** for LDRRMF utilization. Regions with **high COVID shares** may show **elevated utilization** in 2020 relative to DRR PPAs, which were often **deferred**.  
**Analytical cue:** Map these spikes to **2020 timelines** and check if DRR project execution resumed in subsequent years.
""")
    show_table("Table21_COVID_vs_NonCOVID_Utilization", filtered)

    # Stacked COVID vs Non-COVID
    if set(["Region","COVID_Utilization","NonCOVID_Utilization"]).issubset(filtered.columns):
        dfp = filtered.copy()
        dfp["COVID_Utilization"] = pd.to_numeric(dfp["COVID_Utilization"], errors="coerce").fillna(0)
        dfp["NonCOVID_Utilization"] = pd.to_numeric(dfp["NonCOVID_Utilization"], errors="coerce").fillna(0)
        dfp = dfp.sort_values("Total_Utilization" if "Total_Utilization" in dfp.columns else "COVID_Utilization", ascending=False).head(topn)
        idx = np.arange(dfp.shape[0])
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(idx, dfp["NonCOVID_Utilization"], label="Non-COVID")
        ax.bar(idx, dfp["COVID_Utilization"], bottom=dfp["NonCOVID_Utilization"], label="COVID")
        ax.set_xticks(idx); ax.set_xticklabels(dfp["Region"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("PHP")
        ax.set_title("Regional Utilization — COVID vs Non-COVID")
        ax.legend()
        st.pyplot(fig)
        add_pdf_page("COVID vs Non-COVID (Region)", fig)

# 7) Table21_Financing_Gaps
elif table_sel == "Table21_Financing_Gaps":
    st.markdown("### Financing Gaps — Reported Damages vs Pooled QRF")
    st.markdown("""
**Discussion.** Even after **pooling LGUs’ QRFs**, several regions exhibited **liquidity gaps** against **reported damages** (SitReps).  
**Interpretation:** This underscores the **limitations of a 30% QRF** and supports **risk pooling/parametric** solutions to guarantee **early-recovery liquidity** to LGUs.
""")
    show_table("Table21_Financing_Gaps", filtered)

    # Chart: Gaps
    if set(["Region","Gap_Billion"]).issubset(filtered.columns):
        dfp = filtered.sort_values("Gap_Billion", ascending=False).head(topn)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.barh(dfp["Region"].astype(str)[::-1], pd.to_numeric(dfp["Gap_Billion"], errors="coerce").fillna(0).tolist()[::-1])
        ax.set_title(f"Top {topn} Financing Gaps (₱ Billion)")
        ax.set_xlabel("₱ Billion")
        st.pyplot(fig)
        add_pdf_page("Financing Gaps", fig)

# 8) Table21_Utilization_By_IncomeClass
elif table_sel == "Table21_Utilization_By_IncomeClass":
    st.markdown("### Utilization by Income Class — Rates & Averages")
    st.markdown("""
**Discussion.** Utilization rates by income class provide a **capacity lens**.  
If **lower-income classes** systematically under-spend, the constraint may be **execution capacity** rather than funding — tailor **TA**, simplify **procurement**, and consider **pre-arranged** liquidity so response does not crowd-out DRR PPAs.
""")
    show_table("Table21_Utilization_By_IncomeClass", filtered)

    if "Utilization_Rate" in filtered.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(filtered["Income_Class"].astype(str), pd.to_numeric(filtered["Utilization_Rate"], errors="coerce"))
        ax.set_title("Utilization Rate by Income Class")
        ax.set_ylabel("Rate")
        st.pyplot(fig)
        add_pdf_page("Utilization Rate by Income Class", fig)

# 9) Table21_Top10_LGUs_Utilization
elif table_sel == "Table21_Top10_LGUs_Utilization":
    st.markdown("### Top LGUs by Utilization — Spotlight")
    st.markdown("""
**Discussion.** High-utilizing LGUs are not necessarily the largest by **sources**; instead, they may be executing **carry-overs** or responding to **concentrated shocks**.  
**Actionable:** Share **execution practices** (planning/budget reforms) from top performers to peers; evaluate if **COVID share** is driving the rank.
""")
    show_table("Table21_Top10_LGUs_Utilization", filtered)

    # Chart: Utilization (Million)
    if set(["LGU","Utilization_Million"]).issubset(filtered.columns):
        dfp = filtered.sort_values("Utilization_Million", ascending=False).head(topn)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.barh(dfp["LGU"].astype(str)[::-1], pd.to_numeric(dfp["Utilization_Million"], errors="coerce").fillna(0).tolist()[::-1])
        ax.set_title(f"Top {topn} LGUs by Utilization (Million)")
        ax.set_xlabel("₱ Million")
        st.pyplot(fig)
        add_pdf_page("Top LGUs — Utilization", fig)

# ---------- Download filtered view ----------
st.markdown("---")
st.subheader("Download This View")
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv_bytes, file_name=f"{table_sel}.csv", mime="text/csv")

# ---------- Methods & Citations ----------
with st.expander("Methods & Citations (Read Me)", expanded=False):
    st.markdown("""
**Methods.**  
- **Unified CSV** consolidates the chapter’s tables; each row includes `Table_Source` to identify the origin.  
- Visuals are matched to each `Table_Source` (e.g., stacked source composition for regions; stacked COVID vs non-COVID for utilization).  
- Currency figures use simple number formatting; bars may be sorted by **Total Funding** or **COVID utilization** to surface concentration.

**Caveats.**  
- Some chapter tables are **illustrative** and draw from narrative summaries; use them for **pattern recognition** and discussion, not as audited financial statements.  
- For precise analytics, pair this with your **FDP Form 8 full-run** outputs and PSGC mapping.

**Citations (Chapter: “Risk-Sharing – Exploring Mechanisms to Enhance Local Disaster Resilience”)**  
- LDRRMF makeup: **30% QRF** (standby), **70% DRR**; unspent to **STF** with time-bound rules.  
- **Under-utilization** across LGUs; COVID-19 eligibility led to **2020 spikes**.  
- **OCD sitrep** gaps: regions short against damages even after pooling QRFs (e.g., Region V/Bicol).  
- **PCDIP/parametric** pooling for pre-arranged liquidity; address governance & allocation bottlenecks; institutionalize **inter-local mutual aid**.
""")

# ---------- Export PDF ----------
st.markdown("---")
st.subheader("Export to PDF")
st.caption("Build a printable multi-page PDF of the current view and key charts.")
if st.button("Export current view to PDF"):
    pages = []
    # Page 1: Title + summary
    fig = plt.figure(figsize=(11,8.5)); plt.axis("off")
    lines = [
        "DRRM Dashboard — Unified CSV",
        "",
        f"View: {table_sel}",
        f"Rows: {filtered.shape[0]:,}  |  Columns: {filtered.shape[1]:,}",
        "",
        "Notes: Use with the full-run FDP Form 8 results for precise program decisions."
    ]
    plt.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=12)
    pages.append(("Summary", fig))

    # Try to regenerate the chart for the selected table and attach
    # (Reuse the logic above in a minimal way)
    try:
        if table_sel == "Table21_1_CityFunding":
            cols = [c for c in filtered.columns if c.lower().startswith("total")]
            total_col = cols[0] if cols else "Total Funding"
            dfp = filtered.sort_values(total_col, ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(11,8.5))
            ax.barh(dfp[dfp.columns[0]].astype(str)[::-1], pd.to_numeric(dfp[total_col], errors="coerce").fillna(0).tolist()[::-1])
            ax.set_title("Top Cities by Total Funding"); ax.set_xlabel("PHP")
            pages.append(("Top Cities", fig))

        elif table_sel == "Table21_5_RegionFunding":
            components = [c for c in ["LDRRMF","STF","From other LGUs","NDRRMF","Other Sources"] if c in filtered.columns]
            if components and "Region" in filtered.columns:
                dfp = filtered.copy()
                for c in components: dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0)
                dfp = dfp.sort_values(sum(dfp[c] for c in components), ascending=False).head(12)
                idx = np.arange(dfp.shape[0]); bottoms = np.zeros(dfp.shape[0])
                fig, ax = plt.subplots(figsize=(11,8.5))
                for comp in components:
                    vals = dfp[comp].values
                    ax.bar(idx, vals, bottom=bottoms, label=comp); bottoms += vals
                ax.set_xticks(idx); ax.set_xticklabels(dfp["Region"].astype(str), rotation=45, ha="right")
                ax.set_title("Regional Funding — Composition"); ax.set_ylabel("PHP"); ax.legend()
                pages.append(("Regional Composition", fig))

        elif table_sel == "Table21_COVID_vs_NonCOVID_Utilization":
            dfp = filtered.copy()
            dfp["COVID_Utilization"] = pd.to_numeric(dfp["COVID_Utilization"], errors="coerce").fillna(0)
            dfp["NonCOVID_Utilization"] = pd.to_numeric(dfp["NonCOVID_Utilization"], errors="coerce").fillna(0)
            dfp = dfp.sort_values("Total_Utilization" if "Total_Utilization" in dfp.columns else "COVID_Utilization", ascending=False).head(12)
            idx = np.arange(dfp.shape[0])
            fig, ax = plt.subplots(figsize=(11,8.5))
            ax.bar(idx, dfp["NonCOVID_Utilization"], label="Non-COVID")
            ax.bar(idx, dfp["COVID_Utilization"], bottom=dfp["NonCOVID_Utilization"], label="COVID")
            ax.set_xticks(idx); ax.set_xticklabels(dfp["Region"].astype(str), rotation=45, ha="right")
            ax.set_title("COVID vs Non-COVID Utilization"); ax.set_ylabel("PHP"); ax.legend()
            pages.append(("COVID vs Non-COVID", fig))

        elif table_sel == "Table21_Financing_Gaps":
            dfp = filtered.sort_values("Gap_Billion", ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(11,8.5))
            ax.barh(dfp["Region"].astype(str)[::-1], pd.to_numeric(dfp["Gap_Billion"], errors="coerce").fillna(0).tolist()[::-1])
            ax.set_title("Top Financing Gaps (₱ Billion)"); ax.set_xlabel("₱ Billion")
            pages.append(("Financing Gaps", fig))

        elif table_sel == "Table21_Utilization_By_IncomeClass":
            fig, ax = plt.subplots(figsize=(11,8.5))
            ax.bar(filtered["Income_Class"].astype(str), pd.to_numeric(filtered["Utilization_Rate"], errors="coerce"))
            ax.set_title("Utilization Rate by Income Class"); ax.set_ylabel("Rate")
            pages.append(("Utilization by Class", fig))

        elif table_sel == "Table21_Top10_LGUs_Utilization":
            dfp = filtered.sort_values("Utilization_Million", ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(11,8.5))
            ax.barh(dfp["LGU"].astype(str)[::-1], pd.to_numeric(dfp["Utilization_Million"], errors="coerce").fillna(0).tolist()[::-1])
            ax.set_title("Top LGUs by Utilization (Million)"); ax.set_xlabel("₱ Million")
            pages.append(("Top LGUs", fig))

    except Exception as e:
        st.warning(f"Could not generate a chart for PDF: {e}")

    # Export
    export_pdf(pages)
