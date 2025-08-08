import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="TX District HIT Prioritization", layout="wide")

# Simple password protection (Cloud-friendly)
import os
import hashlib

def _get_app_password_hash() -> str:
    # 1) Streamlit Secrets (recommended on Cloud)
    try:
        secret_hash = st.secrets.get("APP_PASSWORD_HASH", "").strip()
        if secret_hash:
            return secret_hash
    except Exception:
        pass
    # 2) Environment variable (for local dev or alternate deployment)
    env_hash = os.environ.get("APP_PASSWORD_HASH", "").strip()
    if env_hash:
        return env_hash
    # 3) Hardcoded fallback (one-off)
    return "9ebb6846136a4506cd9f3b5893dc5d0de77201076a269dce2edf40a72085b268"

APP_PASSWORD_HASH = _get_app_password_hash()

def _rerun():
    r = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(r):
        r()

def check_password() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.title("Texas District High-Impact Tutoring (H.I.T.) Prioritization")
    st.subheader("Access")
    with st.form("login_form", clear_on_submit=False):
        pwd = st.text_input("Enter password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if hashlib.sha256(pwd.encode()).hexdigest() == APP_PASSWORD_HASH:
            st.session_state["authenticated"] = True
            _rerun()
        else:
            st.error("Incorrect password. Please try again.")
    return False

if not check_password():
    st.stop()

# After authenticated, show a logout in the sidebar
with st.sidebar:
    if st.button("Log out"):
        st.session_state.pop("authenticated", None)
        _rerun()

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    # Robust CSV load: tolerate odd quoting and NA tokens
    df = pd.read_csv(
        path,
        engine="python",
        dtype=str,
        na_values=["NA", "N/A", "", "nan", "None", "null", "Null"]
    )
    # Strip whitespace from columns
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    # Coerce numerics
    for col in ["TOTAL ENROLLMENT", "Percent Economically Disadvantaged", "STATE RANK"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalize simple booleans
    for col in ["Charter School?", "Gold Ribbon School?", "Gold Ribbon Eligible?"]:
        if col in df.columns:
            df[col] = df[col].str.title()
            df[col] = df[col].replace({"Yes": True, "No": False})
    # Standardize key text fields
    for col in ["DISTRICT NAME", "CAMPUS NAME", "County Name", "Local Region", "School Level"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def grade_to_numeric(letter: str) -> float:
    if not isinstance(letter, str):
        return np.nan
    s = letter.strip().upper()
    # Handle cases like " or stray characters
    s = s.replace('"', "").replace("'", "")
    base_map = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
    sign = 0.0
    if len(s) >= 2 and s[1] in ["+", "-"]:
        if s[1] == "+":
            sign = 0.3
        elif s[1] == "-":
            sign = -0.3
        s = s[0]
    if s not in base_map:
        return np.nan
    val = base_map[s] + sign
    # Cap at typical 4.3 scale for A+
    return float(np.clip(val, 0.0, 4.3))


def compute_scoring(df: pd.DataFrame,
                    campus_weights,
                    high_need_threshold,
                    min_campus_enrollment,
                    include_charters) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (campus_df_with_scores, district_agg_df)
    """
    working = df.copy()

    # Filter charters if needed
    if not include_charters and "Charter School?" in working.columns:
        working = working[(working["Charter School?"] == False) | (working["Charter School?"].isna())]

    # Minimum campus enrollment filter (to remove very small edge cases)
    if "TOTAL ENROLLMENT" in working.columns:
        working = working[working["TOTAL ENROLLMENT"].fillna(0) >= min_campus_enrollment]

    # Grade conversions
    ach_col = "Student Achievement Grade"
    growth_col = "Growth Grade"
    cr_col = "College Readiness Grade"

    for c in [ach_col, growth_col, cr_col]:
        if c not in working.columns:
            working[c] = np.nan

    working["AchScore"] = working[ach_col].apply(grade_to_numeric)
    working["GrowthScore"] = working[growth_col].apply(grade_to_numeric)
    working["CRScore"] = working[cr_col].apply(grade_to_numeric)

    # Deficits relative to A (4.0). Treat missing as 0 deficit.
    working["AchDef"] = np.maximum(0.0, 4.0 - working["AchScore"].fillna(4.0))
    working["GrowthDef"] = np.maximum(0.0, 4.0 - working["GrowthScore"].fillna(4.0))
    working["CRDef"] = np.maximum(0.0, 4.0 - working["CRScore"].fillna(4.0))

    # Econ 0-1, then scaled to 0-4 so it is comparable in magnitude
    econ_raw = working["Percent Economically Disadvantaged"].fillna(0.0) / 100.0
    working["Econ01"] = econ_raw.clip(0, 1)
    working["EconScaled"] = working["Econ01"] * 4.0

    # Normalize campus weights to sum to 1
    w_ach, w_growth, w_cr, w_econ = campus_weights
    w = np.array([w_ach, w_growth, w_cr, w_econ], dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.array([0.4, 0.35, 0.15, 0.10], dtype=float)
    w_ach, w_growth, w_cr, w_econ = w.tolist()

    # CampusNeed (0-~4 range)
    working["CampusNeed"] = (
        w_ach * working["AchDef"] +
        w_growth * working["GrowthDef"] +
        w_cr * working["CRDef"] +
        w_econ * working["EconScaled"]
    )

    # Impact (need * enrollment)
    enr = working["TOTAL ENROLLMENT"].fillna(0.0)
    working["CampusImpact"] = working["CampusNeed"] * enr

    # Flag high-need campuses
    working["HighNeedCampus"] = working["CampusNeed"] >= high_need_threshold

    # District aggregation
    group_cols = ["DISTRICT NAME", "County Name", "Local Region"]
    for c in group_cols:
        if c not in working.columns:
            working[c] = ""

    agg = working.groupby(group_cols, dropna=False).apply(
        lambda g: pd.Series({
            "TotalSchools": g.shape[0],
            "HighNeedSchoolCount": int(g["HighNeedCampus"].sum()),
            "ImpactedStudents": float(g.loc[g["HighNeedCampus"], "TOTAL ENROLLMENT"].fillna(0).sum()),
            "NeedAvgWeighted": float(g["CampusImpact"].sum() / (g["TOTAL ENROLLMENT"].fillna(0).sum() or 1.0)),
            "EconWeighted": float((g["Econ01"] * g["TOTAL ENROLLMENT"].fillna(0)).sum() / (g["TOTAL ENROLLMENT"].fillna(0).sum() or 1.0)),
            "TotalEnrollment": float(g["TOTAL ENROLLMENT"].fillna(0).sum())
        })
    ).reset_index()

    # Min-max scaling helpers (avoid divide by zero)
    def minmax(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        xmin, xmax = x.min(), x.max()
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax - xmin == 0:
            return pd.Series(0.0, index=x.index)
        return (x - xmin) / (xmax - xmin)

    agg["S_HighNeedSchoolCount"] = minmax(agg["HighNeedSchoolCount"])
    agg["S_ImpactedStudents"] = minmax(agg["ImpactedStudents"])
    agg["S_NeedAvgWeighted"] = minmax(agg["NeedAvgWeighted"])
    agg["S_EconWeighted"] = minmax(agg["EconWeighted"])

    return working, agg


def assign_district_index_and_grade(agg: pd.DataFrame, district_weights):
    w_count, w_students, w_need_avg, w_econ = district_weights
    w = np.array([w_count, w_students, w_need_avg, w_econ], dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.array([0.45, 0.25, 0.20, 0.10], dtype=float)
    w_count, w_students, w_need_avg, w_econ = w.tolist()

    agg = agg.copy()
    agg["PriorityIndex"] = (
        w_count * agg["S_HighNeedSchoolCount"] +
        w_students * agg["S_ImpactedStudents"] +
        w_need_avg * agg["S_NeedAvgWeighted"] +
        w_econ * agg["S_EconWeighted"]
    ).astype(float)

    # Grades by percentile bands: A (top 10%), B (next 20), C (middle 40), D (next 20), F (bottom 10)
    if agg.shape[0] > 0:
        q90 = agg["PriorityIndex"].quantile(0.9)
        q70 = agg["PriorityIndex"].quantile(0.7)
        q30 = agg["PriorityIndex"].quantile(0.3)
        q10 = agg["PriorityIndex"].quantile(0.1)

        def to_grade(x):
            if x >= q90: return "A"
            if x >= q70: return "B"
            if x >= q30: return "C"
            if x >= q10: return "D"
            return "F"

        agg["InterventionPriorityGrade"] = agg["PriorityIndex"].apply(to_grade)
    else:
        agg["InterventionPriorityGrade"] = ""

    # Rank (1 = highest need)
    agg = agg.sort_values("PriorityIndex", ascending=False)
    agg["Rank"] = np.arange(1, len(agg) + 1)

    return agg


def driver_text(d_row: pd.Series, w_tuple) -> list[str]:
    # Provide interpretable drivers based on weighted/scaled components
    w_count, w_students, w_need_avg, w_econ = w_tuple
    parts = []
    # Use actual (unscaled) metrics for phrasing where helpful
    if d_row.get("HighNeedSchoolCount", 0) >= 5:
        parts.append("Many campuses below A-level performance/growth (high count of high-need schools)")
    if d_row.get("ImpactedStudents", 0) >= 3000:
        parts.append("Large number of students in high-need campuses")
    if d_row.get("NeedAvgWeighted", 0) >= 1.5:
        parts.append("High average campus need severity")
    if d_row.get("EconWeighted", 0) >= 0.6:
        parts.append("High concentration of economically disadvantaged students")
    # Default if empty
    if not parts:
        parts.append("Moderate need across indicators")
    return parts[:3]


def recommendation_text(d_row: pd.Series) -> list[str]:
    recs = []
    # Map to actions by dominant indicators
    if d_row.get("NeedAvgWeighted", 0) >= 1.5:
        recs.append("- Implement in-school-day tutoring 3–5x/week targeting ELA/Math skill gaps; use diagnostic grouping and 1:2–1:3 ratios.")
    if d_row.get("HighNeedSchoolCount", 0) >= 5:
        recs.append("- Phase rollout across top high-need campuses first; centralize tutor recruitment and training for consistency.")
    if d_row.get("ImpactedStudents", 0) >= 1500:
        recs.append("- Stagger schedules and leverage paraprofessionals for scale; add progress monitoring every 2–3 weeks.")
    if d_row.get("EconWeighted", 0) >= 0.6:
        recs.append("- Pair tutoring with attendance/engagement supports; offer bilingual supports where applicable.")
    if not recs:
        recs.append("- Start with a pilot in 2–3 highest-need campuses; standardize diagnostics and progress checks; expand after 6–8 weeks.")
    return recs


# Sidebar controls
st.sidebar.header("Configuration")
include_charters = st.sidebar.checkbox("Include Charter Schools", value=True)

st.sidebar.subheader("Campus Need Weights")
w_ach = st.sidebar.slider("Achievement deficit weight", 0.0, 1.0, 0.40, 0.05)
w_growth = st.sidebar.slider("Growth deficit weight", 0.0, 1.0, 0.35, 0.05)
w_cr = st.sidebar.slider("College readiness deficit weight", 0.0, 1.0, 0.15, 0.05)
w_econ = st.sidebar.slider("Economic disadvantage weight", 0.0, 1.0, 0.10, 0.05)
campus_weights = (w_ach, w_growth, w_cr, w_econ)

threshold = st.sidebar.slider("High-Need Campus Threshold (0–4 scale)", 0.0, 4.0, 1.5, 0.1)
min_enrollment = st.sidebar.number_input("Minimum campus enrollment to include", min_value=0, max_value=5000, value=50, step=10)

st.sidebar.subheader("District Composite Weights")
dw_count = st.sidebar.slider("Weight: Count of high-need schools", 0.0, 1.0, 0.45, 0.05)
dw_students = st.sidebar.slider("Weight: Impacted students", 0.0, 1.0, 0.25, 0.05)
dw_need = st.sidebar.slider("Weight: Avg need severity", 0.0, 1.0, 0.20, 0.05)
dw_econ = st.sidebar.slider("Weight: Econ disadvantage", 0.0, 1.0, 0.10, 0.05)
district_weights = (dw_count, dw_students, dw_need, dw_econ)

# Filters
st.sidebar.subheader("Filters")
county_filter = st.sidebar.text_input("Filter by County Name (contains)", value="")
region_filter = st.sidebar.text_input("Filter by Local Region (contains)", value="")
grade_filter_opts = ["All", "A", "B", "C", "D", "F"]
grade_filter = st.sidebar.selectbox("Filter by Intervention Grade", grade_filter_opts, index=0)

# Load and compute
try:
    df_raw = load_data("texas_data.csv")
except Exception as e:
    st.error(f"Failed to load texas_data.csv: {e}")
    st.stop()

campus_df, district_df = compute_scoring(
    df_raw, campus_weights, threshold, min_enrollment, include_charters
)
district_df = assign_district_index_and_grade(district_df, district_weights)

# Apply text filters
df_view = district_df.copy()
if county_filter.strip():
    df_view = df_view[df_view["County Name"].str.contains(county_filter, case=False, na=False)]
if region_filter.strip():
    df_view = df_view[df_view["Local Region"].str.contains(region_filter, case=False, na=False)]
if grade_filter != "All":
    df_view = df_view[df_view["InterventionPriorityGrade"] == grade_filter]

st.title("Texas District High-Impact Tutoring (H.I.T.) Prioritization")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Rankings", "District Detail", "Map (Beta)", "Methodology"])

with tab1:
    st.subheader("District Rankings (Higher = Higher Intervention Priority)")
    if df_view.empty:
        st.info("No districts match current filters.")
    else:
        # Display table with key fields
        show_cols = [
            "Rank", "DISTRICT NAME", "County Name", "Local Region",
            "InterventionPriorityGrade", "PriorityIndex",
            "HighNeedSchoolCount", "ImpactedStudents", "NeedAvgWeighted",
            "EconWeighted", "TotalSchools", "TotalEnrollment"
        ]
        show_cols = [c for c in show_cols if c in df_view.columns]
        st.dataframe(
            df_view[show_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        # Download
        csv_bytes = df_view[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Rankings CSV",
            data=csv_bytes,
            file_name="district_rankings.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("District Detail")
    districts_sorted = district_df.sort_values("PriorityIndex", ascending=False)["DISTRICT NAME"].tolist()
    selected = st.selectbox("Select a district", options=districts_sorted)

    if selected:
        d = district_df[district_df["DISTRICT NAME"] == selected]
        if not d.empty:
            d_row = d.iloc[0]

            # KPIs
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Intervention Grade", d_row.get("InterventionPriorityGrade", ""))
            kpi_cols[1].metric("State Rank (Need)", int(d_row.get("Rank", np.nan)))
            kpi_cols[2].metric("High-Need Schools", int(d_row.get("HighNeedSchoolCount", 0)))
            kpi_cols[3].metric("Impacted Students", int(d_row.get("ImpactedStudents", 0)))

            # Breakdown chart (contributions)
            w = np.array(district_weights, dtype=float)
            if w.sum() > 0: w = w / w.sum()
            contribs = {
                "High-Need School Count": float(d_row.get("S_HighNeedSchoolCount", 0) * w[0]),
                "Impacted Students": float(d_row.get("S_ImpactedStudents", 0) * w[1]),
                "Avg Need Severity": float(d_row.get("S_NeedAvgWeighted", 0) * w[2]),
                "Econ Disadvantage": float(d_row.get("S_EconWeighted", 0) * w[3]),
            }
            fig = px.bar(x=list(contribs.keys()), y=list(contribs.values()),
                         labels={"x": "Component", "y": "Weighted Contribution"},
                         title="Composite Index Contribution by Component")
            st.plotly_chart(fig, use_container_width=True)

            # Drivers
            st.markdown("##### Key Drivers")
            for t in driver_text(d_row, w.tolist()):
                st.write(f"• {t}")

            # Recommendations
            st.markdown("##### Action Recommendations")
            for r in recommendation_text(d_row):
                st.write(r)

            # Top contributing campuses
            st.markdown("##### Top Contributing Campuses")
            c = campus_df[campus_df["DISTRICT NAME"] == selected].copy()
            if not c.empty:
                show_c_cols = [
                    "CAMPUS NAME", "CampusNeed", "CampusImpact", "TOTAL ENROLLMENT",
                    "AchDef", "GrowthDef", "CRDef", "Percent Economically Disadvantaged", "HighNeedCampus"
                ]
                existing_cols = [x for x in show_c_cols if x in c.columns]
                c = c.sort_values("CampusImpact", ascending=False)
                st.dataframe(c[existing_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

                # Export selected district detail
                st.download_button(
                    label="Download Selected District Campus Detail CSV",
                    data=c[existing_cols].to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected.replace(' ', '_').lower()}_campus_detail.csv",
                    mime="text/csv"
                )

            # Nearby districts (county / region peers)
            st.markdown("##### Nearby / Peer Districts (for coordinated outreach)")
            colA, colB = st.columns(2)
            peers_county = district_df[(district_df["County Name"] == d_row.get("County Name")) &
                                       (district_df["DISTRICT NAME"] != selected)] \
                                       .sort_values("PriorityIndex", ascending=False).head(10)
            peers_region = district_df[(district_df["Local Region"] == d_row.get("Local Region")) &
                                       (district_df["DISTRICT NAME"] != selected)] \
                                       .sort_values("PriorityIndex", ascending=False).head(10)
            if not peers_county.empty:
                colA.write(f"In County: {d_row.get('County Name')}")
                colA.dataframe(peers_county[["DISTRICT NAME", "InterventionPriorityGrade", "Rank",
                                             "HighNeedSchoolCount", "ImpactedStudents"]]
                               .reset_index(drop=True),
                               use_container_width=True, hide_index=True)
            else:
                colA.info("No county peers found (after filters).")

            if not peers_region.empty:
                colB.write(f"In Local Region: {d_row.get('Local Region')}")
                colB.dataframe(peers_region[["DISTRICT NAME", "InterventionPriorityGrade", "Rank",
                                             "HighNeedSchoolCount", "ImpactedStudents"]]
                               .reset_index(drop=True),
                               use_container_width=True, hide_index=True)
            else:
                colB.info("No region peers found (after filters).")

with tab3:
    st.subheader("Map (Beta)")
    st.info("This dataset does not include district geocoordinates or boundaries. "
            "The map below embeds the TEA ESC map for geographic context. "
            "Enhanced district-level mapping (with polygons/points) can be added if we incorporate external geospatial data.")
    try:
        st.components.v1.iframe(
            "https://tea.texas.gov/about-tea/other-services/education-service-centers/education-service-centers-map",
            height=550
        )
    except Exception:
        st.warning("Unable to embed TEA map. Please open the provided link in a browser.")

with tab4:
    st.subheader("Methodology")
    st.markdown("""
- Campus letter grades are mapped to numeric on a 4.3 scale (A+=4.3, A=4.0, A-=3.7, ..., F=0).
- Deficits are computed as max(0, 4.0 - score) for Achievement, Growth, and College Readiness.
- Percent Economically Disadvantaged is normalized to 0–1 and scaled to 0–4 for comparability.
- CampusNeed = weighted sum of [Achievement deficit, Growth deficit, College Readiness deficit, Econ-scaled].
- CampusImpact = CampusNeed × Enrollment.
- High-Need campuses are flagged when CampusNeed ≥ threshold (default 1.5 of 0–4 scale).
- District aggregation:
  - HighNeedSchoolCount = number of flagged campuses
  - ImpactedStudents = total enrollment across flagged campuses
  - NeedAvgWeighted = sum(CampusImpact) / sum(Enrollment)
  - EconWeighted = enrollment-weighted Econ (0–1)
- Each district component is min-max scaled across districts; composite index is a weighted sum of scaled components.
- Intervention Priority Grade bands (by PriorityIndex percentiles): A (top 10%), B (next 20%), C (middle 40%), D (next 20%), F (bottom 10%).
- All weights and the campus threshold are adjustable in the sidebar for transparency and what-if analysis.
    """)

    st.markdown("Data caveats:")
    st.markdown("""
- CSV includes campus-level rows; charter indicator is included and can be toggled.
- Geographic mapping is approximate in the MVP due to missing geolocation/boundaries.
- 'Local Region' in the CSV is used as a proxy grouping; ESC alignment can be added later.
    """)

    if st.checkbox("Show sample of raw loaded columns"):
        st.write(df_raw.head(10))

    st.caption("Built for prioritizing outreach where high-impact tutoring can make the largest difference. Adjust weights and threshold in the sidebar to explore scenarios.")
