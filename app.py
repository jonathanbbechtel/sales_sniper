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
st.sidebar.header("Settings")

include_charters = st.sidebar.checkbox(
    "Include charter schools?",
    value=True,
    help="When on, ranked results will include charter-operated schools."
)

threshold = st.sidebar.slider(
    "What counts as a high‑need school? (0–4)",
    0.0, 4.0, 1.5, 0.1,
    help="Schools at or above this score are considered high‑need. 1.5 is a typical cut‑off."
)

min_enrollment = st.sidebar.number_input(
    "Ignore very small schools (min students)",
    min_value=0, max_value=5000, value=50, step=10,
    help="Exclude very small schools that can distort rankings."
)

# Filters
st.sidebar.subheader("Filters")
county_filter = st.sidebar.text_input("County contains", value="", help="Type part of a county name to narrow the list.")
region_filter = st.sidebar.text_input("Region contains", value="", help="Type part of a region name to narrow the list.")
grade_filter_opts = ["All", "A", "B", "C", "D", "F"]
grade_filter = st.sidebar.selectbox("Show only districts with grade", grade_filter_opts, index=0)

with st.sidebar.expander("Advanced: Adjust scoring weights (optional)", expanded=False):
    st.markdown("Campus scoring (how we define a high‑need school)")
    w_ach = st.slider("Weight low test performance (achievement)", 0.0, 1.0, 0.40, 0.05,
                      help="Higher = districts with more low scores rank higher.")
    w_growth = st.slider("Weight slow academic growth", 0.0, 1.0, 0.35, 0.05,
                         help="Higher = more emphasis on year‑over‑year growth gaps.")
    w_cr = st.slider("Weight college readiness gaps", 0.0, 1.0, 0.15, 0.05,
                     help="Higher = more emphasis on readiness indicators.")
    w_econ = st.slider("Weight economic disadvantage", 0.0, 1.0, 0.10, 0.05,
                       help="Higher = more emphasis where more students are economically disadvantaged.")
    campus_weights = (w_ach, w_growth, w_cr, w_econ)

    st.markdown("How we rank districts (combine the pieces)")
    dw_count = st.slider("Count of high‑need schools", 0.0, 1.0, 0.45, 0.05,
                         help="Higher = districts with more high‑need schools move up.")
    dw_students = st.slider("Students in high‑need schools", 0.0, 1.0, 0.25, 0.05,
                            help="Higher = more weight on how many students are affected.")
    dw_need = st.slider("Average school need", 0.0, 1.0, 0.20, 0.05,
                        help="Higher = more weight on the severity of need.")
    dw_econ = st.slider("Economic disadvantage", 0.0, 1.0, 0.10, 0.05,
                        help="Higher = more weight on economic disadvantage.")
    district_weights = (dw_count, dw_students, dw_need, dw_econ)

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
tab1, tab2, tab3 = st.tabs(["Priority List", "District Profile", "FAQ"])

with tab1:
    st.subheader("Priority List")
    st.caption("Sorted by Priority Score (highest first).")
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
        # Friendly column names for display
        rename_map = {
            "Rank": "Statewide Rank",
            "DISTRICT NAME": "District",
            "County Name": "County",
            "Local Region": "Region",
            "InterventionPriorityGrade": "Priority Grade",
            "PriorityIndex": "Priority Score",
            "HighNeedSchoolCount": "High‑Need Schools",
            "ImpactedStudents": "Students in High‑Need Schools",
            "NeedAvgWeighted": "Average School Need",
            "EconWeighted": "Economic Disadvantage (weighted)",
            "TotalSchools": "Total Schools",
            "TotalEnrollment": "Total Enrollment",
        }
        df_disp = df_view[show_cols].rename(columns=rename_map).reset_index(drop=True)
        st.dataframe(df_disp, use_container_width=True, hide_index=True)

        # Download
        csv_bytes = df_view[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download list (CSV)",
            data=csv_bytes,
            file_name="district_rankings.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("District Profile")
    districts_sorted = district_df.sort_values("PriorityIndex", ascending=False)["DISTRICT NAME"].tolist()
    selected = st.selectbox("Select a district", options=districts_sorted)

    if selected:
        d = district_df[district_df["DISTRICT NAME"] == selected]
        if not d.empty:
            d_row = d.iloc[0]

            # KPIs
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Priority Grade", d_row.get("InterventionPriorityGrade", ""))
            kpi_cols[1].metric("Statewide Rank", int(d_row.get("Rank", np.nan)))
            kpi_cols[2].metric("High‑Need Schools", int(d_row.get("HighNeedSchoolCount", 0)))
            kpi_cols[3].metric("Students in High‑Need Schools", int(d_row.get("ImpactedStudents", 0)))

            # Breakdown chart (contributions)
            w = np.array(district_weights, dtype=float)
            if w.sum() > 0: w = w / w.sum()
            contribs = {
                "Number of high‑need schools": float(d_row.get("S_HighNeedSchoolCount", 0) * w[0]),
                "Students affected": float(d_row.get("S_ImpactedStudents", 0) * w[1]),
                "Average school need": float(d_row.get("S_NeedAvgWeighted", 0) * w[2]),
                "Economic disadvantage": float(d_row.get("S_EconWeighted", 0) * w[3]),
            }
            fig = px.bar(
                x=list(contribs.keys()),
                y=list(contribs.values()),
                labels={"x": "What matters", "y": "Relative contribution"},
                title="What drives this district’s ranking"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Higher bars mean this factor is contributing more to the district’s overall Priority Score.")

            # Drivers
            st.markdown("##### Key Drivers")
            for t in driver_text(d_row, w.tolist()):
                st.write(f"• {t}")

            # Recommendations
            st.markdown("##### Action Recommendations")
            for r in recommendation_text(d_row):
                st.write(r)

            # Top contributing campuses
            st.markdown("##### Top Contributing Schools")
            c = campus_df[campus_df["DISTRICT NAME"] == selected].copy()
            if not c.empty:
                show_c_cols = [
                    "CAMPUS NAME", "CampusNeed", "CampusImpact", "TOTAL ENROLLMENT",
                    "AchDef", "GrowthDef", "CRDef", "Percent Economically Disadvantaged", "HighNeedCampus"
                ]
                existing_cols = [x for x in show_c_cols if x in c.columns]
                c = c.sort_values("CampusImpact", ascending=False)
                c_disp = c[existing_cols].rename(columns={
                    "CAMPUS NAME": "School",
                    "CampusNeed": "School Need Score",
                    "CampusImpact": "Impact (Need × Students)",
                    "TOTAL ENROLLMENT": "Enrollment",
                    "AchDef": "Achievement gap",
                    "GrowthDef": "Growth gap",
                    "CRDef": "College readiness gap",
                    "Percent Economically Disadvantaged": "Economically Disadvantaged (%)",
                    "HighNeedCampus": "High‑Need?"
                }).reset_index(drop=True)
                st.dataframe(c_disp, use_container_width=True, hide_index=True)

                # Export selected district detail
                st.download_button(
                    label="Download campus details (CSV)",
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
    st.subheader("FAQ")

    with st.expander("What does this app do?", expanded=True):
        st.markdown("It helps you quickly find Texas school districts where high‑impact tutoring can help the most. Districts are ranked from highest to lowest priority based on the number of high‑need schools, how many students are affected, how severe the needs are, and the level of economic disadvantage.")

    with st.expander("How do I use it?"):
        st.markdown("""
- Start on the Priority List tab. The top rows show where tutoring could have the biggest impact.
- Use the filters on the left to narrow by county, region, or grade.
- Open a District Profile to see key drivers and recommended actions.
- Optional: Open “Advanced: Adjust scoring weights” to change how the score is calculated.
        """)

    with st.expander("What is a “high‑need” school?"):
        st.markdown("A school that scores above your chosen cut‑off (default 1.5 on a 0–4 scale) based on a mix of lower performance, slower growth, college readiness gaps, and economic disadvantage.")

    with st.expander("What do “Priority Score” and “Priority Grade” mean?"):
        st.markdown("""
- Priority Score: A 0–1 score used to sort districts. Higher = higher priority.
- Priority Grade: A simple A–F label based on percentiles of the Priority Score:
  - A = highest priority (top 10%)
  - B = next 20%
  - C = middle 40%
  - D = next 20%
  - F = lowest priority (bottom 10%)
        """)

    with st.expander("What affects the Priority Score?"):
        st.markdown("""
Four ingredients:
1) Number of high‑need schools
2) Students in high‑need schools
3) Average school need (how severe the needs are)
4) Economic disadvantage
You can change their importance in Advanced settings.
        """)

    with st.expander("Where do the data and grades come from?"):
        st.markdown("Campus letter grades (A–F) are converted to numbers. We look at achievement, growth, and college‑readiness, plus the percent of students who are economically disadvantaged. We never show student‑level data—only school/district totals.")

    with st.expander("Why does the ranking change when I move sliders?"):
        st.markdown("Because you’re telling the app which factors matter more. For example, if you increase “Students in high‑need schools,” districts with more affected students will move up.")

    with st.expander("What settings should I start with?"):
        st.markdown("""
- Leave Advanced settings as‑is for a balanced view.
- If you want a shorter list, raise “What counts as a high‑need school?”.
- If very small schools appear, raise “Ignore very small schools.”
        """)

    with st.expander("Can I download the results?"):
        st.markdown("Yes. Use the “Download list (CSV)” button on the Priority List. In a District Profile, you can also download campus‑level details.")

    with st.expander("Definitions (quick reference)"):
        st.markdown("""
- Priority Score: A 0–1 score used to sort districts. Higher = higher priority.
- Priority Grade: A simple A–F label based on the Priority Score.
- High‑need school: A school above the chosen cut‑off on the 0–4 need scale.
- Average school need: Average severity of need across a district’s schools.
- Students in high‑need schools: Total students enrolled in high‑need schools.
        """)

    with st.expander("Advanced / Data"):
        if st.checkbox("Show sample of raw loaded columns"):
            st.write(df_raw.head(10))

    st.caption("Built to prioritize outreach where high‑impact tutoring can make the largest difference. Adjust settings in the sidebar to explore scenarios.")
