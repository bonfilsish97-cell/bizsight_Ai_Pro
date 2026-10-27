"""
BizSight AI: Turn Data Into Decisions.
Stage 1: branding, upload, dataset preview, data-quality checks, column/KPI detection.
Run with:  streamlit run app.py
"""
import base64
import html
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# set_page_config must be the first Streamlit command
st.set_page_config(page_title="BizSight AI", page_icon="📊", layout="wide")

from modules.data_loader import (  # noqa: E402
    MAX_FILE_MB,
    DataLoadError,
    categorical_summary,
    classify_columns,
    get_excel_sheet_names,
    is_excel,
    load_dataset,
    numeric_summary,
    quality_report,
    validate_file,
)
from modules.kpi_engine import ROLES, available_kpis, detect_columns, suggest_charts  # noqa: E402

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

PAGES = ["🏠 Home", "📋 Dataset Overview", "⚠️ Data Quality", "🎯 KPI Detection"]

TYPE_LABELS = {
    "numeric": "🔢 Numeric",
    "categorical": "🏷️ Category",
    "date": "📅 Date",
    "identifier": "🔑 Possible identifier",
    "text": "📝 Long text",
    "empty": "⬜ Empty",
}

CSS = """
<style>
:root{--navy:#0B1F3A;--blue:#1D6FEB;--teal:#0FB5A6;--green:#22A06B;--orange:#F59E0B;--red:#DC2626;--light:#F4F6FA;}
.stApp{background:var(--light);}
#MainMenu, footer{visibility:hidden;}
.block-container{padding-top:2rem;max-width:1200px;}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--navy);}
[data-testid="stSidebar"] *{color:#E6EEF9;}
.side-brand{text-align:center;padding:.5rem 0 1rem 0;}
.side-logo{width:64px;height:64px;border-radius:14px;}
.side-logo-emoji{font-size:2.6rem;line-height:1;}
.side-title{font-size:1.4rem;font-weight:800;color:#fff;margin-top:.3rem;}
.side-tag{font-size:.8rem;color:#5EEAD4;}

/* Buttons and uploader */
.stButton>button{background:var(--blue);color:#fff;border:none;border-radius:10px;font-weight:600;}
.stButton>button:hover{background:#1558BF;color:#fff;}
[data-testid="stFileUploader"] section{border:2px dashed var(--blue);border-radius:14px;background:#fff;padding:1.2rem;}

/* Hero */
.hero{background:linear-gradient(135deg,#0B1F3A 0%,#12356b 60%,#1D6FEB 140%);border-radius:20px;padding:2.2rem;
display:flex;flex-wrap:wrap;gap:2rem;align-items:center;box-shadow:0 10px 30px rgba(11,31,58,.25);margin-bottom:1.5rem;}
.hero-text{flex:1 1 320px;}
.hero-kicker{color:#93C5FD;font-weight:600;letter-spacing:.04em;text-transform:uppercase;font-size:.8rem;}
.hero-title{color:#fff;font-size:2.3rem;font-weight:800;line-height:1.15;margin:.3rem 0;}
.hero-tagline{color:#5EEAD4;font-size:1.35rem;font-weight:700;margin-bottom:.6rem;}
.hero-desc{color:#D6E2F5;font-size:1rem;line-height:1.5;}
.hero-visual{flex:1 1 260px;text-align:center;}
.hero-visual img{max-width:100%;border-radius:14px;}
.mock{background:#fff;border-radius:16px;padding:1rem;box-shadow:0 8px 24px rgba(0,0,0,.25);}
.mock-row{display:flex;gap:.6rem;margin-bottom:.8rem;}
.mock-kpi{flex:1;background:var(--light);border-radius:10px;padding:.5rem;color:var(--navy);font-size:.8rem;text-align:center;}
.mock-kpi b{display:block;}
.mock-bars{display:flex;align-items:flex-end;gap:.5rem;height:110px;}
.mock-bars span{flex:1;background:linear-gradient(180deg,var(--teal),var(--blue));border-radius:6px 6px 0 0;}
@media (max-width:768px){.hero{padding:1.4rem;}.hero-title{font-size:1.7rem;}}

/* Cards */
.biz-card{background:#fff;border-radius:16px;padding:1rem 1.2rem;box-shadow:0 2px 10px rgba(11,31,58,.08);
border-top:4px solid var(--blue);margin-bottom:.8rem;}
.biz-card.accent-teal{border-top-color:var(--teal);}
.biz-card.accent-green{border-top-color:var(--green);}
.biz-card.accent-orange{border-top-color:var(--orange);}
.biz-card.accent-red{border-top-color:var(--red);}
.biz-card-icon{font-size:1.4rem;}
.biz-card-label{color:#5B6B82;font-size:.85rem;font-weight:600;}
.biz-card-value{color:var(--navy);font-size:1.7rem;font-weight:800;}
.biz-card-note{color:#5B6B82;font-size:.75rem;}

.step{background:#fff;border-radius:16px;padding:1rem 1.2rem;box-shadow:0 2px 10px rgba(11,31,58,.08);height:100%;}
.step b{color:var(--navy);}
.step p{color:#5B6B82;margin:.3rem 0 0 0;font-size:.9rem;}

.kpi-status{background:#fff;border-radius:14px;padding:.8rem 1rem;margin-bottom:.7rem;box-shadow:0 2px 8px rgba(11,31,58,.07);
border-left:6px solid var(--green);}
.kpi-status.off{border-left-color:#B8C2D1;opacity:.85;}
.kpi-status-head{color:var(--navy);display:flex;justify-content:space-between;gap:.5rem;align-items:center;}
.kpi-status-body{color:#5B6B82;font-size:.85rem;margin-top:.3rem;}
.badge{font-size:.72rem;font-weight:700;padding:.15rem .5rem;border-radius:999px;white-space:nowrap;}
.badge.ok{background:#DCFCE7;color:#166534;}
.badge.off{background:#E5E9F0;color:#4B5A70;}
</style>
"""

FALLBACK_HERO_VISUAL = (
    '<div class="mock">'
    '<div class="mock-row">'
    '<div class="mock-kpi">💰<b>Revenue</b></div>'
    '<div class="mock-kpi">📈<b>Growth</b></div>'
    '<div class="mock-kpi">🎯<b>Targets</b></div>'
    '</div>'
    '<div class="mock-bars">'
    '<span style="height:35%"></span><span style="height:55%"></span><span style="height:45%"></span>'
    '<span style="height:75%"></span><span style="height:65%"></span><span style="height:95%"></span>'
    '</div></div>'
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def esc(value) -> str:
    """Escape text before placing it inside HTML (dataset content is never trusted)."""
    return html.escape(str(value))


def image_data_uri(path: Path):
    """Return an embeddable image string, or None if the file is missing (so a fallback is used)."""
    if not path.exists():
        return None
    try:
        encoded = base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{encoded}"


def biz_card(icon, label, value, accent="blue", note="") -> str:
    note_html = f'<div class="biz-card-note">{esc(note)}</div>' if note else ""
    return (
        f'<div class="biz-card accent-{accent}"><div class="biz-card-icon">{icon}</div>'
        f'<div class="biz-card-label">{esc(label)}</div>'
        f'<div class="biz-card-value">{esc(value)}</div>{note_html}</div>'
    )


def severity(pct: float) -> str:
    if pct == 0:
        return "green"
    return "orange" if pct < 5 else "red"


def safe_preview(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """First n rows, with mixed-type text columns made display-safe (original data is untouched)."""
    preview = df.head(n).copy()
    for col in preview.columns:
        if preview[col].dtype == "object":
            preview[col] = preview[col].map(lambda v: v if pd.isna(v) else str(v))
    return preview


# --------------------------------------------------------------------------
# Session state (the uploaded data is kept in memory only, never saved to disk)
# --------------------------------------------------------------------------
def init_state():
    defaults = {
        "df": None, "file_name": None, "signature": None,
        "col_types": {}, "quality": None, "mapping": {}, "uploader_n": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_dataset(df: pd.DataFrame, file_name: str, signature: str):
    col_types = classify_columns(df)
    st.session_state.df = df
    st.session_state.file_name = file_name
    st.session_state.signature = signature
    st.session_state.col_types = col_types
    st.session_state.quality = quality_report(df, col_types)
    st.session_state.mapping = detect_columns(df, col_types)


def clear_dataset(reset_uploader: bool = False):
    for key in ("df", "file_name", "signature", "quality"):
        st.session_state[key] = None
    st.session_state.col_types = {}
    st.session_state.mapping = {}
    if reset_uploader:
        st.session_state.uploader_n += 1


def require_dataset() -> bool:
    if st.session_state.df is None:
        st.info("📂 No dataset loaded yet. Go to **🏠 Home** in the sidebar and upload a file first.")
        return False
    return True


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        logo_uri = image_data_uri(ASSETS_DIR / "logo.png")
        logo_html = f'<img class="side-logo" src="{logo_uri}" alt="BizSight AI logo">' if logo_uri \
            else '<div class="side-logo-emoji">📊</div>'
        st.markdown(
            f'<div class="side-brand">{logo_html}<div class="side-title">BizSight AI</div>'
            '<div class="side-tag">Turn Data Into Decisions.</div></div>',
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
        st.markdown("---")
        if st.session_state.df is not None:
            q = st.session_state.quality
            st.markdown(f"**📁 {esc(st.session_state.file_name)}**", unsafe_allow_html=True)
            st.caption(f"{q['rows']:,} rows × {q['columns']} columns")
            if st.button("🗑️ Remove dataset"):
                clear_dataset(reset_uploader=True)
                st.rerun()
        else:
            st.caption("No dataset loaded.")
        st.caption("Stage 1 · Your data stays in memory for this session only.")
    return page


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def render_hero():
    hero_uri = image_data_uri(ASSETS_DIR / "dashboard_hero.png")
    visual = f'<img src="{hero_uri}" alt="Business analytics dashboard illustration">' if hero_uri \
        else FALLBACK_HERO_VISUAL
    st.markdown(
        '<div class="hero"><div class="hero-text">'
        '<div class="hero-kicker">📊 BizSight AI</div>'
        '<div class="hero-title">Welcome to BizSight AI</div>'
        '<div class="hero-tagline">Turn Data Into Decisions.</div>'
        '<div class="hero-desc">Upload your business dataset and automatically discover the KPIs, '
        'trends, and insights that matter most.</div></div>'
        f'<div class="hero-visual">{visual}</div></div>',
        unsafe_allow_html=True,
    )


def handle_upload():
    uploaded = st.file_uploader(
        "Drag and drop a CSV, XLS, or XLSX file here, or click to browse",
        type=["csv", "xls", "xlsx"],
        key=f"uploader_{st.session_state.uploader_n}",
    )
    if uploaded is None:
        return
    try:
        file_bytes = uploaded.getvalue()
        validate_file(uploaded.name, len(file_bytes))

        sheet = None
        if is_excel(uploaded.name):
            sheets = get_excel_sheet_names(file_bytes, uploaded.name)
            if len(sheets) == 1:
                sheet = sheets[0]
            else:
                sheet = st.selectbox("📄 This workbook has several sheets. Choose one to analyse:",
                                     sheets, key=f"sheet_{uploaded.name}_{len(file_bytes)}")

        signature = f"{uploaded.name}|{len(file_bytes)}|{sheet}"
        if signature != st.session_state.signature:
            with st.spinner("Reading and analysing your data..."):
                df = load_dataset(file_bytes, uploaded.name, sheet)
                set_dataset(df, uploaded.name, signature)
    except DataLoadError as exc:
        clear_dataset()
        st.error(f"❌ {exc}")
    except Exception as exc:  # last safety net so the app never crashes on a bad file
        clear_dataset()
        st.error("❌ Something unexpected went wrong while reading this file. "
                 "Please check that it is a valid spreadsheet and try again.")
        with st.expander("Technical details"):
            st.code(str(exc))


def page_home():
    render_hero()
    st.markdown("### ⬆️ Upload your dataset")
    st.caption(f"Supported formats: CSV, XLS, XLSX · Maximum size: {MAX_FILE_MB} MB")
    handle_upload()

    if st.session_state.df is not None:
        q = st.session_state.quality
        st.success(
            f"✅ `{st.session_state.file_name}` loaded: {q['rows']:,} rows × {q['columns']} columns. "
            "Use the sidebar to explore **Dataset Overview**, **Data Quality**, and **KPI Detection**."
        )

    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    steps = [
        ("1️⃣ Upload", "Add a CSV or Excel file with your sales or business records."),
        ("2️⃣ Review", "Check the detected column types, data quality, and column roles. Correct them if needed."),
        ("3️⃣ Understand", "Get KPIs, charts, and plain-English insights in the next versions of BizSight AI."),
    ]
    for col, (title, text) in zip((c1, c2, c3), steps):
        col.markdown(f'<div class="step"><b>{title}</b><p>{text}</p></div>', unsafe_allow_html=True)

    with st.expander("🔒 Data handling and limitations"):
        st.markdown(
            "- Your file is read into memory for your current session. BizSight AI does not save it to disk or a database.\n"
            "- Uploaded files are only read as data. They are never run as programs, and their content is never treated as instructions.\n"
            "- This is a demonstration tool. It has no formal security controls, so **do not upload highly sensitive or confidential data**.\n"
            "- Automatic detection is a best guess. Always review the detected columns before trusting results."
        )


def page_overview():
    if not require_dataset():
        return
    df, col_types, q = st.session_state.df, st.session_state.col_types, st.session_state.quality
    counts = Counter(col_types.values())

    st.markdown("## 📋 Dataset Overview")
    row1 = st.columns(4)
    row1[0].markdown(biz_card("🧾", "Rows", f"{q['rows']:,}"), unsafe_allow_html=True)
    row1[1].markdown(biz_card("🧱", "Columns", f"{q['columns']}"), unsafe_allow_html=True)
    row1[2].markdown(biz_card("🔢", "Numeric columns", counts.get("numeric", 0), "teal"), unsafe_allow_html=True)
    row1[3].markdown(biz_card("📅", "Date columns", counts.get("date", 0), "teal"), unsafe_allow_html=True)
    row2 = st.columns(4)
    row2[0].markdown(biz_card("🏷️", "Category columns", counts.get("categorical", 0), "green"), unsafe_allow_html=True)
    row2[1].markdown(biz_card("🔑", "Possible identifiers", counts.get("identifier", 0), "green"), unsafe_allow_html=True)
    row2[2].markdown(biz_card("⚠️", "Missing cells", f"{q['missing_cells']:,}", severity(q["missing_pct"]),
                              f"{q['missing_pct']:.1f}% of all cells"), unsafe_allow_html=True)
    row2[3].markdown(biz_card("📑", "Duplicate rows", f"{q['duplicate_rows']:,}", severity(q["duplicate_pct"]),
                              f"{q['duplicate_pct']:.1f}% of rows"), unsafe_allow_html=True)

    st.markdown("### 👀 Data preview (first 50 rows)")
    st.dataframe(safe_preview(df))

    st.markdown("### 🧬 Column types")
    profile = q["column_profile"].copy()
    profile["Detected type"] = profile["Detected type"].map(TYPE_LABELS)
    st.dataframe(profile[["Column", "Detected type", "Stored as", "Non-missing", "Unique values"]])
    st.caption("Types are detected automatically from column names and values. Numbers or dates saved as text are recognised too.")

    st.markdown("### 📐 Descriptive statistics")
    num = numeric_summary(df, col_types)
    if num.empty:
        st.info("This dataset has no numeric columns, so number statistics are not shown.")
    else:
        st.markdown("**Numeric columns**")
        st.dataframe(num)
    cat = categorical_summary(df, col_types)
    if cat.empty:
        st.info("This dataset has no category or identifier columns.")
    else:
        st.markdown("**Category and identifier columns**")
        st.dataframe(cat)


def page_quality():
    if not require_dataset():
        return
    df, q = st.session_state.df, st.session_state.quality
    st.markdown("## ⚠️ Data Quality")
    st.caption("Nothing has been changed in your data. Cleaning options arrive in Stage 3, and you will see every change before it is applied.")

    n_invalid = int(q["invalid_values"]["Invalid values"].sum()) if not q["invalid_values"].empty else 0
    cards = st.columns(4)
    cards[0].markdown(biz_card("🕳️", "Missing cells", f"{q['missing_cells']:,}", severity(q["missing_pct"]),
                               f"{q['missing_pct']:.1f}% of {q['total_cells']:,} cells"), unsafe_allow_html=True)
    cards[1].markdown(biz_card("📑", "Duplicate rows", f"{q['duplicate_rows']:,}", severity(q["duplicate_pct"]),
                               f"{q['duplicate_pct']:.1f}% of rows"), unsafe_allow_html=True)
    cards[2].markdown(biz_card("⬜", "Empty columns", len(q["empty_columns"]),
                               "red" if q["empty_columns"] else "green"), unsafe_allow_html=True)
    cards[3].markdown(biz_card("🚫", "Invalid values", f"{n_invalid:,}",
                               "red" if n_invalid else "green"), unsafe_allow_html=True)

    st.markdown("### What we found")
    if q["missing_cells"] == 0:
        st.success("✅ No missing values found.")
    elif q["missing_pct"] < 5:
        st.warning(f"⚠️ {q['missing_cells']:,} cells ({q['missing_pct']:.1f}%) are empty. This is a small share, but check the columns below.")
    else:
        st.error(f"🚨 {q['missing_cells']:,} cells ({q['missing_pct']:.1f}%) are empty. Results based on these columns may be unreliable.")

    if q["duplicate_rows"] == 0:
        st.success("✅ No duplicate rows found.")
    else:
        st.warning(f"⚠️ {q['duplicate_rows']:,} rows are exact copies of an earlier row. They may inflate totals such as revenue.")
        with st.expander("Show the duplicated rows"):
            st.dataframe(safe_preview(df[df.duplicated(keep=False)], 100))

    if q["empty_columns"]:
        st.error("🚨 These columns contain no data at all: " + ", ".join(f"`{c}`" for c in q["empty_columns"]))
    if q["constant_columns"]:
        st.info("ℹ️ These columns have the same value in every row, so they cannot show differences: "
                + ", ".join(f"`{c}`" for c in q["constant_columns"]))
    if q["invalid_values"].empty:
        st.success("✅ No values were found that clash with their column's type.")
    else:
        st.error("🚨 Some values do not fit their column (for example, words inside a number column):")
        st.dataframe(q["invalid_values"])

    missing = q["missing_by_column"]
    missing = missing[missing["Missing"] > 0].head(15)
    if not missing.empty:
        fig = px.bar(missing, x="Column", y="Missing %", hover_data=["Missing"],
                     title="Missing values by column (%)", color_discrete_sequence=["#F59E0B"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Missing (%)",
                          xaxis_title="", margin=dict(t=60, l=10, r=10, b=10))
        st.plotly_chart(fig)

    with st.expander("📋 Full column report"):
        profile = q["column_profile"].copy()
        profile["Detected type"] = profile["Detected type"].map(TYPE_LABELS)
        st.dataframe(profile)


def page_kpi_detection():
    if not require_dataset():
        return
    df, col_types, sig = st.session_state.df, st.session_state.col_types, st.session_state.signature
    st.markdown("## 🎯 KPI Detection")
    st.write("BizSight AI guessed which column plays which business role. **Please review and correct any guesses.** "
             "KPIs below update instantly.")

    NONE = "— none —"
    options = [NONE] + list(df.columns)
    detected = st.session_state.mapping
    chosen = {}
    grid = st.columns(2)
    for i, (role, info) in enumerate(ROLES.items()):
        current = detected.get(role)
        index = options.index(current) if current in options else 0
        with grid[i % 2]:
            pick = st.selectbox(f"{info['icon']} {info['label']}", options, index=index,
                                key=f"map_{role}_{sig}", help=info["help"])
        chosen[role] = None if pick == NONE else pick
    st.session_state.mapping = chosen

    # Gentle warnings for questionable choices
    for role, col in chosen.items():
        if col and col_types.get(col) not in ROLES[role]["types"]:
            expected = " / ".join(sorted(ROLES[role]["types"]))
            st.warning(f"⚠️ {ROLES[role]['label']}: `{col}` looks like a **{col_types.get(col)}** column, "
                       f"but this role normally needs **{expected}**. Results may be wrong.")
    used = [c for c in chosen.values() if c]
    for col in sorted({c for c in used if used.count(c) > 1}):
        st.warning(f"⚠️ `{col}` is selected for more than one role. Double-check this.")

    kpis = available_kpis(chosen)
    n_ok = sum(1 for k in kpis if k["available"])
    st.markdown(f"### 📊 KPIs your data can support: {n_ok} of {len(kpis)}")
    left, right = st.columns(2)
    for i, k in enumerate(kpis):
        cls = "" if k["available"] else " off"
        badge_cls = "ok" if k["available"] else "off"
        badge = "✅ Available" if k["available"] else "🔒 Not available"
        body = f"<b>How it's calculated:</b> {esc(k['how'])}"
        if not k["available"]:
            body += f"<br><b>Needs:</b> {esc(k['needs'])}"
        card = (f'<div class="kpi-status{cls}"><div class="kpi-status-head"><span>{k["icon"]} <b>{esc(k["name"])}</b></span>'
                f'<span class="badge {badge_cls}">{badge}</span></div><div class="kpi-status-body">{body}</div></div>')
        (left if i % 2 == 0 else right).markdown(card, unsafe_allow_html=True)

    st.markdown("### 📈 Charts that make sense for this dataset")
    for chart in suggest_charts(chosen, col_types):
        if chart["available"]:
            st.markdown(f"- ✅ **{chart['name']}**")
        else:
            st.markdown(f"- 🔒 {chart['name']}: needs {chart['needs']}")
    st.caption("KPI values and charts are built in Stage 2. This page only shows what is possible with your data.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    page = render_sidebar()

    routes = {
        "🏠 Home": page_home,
        "📋 Dataset Overview": page_overview,
        "⚠️ Data Quality": page_quality,
        "🎯 KPI Detection": page_kpi_detection,
    }
    try:
        routes[page]()
    except Exception as exc:  # one broken page should never take down the whole dashboard
        st.error("😕 This section ran into a problem, but the rest of the app still works. "
                 "Try another section or re-upload your file.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")


main()
