from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="نظام التنبؤ بازدحام المعتمرين",
    page_icon="🕋",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# Constants
# =====================================================

MODEL_PATHS = [
    Path("models/growth_rate_xgboost_model.pkl"),
    Path("growth_rate_xgboost_model.pkl")
]

DATE_COL = "Gregorian_Date"
MONTH_COL = "Hijri_Month"
TARGET_COL = "المعتمرين"

HIJRI_MONTHS = [
    "محرم", "صفر", "ربيع الأول", "ربيع الآخر",
    "جماد الأول", "جماد الآخر",
    "رجب", "شعبان", "رمضان", "شوال",
    "ذو القعدة", "ذو الحجة"
]

NATIONALITY_OPTIONS = ["سعودي", "غير سعودي"]

LEVEL_COLORS = {
    "منخفض": "#1f8a52",
    "متوسط": "#c28a20",
    "مرتفع": "#c94b45"
}

LEVEL_BG = {
    "منخفض": "#eef8f1",
    "متوسط": "#fff8e8",
    "مرتفع": "#fff1f0"
}

LEVEL_BORDER = {
    "منخفض": "#b9dec6",
    "متوسط": "#ead29b",
    "مرتفع": "#e7b7b2"
}


# =====================================================
# Helper Functions
# =====================================================

def H(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def esc(x):
    if pd.isna(x):
        return ""
    return html.escape(str(x))


def format_number(value):
    try:
        if pd.isna(value):
            return "غير متاح"
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def format_seasonal_count(value):
    try:
        if pd.isna(value):
            return "لا يوجد"
        value = float(value)
        if value <= 0:
            return "لا يوجد"
        return f"{int(round(value)):,}"
    except Exception:
        return "لا يوجد"


def normalize_level(level):
    text = str(level).strip()

    mapping = {
        "Low": "منخفض",
        "Medium": "متوسط",
        "High": "مرتفع",
        "Critical": "مرتفع",
        "low": "منخفض",
        "medium": "متوسط",
        "high": "مرتفع",
        "critical": "مرتفع",
        "منخفض": "منخفض",
        "متوسط": "متوسط",
        "مرتفع": "مرتفع",
        "حرج": "مرتفع",
        "عالي": "مرتفع",
        "مرتفع جدًا": "مرتفع",
    }

    return mapping.get(text, text)


def english_level(level):
    level = normalize_level(level)

    return {
        "منخفض": "Low",
        "متوسط": "Medium",
        "مرتفع": "High"
    }.get(level, "Medium")


def is_hajj_related_month(month_name):
    return str(month_name).strip() in ["ذو القعدة", "ذو الحجة"]


def get_existing_model_path():
    for path in MODEL_PATHS:
        if path.exists():
            return path
    return None


def extract_hijri_day(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    nums = []
    current = ""

    for ch in text:
        if ch.isdigit():
            current += ch
        else:
            if current:
                nums.append(int(current))
                current = ""

    if current:
        nums.append(int(current))

    if not nums:
        return np.nan

    if nums[0] >= 1400 and len(nums) >= 3:
        return nums[2]

    if nums[-1] >= 1400 and len(nums) >= 3:
        return nums[0]

    return nums[0]


def classify_by_quantiles(series):
    series = pd.to_numeric(series, errors="coerce")

    q1 = series.quantile(0.33)
    q2 = series.quantile(0.66)

    def classify(x):
        if pd.isna(x):
            return "متوسط"
        if x <= q1:
            return "منخفض"
        elif x <= q2:
            return "متوسط"
        else:
            return "مرتفع"

    return series.apply(classify)


def normalize_columns(df):
    df = df.copy()

    rename_map = {
        "Hajj_Feature": "Hajj",
        "Tawaf_Ifadah_Feature": "Tawaf_Ifadah",
        "Tawaf_Ifadah_Featureh": "Tawaf_Ifadah",
        "Umrah_Count": "المعتمرين",
        "Visitors": "المعتمرين",
        "Predicted": "Prediction",
        "Prediction_Count": "Prediction",
        "predicted_visitors": "Prediction",
        "Gregorian Date": "Gregorian_Date",
        "Hijri Month": "Hijri_Month",
        "Hijri Day": "Hijri_Day"
    }

    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})


def add_display_cols(df):
    df = normalize_columns(df)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    else:
        df[DATE_COL] = pd.NaT

    if "Hijri_Date" not in df.columns:
        df["Hijri_Date"] = ""

    if "Hijri_Day" not in df.columns:
        df["Hijri_Day"] = df["Hijri_Date"].apply(extract_hijri_day)

    df["Hijri_Day_Num"] = pd.to_numeric(df["Hijri_Day"], errors="coerce")

    if df["Hijri_Day_Num"].isna().all():
        df["Hijri_Day_Num"] = df["Hijri_Date"].apply(extract_hijri_day)

    if "Weekday_AR" not in df.columns:
        weekday_map = {
            0: "الاثنين",
            1: "الثلاثاء",
            2: "الأربعاء",
            3: "الخميس",
            4: "الجمعة",
            5: "السبت",
            6: "الأحد",
        }

        if DATE_COL in df.columns:
            df["Weekday_AR"] = df[DATE_COL].dt.dayofweek.map(weekday_map)
        else:
            df["Weekday_AR"] = "غير محدد"

    if "AvgTemp_C" not in df.columns:
        df["AvgTemp_C"] = np.nan

    if "Hajj" not in df.columns:
        df["Hajj"] = 0

    if "Tawaf_Ifadah" not in df.columns:
        df["Tawaf_Ifadah"] = 0

    df["Hajj"] = pd.to_numeric(df["Hajj"], errors="coerce").fillna(0)
    df["Tawaf_Ifadah"] = pd.to_numeric(df["Tawaf_Ifadah"], errors="coerce").fillna(0)

    if "Prediction" not in df.columns:
        if TARGET_COL in df.columns:
            df["Prediction"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        else:
            st.error("ملف المودل لا يحتوي على عمود Prediction أو عمود المعتمرين.")
            st.stop()

    df["Prediction"] = pd.to_numeric(df["Prediction"], errors="coerce")

    if "Crowding_Level" not in df.columns:
        df["Crowding_Level"] = classify_by_quantiles(df["Prediction"])
    else:
        df["Crowding_Level"] = df["Crowding_Level"].apply(normalize_level)

        valid = ["منخفض", "متوسط", "مرتفع"]
        invalid_mask = ~df["Crowding_Level"].isin(valid)

        if invalid_mask.any():
            fallback = classify_by_quantiles(df["Prediction"])
            df.loc[invalid_mask, "Crowding_Level"] = fallback.loc[invalid_mask]

    return df


# =====================================================
# Load Model Results
# =====================================================

@st.cache_data
def load_data():
    model_path = get_existing_model_path()

    if model_path is None:
        st.error(
            "لم يتم العثور على ملف المودل. "
            "ارفع ملف growth_rate_xgboost_model.pkl داخل مجلد models."
        )
        st.stop()

    try:
        package = joblib.load(model_path)
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل ملف المودل: {e}")
        st.stop()

    if isinstance(package, pd.DataFrame):
        return add_display_cols(package)

    if isinstance(package, dict):
        if "test_df" in package:
            return add_display_cols(package["test_df"])

        if "results_df" in package:
            return add_display_cols(package["results_df"])

        if "df" in package:
            return add_display_cols(package["df"])

    st.error(
        "ملف المودل يجب أن يحتوي على DataFrame داخل test_df أو results_df أو df "
        "ويحتوي على عمود Prediction."
    )
    st.stop()


# =====================================================
# Logic
# =====================================================

def get_7_days(df, month, day):
    start_day = int(day)
    end_day = min(start_day + 6, 30)

    out = df[
        (df[MONTH_COL].astype(str).str.strip() == str(month).strip()) &
        (df["Hijri_Day_Num"] >= start_day) &
        (df["Hijri_Day_Num"] <= end_day)
    ].copy()

    out = out.sort_values("Hijri_Day_Num").reset_index(drop=True)

    out["Local_Crowding_Level"] = classify_by_quantiles(out["Prediction"])
    out["Local_Color"] = out["Local_Crowding_Level"].map(LEVEL_COLORS)

    return out


def get_best_day(df7, current_day):
    alt = df7[df7["Hijri_Day_Num"] != int(current_day)].copy()

    if alt.empty:
        return None

    return alt.sort_values("Prediction").iloc[0]


def get_decision(level):
    level = normalize_level(level)

    if level == "منخفض":
        return "مناسب"
    elif level == "مرتفع":
        return "يفضل اختيار يوم آخر"
    else:
        return "مناسب مع الحذر"


def get_reason(decision):
    if decision == "مناسب":
        return "الازدحام المتوقع منخفض، لذلك يعتبر اليوم مناسبًا لأداء العمرة."

    if decision == "يفضل اختيار يوم آخر":
        return "الازدحام المتوقع مرتفع مقارنةً بالأيام القريبة، لذلك يفضل اختيار يوم أقل ازدحامًا."

    return "الازدحام المتوقع ضمن المستوى المتوسط، لذلك يمكن أداء العمرة مع الحذر واختيار الوقت المناسب."


def prepare_display_table(df):
    display_df = df.copy()

    display_df = display_df[
        ["Hijri_Date", "Weekday_AR", "Prediction", "Local_Crowding_Level"]
    ].rename(columns={
        "Hijri_Date": "التاريخ الهجري",
        "Weekday_AR": "اليوم",
        "Prediction": "المعتمرون المتوقعون",
        "Local_Crowding_Level": "مستوى الازدحام",
    })

    display_df["المعتمرون المتوقعون"] = display_df["المعتمرون المتوقعون"].apply(format_number)

    return display_df


# =====================================================
# Chart
# =====================================================

def build_chart(df7):
    cdf = df7.copy().reset_index(drop=True)
    cdf["Prediction"] = pd.to_numeric(cdf["Prediction"], errors="coerce")

    cdf["x_label"] = (
        cdf["Weekday_AR"].astype(str) +
        "<br>" +
        cdf["Hijri_Day_Num"].astype(int).astype(str) +
        "-" +
        cdf[MONTH_COL].astype(str)
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cdf["x_label"],
        y=cdf["Prediction"],
        mode="lines",
        line=dict(
            color="#c69a35",
            width=3,
            shape="spline"
        ),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=cdf["x_label"],
        y=cdf["Prediction"],
        mode="markers+text",
        marker=dict(
            size=10,
            color=cdf["Local_Color"],
            line=dict(color="white", width=2)
        ),
        text=cdf["Prediction"].apply(
            lambda x: f"{int(round(float(x))):,}" if not pd.isna(x) else ""
        ),
        textposition="top center",
        hovertemplate="<b>%{x}</b><br>المعتمرون المتوقعون: %{y:,.0f}<extra></extra>",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=8, color=LEVEL_COLORS["منخفض"]),
        name="منخفض"
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=8, color=LEVEL_COLORS["متوسط"]),
        name="متوسط"
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=8, color=LEVEL_COLORS["مرتفع"]),
        name="مرتفع"
    ))

    q1 = cdf["Prediction"].quantile(0.33)
    q2 = cdf["Prediction"].quantile(0.66)

    fig.add_hline(
        y=q1,
        line_dash="dot",
        line_color="rgba(32,139,85,0.25)",
        line_width=1
    )

    fig.add_hline(
        y=q2,
        line_dash="dot",
        line_color="rgba(194,138,32,0.25)",
        line_width=1
    )

    y_min = cdf["Prediction"].min()
    y_max = cdf["Prediction"].max()
    pad = max((y_max - y_min) * 0.22, 400)

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffdfa",
        font=dict(family="Cairo", size=11, color="#064b3b"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0.01,
            title="مستوى الازدحام",
            font=dict(size=10)
        ),
        xaxis=dict(
            showgrid=False,
            title=""
        ),
        yaxis=dict(
            range=[max(0, y_min - pad), y_max + pad],
            showgrid=True,
            gridcolor="rgba(120,100,60,0.10)",
            tickformat=",",
            title=dict(text="عدد المعتمرين", font=dict(size=10))
        )
    )

    return fig


# =====================================================
# UI Helpers
# =====================================================

def top_card(title, value, subtitle, icon, level=None):
    if level is None:
        color = "#064b3b"
        bg = "rgba(255,255,255,0.90)"
        border = "#e5d4aa"
        badge_bg = "#eef3ea"
        badge_text = "#0a6b52"
    else:
        level = normalize_level(level)
        color = LEVEL_COLORS.get(level, "#064b3b")
        bg = LEVEL_BG.get(level, "#ffffff")
        border = LEVEL_BORDER.get(level, "#e5d4aa")
        badge_bg = LEVEL_BG.get(level, "#eef3ea")
        badge_text = color

    H(f"""
    <div class="top-card" style="background:{bg}; border-color:{border};">
        <div class="top-icon">{esc(icon)}</div>
        <div class="top-title">{esc(title)}</div>
        <div class="top-value" style="color:{color};">{esc(value)}</div>
        <div class="top-badge" style="background:{badge_bg}; color:{badge_text};">{esc(subtitle)}</div>
    </div>
    """)


def info_pill(text):
    H(f"""
    <div class="small-pill">{esc(text)}</div>
    """)


# =====================================================
# CSS
# =====================================================

H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&family=Noto+Kufi+Arabic:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', 'Noto Kufi Arabic', sans-serif !important;
    direction: rtl;
}

.stApp {
    background:
      radial-gradient(circle at 8% 12%, rgba(212,177,97,0.07), transparent 18%),
      radial-gradient(circle at 92% 10%, rgba(7,86,68,0.06), transparent 18%),
      linear-gradient(180deg, #fbf8f1 0%, #f5eee1 100%);
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    max-width: 1260px !important;
    padding-top: 0.45rem !important;
    padding-bottom: 1.0rem !important;
}

/* Sidebar - light compact */
section[data-testid="stSidebar"] {
    width: 190px !important;
    min-width: 190px !important;
    max-width: 190px !important;
    background:
      radial-gradient(circle at 50% 10%, rgba(212,177,97,0.10), transparent 24%),
      linear-gradient(180deg, #fbf8f1 0%, #f3ecdd 100%) !important;
    border-left: 1px solid #e2cf9d !important;
    box-shadow: -4px 0 18px rgba(105,84,35,0.05);
}

section[data-testid="stSidebar"] > div {
    width: 190px !important;
    min-width: 190px !important;
    max-width: 190px !important;
    padding-top: 0.9rem !important;
}

.sidebar-wrap {
    text-align:center;
    padding: 18px 8px 14px 8px;
}

.sidebar-title {
    color:#064b3b;
    font-size: 16px;
    font-weight: 900;
    margin-top: 4px;
}

.sidebar-sub {
    color:#9b6b16;
    font-size: 9px;
    font-weight:700;
    margin-top: 5px;
}

.sidebar-line {
    height:1px;
    background: linear-gradient(90deg, transparent, #d6bd7f, transparent);
    margin: 14px 24px 18px 24px;
}

section[data-testid="stSidebar"] .stButton {
    display: flex !important;
    justify-content: center !important;
}

/* Sidebar text without boxes */
section[data-testid="stSidebar"] .stButton button {
    width: fit-content !important;
    min-width: 0 !important;
    max-width: none !important;
    padding: 0 4px !important;
    background: transparent !important;
    color: #064b3b !important;
    border: none !important;
    text-align: right !important;
    font-weight: 800 !important;
    font-size: 13px !important;
    border-radius: 0 !important;
    height: 34px !important;
    margin-bottom: 14px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background: transparent !important;
    color: #9b6b16 !important;
    border: none !important;
    text-decoration: underline;
}

.stButton button {
    background: linear-gradient(135deg, #0a6b52, #064b3b) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    height: 40px !important;
    font-weight: 800 !important;
    font-size: 13px !important;
}

.main-header {
    background: linear-gradient(90deg, rgba(255,255,255,0.93), rgba(255,250,241,0.96));
    border: 1px solid #e5d4aa;
    border-radius: 20px;
    padding: 12px 22px;
    min-height: 84px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 7px 16px rgba(105,84,35,0.04);
    margin-bottom: 10px;
}

.header-main-title {
    color: #064b3b;
    font-family: 'Noto Kufi Arabic', 'Cairo', sans-serif;
    font-size: 22px;
    font-weight: 800;
    text-align: center;
}

.header-subtitle {
    color: #c08a19;
    font-size: 11px;
    font-weight: 800;
    text-align: center;
    margin-top: 4px;
}

.header-decor {
    width: 220px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #d3a33c, transparent);
    margin: 6px auto 0 auto;
}

.header-logo-box {
    width: 50px;
    height: 50px;
    border-radius: 16px;
    background: #f8ebcd;
    border: 1px solid #dfc98f;
    display: grid;
    place-items: center;
    font-size: 24px;
}

.hero-box {
    background: rgba(255,255,255,0.82);
    border: 1px solid #e5d4aa;
    border-radius: 20px;
    padding: 22px;
    text-align:center;
    margin-bottom: 12px;
}

.hero-icon {
    width: 56px;
    height: 56px;
    margin:auto;
    border-radius:50%;
    background: rgba(213,170,72,0.12);
    border:1px solid rgba(213,170,72,0.33);
    display:grid;
    place-items:center;
    font-size: 26px;
}

.hero-title {
    color:#064b3b;
    font-family:'Noto Kufi Arabic','Cairo',sans-serif;
    font-size:20px;
    font-weight:800;
    margin-top:8px;
}

.hero-sub {
    color:#be8919;
    font-size:12px;
    font-weight:800;
    margin-top:6px;
}

.hero-text {
    color:#51442a;
    font-size:12px;
    line-height:1.7;
    font-weight:600;
    margin-top:8px;
}

.top-card {
    background: rgba(255,255,255,0.90);
    border: 1px solid #e5d4aa;
    border-radius: 18px;
    padding: 12px 12px;
    min-height: 112px;
    box-shadow: 0 6px 14px rgba(105,84,35,0.04);
    text-align: center;
}

.top-icon {
    width: 42px;
    height: 42px;
    margin: 0 auto 5px auto;
    border-radius: 14px;
    display:grid;
    place-items:center;
    font-size:18px;
    background:#f6f2e9;
    border:1px solid #ead9b2;
}

.top-title {
    color:#3d2f16;
    font-size:11px;
    font-weight:800;
    margin-top:3px;
}

.top-value {
    font-family:'Noto Kufi Arabic','Cairo',sans-serif;
    font-size:21px;
    font-weight:800;
    line-height:1.2;
    margin-top:5px;
}

.top-badge {
    width:fit-content;
    margin: 6px auto 0 auto;
    padding:4px 10px;
    border-radius:999px;
    font-size:10px;
    font-weight:800;
}

.reco-box {
    background: rgba(255,255,255,0.78);
    border: 1px solid #e3d0a0;
    border-radius: 18px;
    padding: 12px 16px;
    box-shadow: 0 5px 12px rgba(105,84,35,0.03);
}

.reco-label {
    text-align:center;
    color:#5a4b2d;
    font-size:10px;
    font-weight:800;
}

.reco-title {
    text-align:center;
    font-family:'Noto Kufi Arabic','Cairo',sans-serif;
    font-size:18px;
    font-weight:800;
    margin-top:2px;
}

.reco-line {
    width:100px;
    height:2px;
    background: linear-gradient(90deg, transparent, #d0a347, transparent);
    margin:5px auto 6px auto;
}

.reco-text {
    text-align:center;
    color:#372f20;
    font-size:13px;
    font-weight:700;
    line-height:1.75;
    max-width:920px;
    margin:auto;
}

.small-pill {
    background: rgba(255,250,242,0.92);
    border: 1px solid #e7d7af;
    color:#84621c;
    border-radius: 999px;
    padding: 5px 11px;
    min-height: 28px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:10px;
    font-weight:800;
}

.section-box {
    background: rgba(255,255,255,0.70);
    border: 1px solid rgba(180,155,96,0.28);
    border-radius: 16px;
    padding: 11px 12px;
    box-shadow: 0 4px 10px rgba(105,84,35,0.03);
    height: 100%;
}

.section-title {
    text-align:center;
    color:#064b3b;
    font-family:'Noto Kufi Arabic','Cairo',sans-serif;
    font-size:14px;
    font-weight:800;
}

.section-line {
    width:68px;
    height:2px;
    background: linear-gradient(90deg, transparent, #d0a347, transparent);
    margin: 4px auto 8px auto;
}

.best-panel {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(180,155,96,0.30);
    border-radius: 16px;
    padding: 11px 12px;
    box-shadow: 0 4px 10px rgba(105,84,35,0.03);
    height: 100%;
}

.best-title {
    color:#064b3b;
    font-family:'Noto Kufi Arabic','Cairo',sans-serif;
    font-size:14px;
    font-weight:800;
    text-align:center;
}

.best-sub {
    color:#5a4d30;
    font-size:10px;
    font-weight:600;
    text-align:center;
    margin-top:5px;
    line-height:1.5;
}

.best-result-card {
    background: rgba(255,255,255,0.88);
    border:1px solid #ead8ae;
    border-radius:14px;
    padding:10px 12px;
    text-align:center;
    margin-top:10px;
}

.best-result-label {
    color:#6e5a2b;
    font-size:10px;
    font-weight:800;
}

.best-result-day {
    color:#064b3b;
    font-family:'Noto Kufi Arabic','Cairo',sans-serif;
    font-size:18px;
    font-weight:800;
    margin-top:4px;
}

.best-result-date {
    color:#3e3526;
    font-size:10px;
    font-weight:700;
    margin-top:4px;
}

.best-result-badge {
    width:fit-content;
    margin:7px auto 0 auto;
    padding:4px 11px;
    border-radius:999px;
    font-weight:800;
    font-size:10px;
}

.best-result-number {
    color:#7a6a44;
    font-size:10px;
    font-weight:700;
    margin-top:6px;
}

div[data-testid="column"]:has(.best-panel) .stButton button {
    width: 100% !important;
    height: 36px !important;
    border-radius: 12px !important;
    margin-top: -2px !important;
    box-shadow: 0 4px 10px rgba(6,75,59,0.12) !important;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.82) !important;
    border:1px solid #e2cf9d !important;
    border-radius: 20px !important;
    padding: 22px 28px !important;
    box-shadow: 0 7px 16px rgba(105,84,35,0.04);
}

.form-title {
    text-align:center;
    color:#064b3b;
    font-size:22px;
    font-weight:800;
    font-family:'Noto Kufi Arabic','Cairo',sans-serif;
}

.form-sub {
    text-align:center;
    color:#bf8d21;
    font-size:12px;
    font-weight:800;
    margin-top:5px;
}

.form-line {
    width:220px;
    height:2px;
    background:linear-gradient(90deg, transparent, #d0a347, transparent);
    margin:10px auto 17px auto;
}

.form-section {
    text-align:center;
    color:#064b3b;
    font-size:18px;
    font-weight:800;
    margin-bottom:16px;
}

.stTextInput input, .stSelectbox > div > div {
    background:#fffdfa !important;
    border:1px solid #d9c28a !important;
    border-radius:12px !important;
    min-height:40px !important;
}

div[data-testid="stHorizontalBlock"] {
    gap: 0.7rem !important;
}
</style>
""")


# =====================================================
# UI Components
# =====================================================

def show_header():
    H("""
    <div class="main-header">
        <div class="header-logo-box">🕋</div>
        <div style="flex:1;">
            <div class="header-main-title">نظام التنبؤ بمستويات ازدحام المعتمرين</div>
            <div class="header-subtitle">المسجد الحرام الشريف</div>
            <div class="header-decor"></div>
        </div>
    </div>
    """)


# =====================================================
# Session State
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "entered" not in st.session_state:
    st.session_state.entered = False

if "show_best_day" not in st.session_state:
    st.session_state.show_best_day = False


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:
    H("""
    <div class="sidebar-wrap">
        <div class="sidebar-title">التنبؤ</div>
        <div class="sidebar-sub">بازدحام المعتمرين</div>
        <div class="sidebar-line"></div>
    </div>
    """)

    if st.button("الرئيسية 🏠"):
        st.session_state.page = "home"
        st.rerun()

    if st.button("إدخال البيانات 📝"):
        st.session_state.page = "input"
        st.rerun()

    if st.button("لوحة النتائج 📈"):
        st.session_state.page = "dashboard"
        st.rerun()


# =====================================================
# Home Page
# =====================================================

def home_page():
    show_header()

    H("""
    <div class="hero-box">
        <div class="hero-icon">🕋</div>
        <div class="hero-title">مرحبًا بك في نظام التنبؤ بازدحام المعتمرين</div>
        <div class="hero-sub">Umrah Visitors Smart Forecasting System</div>
        <div class="hero-text">
            لوحة تفاعلية لعرض توقعات الازدحام واقتراح اليوم الأنسب لأداء العمرة.
        </div>
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    cards = [
        ("📈", "توقعات ذكية", "عرض مستوى الازدحام المتوقع."),
        ("📅", "أفضل يوم", "اقتراح يوم أقل ازدحامًا."),
        ("✅", "توصية واضحة", "مساعدة المعتمر في القرار."),
    ]

    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            H(f"""
            <div class="section-box" style="text-align:center;">
                <div style="font-size:22px;">{icon}</div>
                <div style="color:#064b3b; font-weight:800; font-size:14px; margin-top:8px;">{esc(title)}</div>
                <div style="color:#705f39; font-weight:600; font-size:11px; margin-top:5px;">{esc(desc)}</div>
            </div>
            """)

    st.markdown("<br>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.05, 1])

    with mid:
        if st.button("ابدأ إدخال البيانات", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()


# =====================================================
# Input Page
# =====================================================

def input_page():
    show_header()

    with st.form("input_form"):
        H("""
        <div class="form-title">نظام التنبؤ بمستويات ازدحام المعتمرين</div>
        <div class="form-sub">المسجد الحرام الشريف</div>
        <div class="form-line"></div>
        <div class="form-section">✎ إدخال بيانات</div>
        """)

        c1, c2 = st.columns(2)

        with c1:
            name = st.text_input("الاسم الكامل", placeholder="أدخل اسم المعتمر")

        with c2:
            nationality = st.selectbox("الجنسية", NATIONALITY_OPTIONS, index=0)

        c3, c4 = st.columns(2)

        with c3:
            month = st.selectbox("الشهر الهجري", HIJRI_MONTHS, index=10)

        with c4:
            day = st.selectbox("التاريخ الهجري", list(range(1, 31)), index=18)

        blocked = (
            month == "ذو القعدة" and
            nationality == "غير سعودي" and
            1 <= int(day) <= 15
        )

        if blocked:
            st.error("لا يمكن عرض النتائج لهذا الاختيار؛ لأن الجنسية غير سعودي خلال الفترة من 1 إلى 15 ذو القعدة.")

        st.markdown("<br>", unsafe_allow_html=True)

        _, mid, _ = st.columns([1, 1.1, 1])

        with mid:
            submitted = st.form_submit_button("عرض النتائج", use_container_width=True)

        if submitted:
            if not name.strip():
                st.warning("الرجاء إدخال الاسم الكامل.")
                return

            if blocked:
                return

            st.session_state.entered = True
            st.session_state.show_best_day = False
            st.session_state.selected_name = name.strip()
            st.session_state.selected_nationality = nationality
            st.session_state.selected_month = month
            st.session_state.selected_day = int(day)
            st.session_state.page = "dashboard"
            st.rerun()


# =====================================================
# Dashboard Page
# =====================================================

def dashboard_page():
    show_header()

    if not st.session_state.entered:
        st.warning("الرجاء إدخال البيانات أولًا.")

        if st.button("الذهاب إلى إدخال البيانات"):
            st.session_state.page = "input"
            st.rerun()

        return

    month = st.session_state.selected_month
    day = st.session_state.selected_day
    show_hajj = is_hajj_related_month(month)

    df = load_data()
    df7 = get_7_days(df, month, day)

    if df7.empty:
        st.error("لا توجد بيانات مطابقة لليوم أو الشهر المختار داخل ملف المودل.")
        return

    selected_df = df7[df7["Hijri_Day_Num"] == int(day)]

    if selected_df.empty:
        st.error("لا توجد بيانات لليوم المختار داخل ملف المودل.")
        return

    row = selected_df.iloc[0]

    weekday = row.get("Weekday_AR", "غير محدد")
    hijri_date = row.get("Hijri_Date", f"{day}-{month}")
    prediction = float(row.get("Prediction", 0))
    crowd_level = normalize_level(row.get("Local_Crowding_Level", "متوسط"))
    decision = get_decision(crowd_level)
    reason = get_reason(decision)

    temp = row.get("AvgTemp_C", np.nan)
    temp_text = "غير متاحة" if pd.isna(temp) else f"{float(temp):.1f}°C"

    hajj_count = format_seasonal_count(row.get("Hajj", 0))
    tawaf_count = format_seasonal_count(row.get("Tawaf_Ifadah", 0))

    best_day = get_best_day(df7, day)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        top_card("اليوم المختار", weekday, hijri_date, "📅")

    with c2:
        top_card("المعتمرون المتوقعون", format_number(prediction), "معتمر", "👥")

    with c3:
        top_card("مستوى الازدحام", crowd_level, english_level(crowd_level), "📊", crowd_level)

    with c4:
        top_card("درجة الحرارة", temp_text, "متوسط اليوم", "🌡️")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    H(f"""
    <div class="reco-box">
        <div class="reco-label">التوصية النهائية</div>
        <div class="reco-title" style="color:{LEVEL_COLORS.get(crowd_level, "#064b3b")};">{esc(decision)}</div>
        <div class="reco-line"></div>
        <div class="reco-text">{esc(reason)}</div>
    </div>
    """)

    if show_hajj:
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        s1, s2 = st.columns(2)

        with s1:
            info_pill(f"عدد الحجاج المتوقع: {hajj_count}")

        with s2:
            info_pill(f"عدد طواف الإفاضة المتوقع: {tawaf_count}")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([2.35, 0.78], gap="medium")

    with left_col:
        H("""
        <div class="section-box">
            <div class="section-title">توقعات ازدحام المعتمرين – الأيام القادمة (7 أيام)</div>
            <div class="section-line"></div>
        </div>
        """)

        fig = build_chart(df7)
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        H("""
        <div class="best-panel">
            <div class="best-title">هل تبحث عن يوم أفضل؟</div>
            <div class="best-sub">يمكنك عرض اليوم الأقل ازدحامًا ضمن الأيام السبعة القريبة.</div>
        </div>
        """)

        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

        if st.button("عرض اليوم الأفضل", use_container_width=True):
            st.session_state.show_best_day = True
            st.rerun()

        if st.session_state.show_best_day and best_day is not None:
            best_level = normalize_level(best_day.get("Local_Crowding_Level", "منخفض"))

            H(f"""
            <div class="best-result-card">
                <div class="best-result-label">اليوم المقترح</div>
                <div class="best-result-day">{esc(best_day.get("Weekday_AR", "غير متاح"))}</div>
                <div class="best-result-date">{esc(best_day.get("Hijri_Date", ""))}</div>
                <div class="best-result-badge"
                     style="background:{LEVEL_BG.get(best_level, '#eef8f1')};
                            color:{LEVEL_COLORS.get(best_level, '#208b55')};
                            border:1px solid {LEVEL_BORDER.get(best_level, '#b9dec6')};">
                    مستوى {esc(best_level)}
                </div>
                <div class="best-result-number">
                    المعتمرون المتوقعون: {format_number(best_day.get("Prediction", 0))}
                </div>
            </div>
            """)

    with st.expander("عرض تفاصيل الأيام السبعة"):
        display_df = prepare_display_table(df7)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])

    with mid:
        if st.button("رجوع لإدخال بيانات جديدة", use_container_width=True):
            st.session_state.page = "input"
            st.session_state.show_best_day = False
            st.rerun()


# =====================================================
# Router
# =====================================================

if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "input":
    input_page()

else:
    dashboard_page()
