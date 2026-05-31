from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="نظام ذكي للتنبؤ بمستويات ازدحام المعتمرين",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_PATHS = [
    Path("models/growth_rate_xgboost_model.pkl"),
    Path("growth_rate_xgboost_model.pkl")
]

DATE_COL   = "Gregorian_Date"
MONTH_COL  = "Hijri_Month"
TARGET_COL = "المعتمرين"

HIJRI_MONTHS = [
    "محرم", "صفر", "ربيع الأول", "ربيع الآخر",
    "جماد الأول", "جماد الآخر",
    "رجب", "شعبان", "رمضان", "شوال",
    "ذو القعدة", "ذو الحجة"
]

NATIONALITY_OPTIONS = ["سعودي", "غير سعودي"]

LEVEL_COLORS = {
    "منخفض": "#1A9B6C",
    "متوسط": "#C8921E",
    "مرتفع": "#D05045"
}
LEVEL_BG = {
    "منخفض": "#EEF9F3",
    "متوسط": "#FFF8E8",
    "مرتفع": "#FFF1F0"
}
LEVEL_BORDER = {
    "منخفض": "#A8D9BF",
    "متوسط": "#E8CC94",
    "مرتفع": "#E4AFAA"
}


# ── Utility helpers ──────────────────────────────────────────────────────────

def H(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def esc(x):
    if pd.isna(x):
        return ""
    return html.escape(str(x))


def normalize_month_name(month_name):
    text = str(month_name).strip()
    mapping = {
        "محرم": "محرم", "صفر": "صفر",
        "ربيع الأول": "ربيع الأول", "ربيع الاول": "ربيع الأول", "ربيع اول": "ربيع الأول",
        "ربيع الآخر": "ربيع الآخر", "ربيع الاخر": "ربيع الآخر", "ربيع الثاني": "ربيع الآخر",
        "جمادى الأولى": "جماد الأول", "جمادى الاولى": "جماد الأول",
        "جماد الأولى": "جماد الأول", "جماد الاولى": "جماد الأول",
        "جماد الأول": "جماد الأول", "جماد الاول": "جماد الأول",
        "جمادى الآخرة": "جماد الآخر", "جمادى الاخرة": "جماد الآخر",
        "جمادى الثانية": "جماد الآخر", "جمادى الثاني": "جماد الآخر",
        "جماد الآخرة": "جماد الآخر", "جماد الاخرة": "جماد الآخر",
        "جماد الآخر": "جماد الآخر", "جماد الاخر": "جماد الآخر",
        "جماد الثاني": "جماد الآخر",
        "رجب": "رجب", "شعبان": "شعبان", "رمضان": "رمضان", "شوال": "شوال",
        "ذو القعدة": "ذو القعدة", "ذو القعده": "ذو القعدة",
        "ذو الحجة": "ذو الحجة", "ذو الحجه": "ذو الحجة",
    }
    return mapping.get(text, text)


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
        "Low": "منخفض", "Medium": "متوسط", "High": "مرتفع", "Critical": "مرتفع",
        "low": "منخفض", "medium": "متوسط", "high": "مرتفع", "critical": "مرتفع",
        "منخفض": "منخفض", "متوسط": "متوسط", "مرتفع": "مرتفع",
        "حرج": "مرتفع", "عالي": "مرتفع", "مرتفع جدًا": "مرتفع",
    }
    return mapping.get(text, text)


def should_show_hajj_count(month_name, day):
    month_name = normalize_month_name(month_name)
    try:
        day = int(day)
    except Exception:
        return False
    if month_name == "ذو القعدة":
        return True
    if month_name == "ذو الحجة" and 1 <= day <= 9:
        return True
    return False


def should_show_tawaf_ifadah(month_name, day):
    month_name = normalize_month_name(month_name)
    try:
        day = int(day)
    except Exception:
        return False
    return month_name == "ذو الحجة" and day >= 10


def hex_to_rgba(hex_color, alpha=0.18):
    hex_color = str(hex_color).replace("#", "").strip()
    if len(hex_color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_existing_model_path():
    for path in MODEL_PATHS:
        if path.exists():
            return path
    return None


def extract_hijri_day(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    nums, current = [], ""
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
        return "مرتفع"

    return series.apply(classify)


def normalize_columns(df):
    df = df.copy()
    rename_map = {
        "Hajj_Feature": "Hajj", "Tawaf_Ifadah_Feature": "Tawaf_Ifadah",
        "Tawaf_Ifadah_Featureh": "Tawaf_Ifadah", "Umrah_Count": "المعتمرين",
        "Visitors": "المعتمرين", "Predicted": "Prediction",
        "Prediction_Count": "Prediction", "predicted_visitors": "Prediction",
        "Gregorian Date": "Gregorian_Date", "Hijri Month": "Hijri_Month",
        "Hijri Day": "Hijri_Day"
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    if MONTH_COL in df.columns:
        df[MONTH_COL] = df[MONTH_COL].apply(normalize_month_name)
    return df


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
        weekday_map = {0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس",
                       4: "الجمعة", 5: "السبت", 6: "الأحد"}
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
    df = df.dropna(subset=[DATE_COL, "Hijri_Day_Num"])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


@st.cache_data
def load_data():
    model_path = get_existing_model_path()
    if model_path is None:
        st.error("لم يتم العثور على ملف المودل. تأكد أن الملف موجود باسم growth_rate_xgboost_model.pkl داخل مجلد models.")
        st.stop()
    try:
        package = joblib.load(model_path)
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل ملف المودل: {e}")
        st.stop()
    if isinstance(package, pd.DataFrame):
        return add_display_cols(package)
    if isinstance(package, dict):
        for key in ["test_df", "results_df", "df", "data", "predictions"]:
            if key in package and isinstance(package[key], pd.DataFrame):
                return add_display_cols(package[key])
    st.error("ملف المودل تم تحميله، لكن لا يحتوي على جدول نتائج مناسب.")
    st.stop()


def get_7_days(df, month, day):
    month = normalize_month_name(month)
    selected = df[
        (df[MONTH_COL].astype(str).str.strip() == str(month).strip()) &
        (df["Hijri_Day_Num"] == int(day))
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    selected = selected.sort_values(DATE_COL)
    start_date = selected.iloc[0][DATE_COL]
    if pd.isna(start_date):
        return pd.DataFrame()
    end_date = start_date + pd.Timedelta(days=6)
    out = df[(df[DATE_COL] >= start_date) & (df[DATE_COL] <= end_date)].copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    out["Local_Crowding_Level"] = classify_by_quantiles(out["Prediction"])
    out["Local_Crowding_Level"] = out["Local_Crowding_Level"].apply(normalize_level)
    return out


def get_available_days(df, month):
    month = normalize_month_name(month)
    days = (
        df[df[MONTH_COL].astype(str).str.strip() == str(month).strip()]["Hijri_Day_Num"]
        .dropna().astype(int).sort_values().unique().tolist()
    )
    return days


def get_best_day(df7, current_day, current_month):
    current_month = normalize_month_name(current_month)
    alt = df7[
        ~(
            (df7["Hijri_Day_Num"] == int(current_day)) &
            (df7[MONTH_COL].astype(str).str.strip() == str(current_month).strip())
        )
    ].copy()
    if alt.empty:
        return None
    return alt.sort_values("Prediction").iloc[0]


def get_decision(level):
    level = normalize_level(level)
    if level == "منخفض":
        return "مناسب للزيارة"
    elif level == "مرتفع":
        return "يفضل اختيار يوم آخر"
    return "مناسب مع الحذر"


def get_reason(decision):
    if decision == "مناسب للزيارة":
        return "من المتوقع أن يكون مستوى الازدحام منخفضًا مقارنة بالأيام القريبة، لذلك يعد هذا اليوم خيارًا مناسبًا لأداء العمرة."
    if decision == "يفضل اختيار يوم آخر":
        return "الازدحام المتوقع مرتفع مقارنةً بالأيام القريبة، لذلك يفضل اختيار يوم أقل ازدحامًا."
    return "الازدحام المتوقع ضمن المستوى المتوسط، لذلك يمكن أداء العمرة مع الحذر واختيار الوقت المناسب."


# ── Chart ────────────────────────────────────────────────────────────────────

def build_chart(df7, selected_month=None, selected_day=None):
    cdf = df7.copy().reset_index(drop=True)
    cdf["Prediction"] = pd.to_numeric(cdf["Prediction"], errors="coerce")
    cdf["Local_Crowding_Level"] = cdf["Local_Crowding_Level"].apply(normalize_level)
    cdf["x_label"] = (
        cdf["Weekday_AR"].astype(str) + "<br>" +
        cdf["Hijri_Day_Num"].astype(int).astype(str) + " " +
        cdf[MONTH_COL].astype(str)
    )

    selected_mask = pd.Series(False, index=cdf.index)
    if selected_month is not None and selected_day is not None:
        selected_month = normalize_month_name(selected_month)
        selected_mask = (
            (cdf[MONTH_COL].astype(str).str.strip() == str(selected_month).strip()) &
            (cdf["Hijri_Day_Num"].astype(int) == int(selected_day))
        )

    y_min = cdf["Prediction"].min()
    y_max = cdf["Prediction"].max()
    pad   = max((y_max - y_min) * 0.32, 500)
    base_y = max(0, y_min - pad * 0.6)

    fig = go.Figure()

    # Gradient fill areas under each segment
    for i in range(len(cdf) - 1):
        x0, x1 = cdf.loc[i, "x_label"], cdf.loc[i + 1, "x_label"]
        y0, y1 = cdf.loc[i, "Prediction"], cdf.loc[i + 1, "Prediction"]
        l1 = normalize_level(cdf.loc[i, "Local_Crowding_Level"])
        l2 = normalize_level(cdf.loc[i + 1, "Local_Crowding_Level"])
        seg_color = LEVEL_COLORS["مرتفع"] if "مرتفع" in [l1, l2] else (
            LEVEL_COLORS["متوسط"] if "متوسط" in [l1, l2] else LEVEL_COLORS["منخفض"])
        fig.add_trace(go.Scatter(
            x=[x0, x0, x1, x1], y=[base_y, y0, y1, base_y],
            mode="lines", fill="toself",
            fillcolor=hex_to_rgba(seg_color, 0.13),
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip", showlegend=False
        ))

    # Color-coded line segments
    for i in range(len(cdf) - 1):
        x0, x1 = cdf.loc[i, "x_label"], cdf.loc[i + 1, "x_label"]
        y0, y1 = cdf.loc[i, "Prediction"], cdf.loc[i + 1, "Prediction"]
        l1 = normalize_level(cdf.loc[i, "Local_Crowding_Level"])
        l2 = normalize_level(cdf.loc[i + 1, "Local_Crowding_Level"])
        seg_color = LEVEL_COLORS["مرتفع"] if "مرتفع" in [l1, l2] else (
            LEVEL_COLORS["متوسط"] if "متوسط" in [l1, l2] else LEVEL_COLORS["منخفض"])
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=seg_color, width=3.5, shape="spline"),
            hoverinfo="skip", showlegend=False
        ))

    # Data points by level
    for level in ["منخفض", "متوسط", "مرتفع"]:
        ldf = cdf[cdf["Local_Crowding_Level"] == level].copy()
        if ldf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=ldf["x_label"], y=ldf["Prediction"],
            mode="markers+text",
            marker=dict(size=13, color=LEVEL_COLORS[level],
                        line=dict(color="white", width=2.5)),
            text=ldf["Prediction"].apply(
                lambda v: f"{int(round(float(v))):,}" if not pd.isna(v) else ""),
            textposition="top center",
            textfont=dict(size=10.5, color="#053D33", family="Cairo"),
            hovertemplate="<b>%{x}</b><br>العدد المتوقع: %{y:,.0f}<extra></extra>",
            name=level
        ))

    # Selected day ring + label
    if selected_mask.any():
        sp = cdf[selected_mask].iloc[[0]]
        fig.add_trace(go.Scatter(
            x=sp["x_label"], y=sp["Prediction"], mode="markers",
            marker=dict(size=42, color="rgba(196,144,63,0.12)",
                        line=dict(color="rgba(196,144,63,0.30)", width=1.5)),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=sp["x_label"], y=sp["Prediction"], mode="markers+text",
            marker=dict(size=20, color="#FFF8EC",
                        line=dict(color="#C4903F", width=3.5)),
            text=["اليوم المختار"],
            textposition="bottom center",
            textfont=dict(size=11, color="#062B25", family="Cairo"),
            hovertemplate="<b>اليوم المختار</b><br>%{x}<br>العدد المتوقع: %{y:,.0f}<extra></extra>",
            showlegend=False
        ))

    q1 = cdf["Prediction"].quantile(0.33)
    q2 = cdf["Prediction"].quantile(0.66)
    fig.add_hline(y=q1, line_dash="dot", line_color="rgba(26,155,108,0.22)", line_width=1)
    fig.add_hline(y=q2, line_dash="dot", line_color="rgba(200,146,30,0.22)", line_width=1)

    fig.update_layout(
        height=370,
        margin=dict(l=14, r=14, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.48)",
        font=dict(family="Cairo", size=11.5, color="#082E28"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06,
            xanchor="left", x=0,
            title_text="مستوى الازدحام: ",
            title_font=dict(size=11),
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(showgrid=False, title="",
                   tickfont=dict(family="Cairo", size=11),
                   linecolor="rgba(5,42,36,0.10)"),
        yaxis=dict(
            range=[base_y, y_max + pad],
            showgrid=True, gridcolor="rgba(120,100,60,0.07)",
            tickformat=",",
            title=dict(text="عدد المعتمرين", font=dict(size=10)),
            tickfont=dict(size=10)
        ),
        hoverlabel=dict(
            bgcolor="#052A24", bordercolor="#C4903F",
            font=dict(color="#FFF8EC", family="Cairo", size=12)
        )
    )
    return fig


# ── Global CSS ────────────────────────────────────────────────────────────────

H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800;900&display=swap');

/* ── Design tokens ── */
:root {
  --g950: #010E0C;
  --g900: #052A24;
  --g800: #084038;
  --g700: #0D6B4F;
  --g600: #1A9B6C;
  --gold:  #C4903F;
  --gold2: #D8B870;
  --gold3: #F0DCA8;
  --cream: #FAF6EE;
  --ink:   #0A2A24;
  --muted: #5E6D66;
  --low:   #1A9B6C;
  --mid:   #C8921E;
  --high:  #D05045;
  --r-sm:  14px;
  --r-md:  20px;
  --r-lg:  26px;
  --r-xl:  32px;
}

/* ── Reset / base ── */
html, body, [class*="css"] {
  font-family: 'Cairo', sans-serif !important;
  direction: rtl;
}
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"]   { display: none !important; }
.js-plotly-plot .plotly .modebar   { display: none !important; }

/* ── App background ── */
.stApp {
  background:
    radial-gradient(ellipse at 6% 5%,   rgba(196,144,63,0.15)  0%, transparent 26%),
    radial-gradient(ellipse at 94% 8%,  rgba(13,107,79,0.13)   0%, transparent 26%),
    radial-gradient(ellipse at 48% 96%, rgba(5,42,36,0.07)     0%, transparent 32%),
    linear-gradient(150deg, #FEFBF4 0%, #F6EDD9 52%, #FDFAF3 100%);
  color: var(--ink);
  min-height: 100vh;
}

/* Subtle Islamic-geometry tessellation watermark */
.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.045;
  background-image:
    linear-gradient(30deg,  rgba(5,42,36,0.55) 12%, transparent 12.5%, transparent 87%, rgba(5,42,36,0.55) 87.5%),
    linear-gradient(150deg, rgba(5,42,36,0.55) 12%, transparent 12.5%, transparent 87%, rgba(5,42,36,0.55) 87.5%),
    linear-gradient(30deg,  rgba(5,42,36,0.55) 12%, transparent 12.5%, transparent 87%, rgba(5,42,36,0.55) 87.5%),
    linear-gradient(150deg, rgba(5,42,36,0.55) 12%, transparent 12.5%, transparent 87%, rgba(5,42,36,0.55) 87.5%);
  background-size: 72px 124px;
  background-position: 0 0, 0 0, 36px 62px, 36px 62px;
}
.stApp > * { position: relative; z-index: 1; }

/* ── Layout ── */
.block-container {
  max-width: 1360px !important;
  padding-top:    0.65rem !important;
  padding-bottom: 2rem !important;
}
div[data-testid="stHorizontalBlock"] {
  gap: 1rem !important;
  align-items: stretch !important;
}

/* ── Buttons ── */
.stButton button {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #0D6B4F 0%, #052A24 100%) !important;
  color: #FFF8EC !important;
  border: 1px solid rgba(196,144,63,0.38) !important;
  border-radius: var(--r-sm) !important;
  height: 48px !important;
  font-size: 14px !important;
  font-weight: 800 !important;
  box-shadow: 0 10px 24px rgba(5,42,36,0.15), inset 0 1px 0 rgba(255,255,255,0.09) !important;
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
}
.stButton button::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(105deg, transparent 18%, rgba(255,255,255,0.11) 50%, transparent 82%);
  transform: translateX(120%);
  transition: .36s ease;
}
.stButton button:hover {
  transform: translateY(-2px) !important;
  filter: brightness(1.08) !important;
  box-shadow: 0 18px 36px rgba(5,42,36,0.20) !important;
}
.stButton button:hover::after { transform: translateX(-120%); }
.stButton button:focus, .stButton button:active {
  outline: none !important;
  box-shadow: 0 10px 24px rgba(5,42,36,0.15) !important;
}

/* Nav active state (injected per-page via wrapper class) */
.nav-active-1 div[data-testid="column"]:nth-child(1) .stButton button,
.nav-active-2 div[data-testid="column"]:nth-child(2) .stButton button,
.nav-active-3 div[data-testid="column"]:nth-child(3) .stButton button {
  background: linear-gradient(135deg, #B07A28 0%, #8A5C12 100%) !important;
  border-color: rgba(196,144,63,0.62) !important;
  box-shadow: 0 10px 28px rgba(196,144,63,0.28), inset 0 1px 0 rgba(255,255,255,0.12) !important;
  color: #FFF8EC !important;
}

/* ── Header ── */
.site-header {
  position: relative;
  overflow: hidden;
  min-height: 114px;
  padding: 20px 40px;
  margin-bottom: 14px;
  border-radius: var(--r-xl);
  border: 1px solid rgba(196,144,63,0.34);
  background:
    radial-gradient(circle at 12% 22%, rgba(216,184,112,0.22) 0%, transparent 30%),
    radial-gradient(circle at 88% 72%, rgba(26,155,108,0.18)  0%, transparent 30%),
    linear-gradient(140deg, #011410 0%, #06382E 50%, #031C18 100%);
  box-shadow: 0 22px 52px rgba(5,42,36,0.20), inset 0 1px 0 rgba(255,255,255,0.07);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

/* Fine grid texture on header */
.site-header::before {
  content: "";
  position: absolute;
  inset: 0;
  opacity: 0.085;
  background-image:
    linear-gradient(60deg,  rgba(255,255,255,0.28) 1px, transparent 1px),
    linear-gradient(120deg, rgba(216,184,112,0.28) 1px, transparent 1px);
  background-size: 38px 38px;
  pointer-events: none;
}
/* Gloss sweep */
.site-header::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(112deg, transparent 0%, rgba(255,255,255,0.07) 44%, transparent 76%);
  pointer-events: none;
}

.header-left {
  position: relative;
  z-index: 2;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}
.header-platform-tag {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 13px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(216,184,112,0.24);
  color: #D8B870;
  font-size: 10.5px;
  font-weight: 900;
  letter-spacing: 0.3px;
  margin-bottom: 6px;
}
.hpt-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #20C97A;
  box-shadow: 0 0 7px 3px rgba(32,201,122,0.35);
}
.header-title {
  color: #FFF8EC;
  font-size: 28px;
  font-weight: 900;
  letter-spacing: -0.7px;
  line-height: 1.28;
  text-shadow: 0 6px 20px rgba(0,0,0,0.18);
}
.header-sub {
  color: rgba(216,184,112,0.80);
  font-size: 13px;
  font-weight: 700;
  margin-top: 5px;
}

/* Decorative 8-point star on header right */
.header-star {
  position: relative;
  z-index: 2;
  width: 88px;
  height: 88px;
  opacity: 0.28;
  flex-shrink: 0;
}
.header-star svg { width: 100%; height: 100%; }

/* Gold rule below header text */
.header-rule {
  width: 220px;
  height: 1.5px;
  margin-top: 10px;
  background: linear-gradient(90deg, rgba(216,184,112,0.90), transparent);
}

/* ── Navigation ── */
.nav-wrap {
  margin-bottom: 2px;
}
.nav-meta {
  text-align: center;
  font-size: 10px;
  font-weight: 900;
  color: rgba(5,42,36,0.48);
  margin: 0 0 7px 0;
  letter-spacing: 0.4px;
}

/* ── Home: Hero ── */
.hero-wrap {
  position: relative;
  overflow: hidden;
  margin-bottom: 18px;
  border-radius: var(--r-xl);
  border: 1px solid rgba(196,144,63,0.22);
  background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(255,252,242,0.76));
  box-shadow: 0 20px 48px rgba(5,42,36,0.07);
  backdrop-filter: blur(18px);
  display: flex;
  align-items: stretch;
  min-height: 260px;
}
.hero-wrap::before {
  content: "";
  position: absolute;
  inset: 18px;
  border-radius: 22px;
  border: 1px solid rgba(196,144,63,0.10);
  pointer-events: none;
}

/* Text side */
.hero-text-side {
  flex: 1;
  padding: 44px 44px 40px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: right;
  position: relative;
  z-index: 2;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 7px 16px;
  border-radius: 999px;
  color: #0D6B4F;
  background: rgba(13,107,79,0.08);
  border: 1px solid rgba(13,107,79,0.14);
  font-size: 11px;
  font-weight: 900;
  margin-bottom: 16px;
  width: fit-content;
}
.hero-title {
  color: #052A24;
  font-size: 36px;
  font-weight: 900;
  letter-spacing: -1px;
  line-height: 1.28;
}
.hero-gold {
  color: #9C6E18;
  font-size: 15px;
  font-weight: 800;
  margin-top: 10px;
}
.hero-body {
  color: #4A5A54;
  font-size: 13.5px;
  font-weight: 600;
  max-width: 560px;
  margin-top: 14px;
  line-height: 2;
}

/* Visual side — CSS crowd level gauge */
.hero-visual-side {
  width: 260px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 28px 20px;
  position: relative;
  z-index: 2;
}
/* decorative radial gradient blob behind the gauge */
.hero-visual-side::before {
  content: "";
  position: absolute;
  width: 200px;
  height: 200px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(13,107,79,0.07), transparent 68%);
  pointer-events: none;
}

.level-gauge {
  position: relative;
  width: 180px;
  height: 180px;
}
.level-gauge svg { width: 100%; height: 100%; overflow: visible; }

/* ── Home: Platform stats strip ── */
.platform-stats {
  display: flex;
  gap: 12px;
  margin-bottom: 18px;
  flex-wrap: wrap;
}
.pstat {
  flex: 1;
  min-width: 130px;
  padding: 16px 18px;
  border-radius: var(--r-md);
  background: linear-gradient(145deg, rgba(255,255,255,0.88), rgba(255,249,232,0.70));
  border: 1px solid rgba(196,144,63,0.20);
  box-shadow: 0 10px 24px rgba(5,42,36,0.055);
  backdrop-filter: blur(12px);
  text-align: center;
}
.pstat-n {
  color: #052A24;
  font-size: 26px;
  font-weight: 900;
  line-height: 1;
  letter-spacing: -0.5px;
}
.pstat-l {
  color: #607068;
  font-size: 11px;
  font-weight: 800;
  margin-top: 6px;
}

/* ── Home: Feature cards ── */
.feat-card {
  position: relative;
  overflow: hidden;
  padding: 26px 22px 24px;
  border-radius: 22px;
  background: linear-gradient(145deg, rgba(255,255,255,0.88), rgba(255,249,232,0.72));
  border: 1px solid rgba(196,144,63,0.20);
  box-shadow: 0 14px 36px rgba(5,42,36,0.06);
  backdrop-filter: blur(14px);
  transition: transform .2s ease, box-shadow .2s ease;
  height: 100%;
}
.feat-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 24px 54px rgba(5,42,36,0.10);
}
.feat-card::before {
  content: "";
  position: absolute;
  top: 0; right: 0; left: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--gold), rgba(13,107,79,0.50), transparent);
}
.feat-card::after {
  content: "";
  position: absolute;
  width: 72px; height: 72px;
  border-radius: 50%;
  left: -26px; bottom: -26px;
  background: rgba(196,144,63,0.07);
}
.feat-icon {
  width: 44px; height: 44px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  color: #9C6E18;
  background: rgba(216,184,112,0.14);
  border: 1px solid rgba(196,144,63,0.26);
  font-size: 14px;
  font-weight: 900;
  margin-bottom: 14px;
}
.feat-title { color: #052A24; font-size: 17px; font-weight: 900; }
.feat-desc  { color: #607068; font-size: 12.5px; font-weight: 600; line-height: 1.85; margin-top: 8px; }

/* ── Input page ── */
.form-page-header {
  text-align: center;
  margin-bottom: 22px;
}
.form-step-badge {
  display: inline-block;
  padding: 7px 18px;
  border-radius: 999px;
  background: rgba(13,107,79,0.08);
  border: 1px solid rgba(13,107,79,0.14);
  color: #0D6B4F;
  font-size: 11px;
  font-weight: 900;
  margin-bottom: 12px;
}
.form-page-title {
  color: #052A24;
  font-size: 26px;
  font-weight: 900;
  letter-spacing: -0.5px;
}
.form-page-sub {
  color: #9C6E18;
  font-size: 13px;
  font-weight: 700;
  margin-top: 7px;
}
.form-rule {
  width: 220px;
  height: 1.5px;
  margin: 13px auto 0;
  background: linear-gradient(90deg, transparent, #C4903F, transparent);
}

.form-shell {
  position: relative;
  overflow: hidden;
  padding: 32px 36px 28px;
  border-radius: var(--r-xl);
  background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(255,252,242,0.76));
  border: 1px solid rgba(196,144,63,0.24);
  box-shadow: 0 18px 44px rgba(5,42,36,0.07);
  backdrop-filter: blur(18px);
}
.form-shell::before {
  content: "";
  position: absolute;
  top: 0; right: 0; left: 0;
  height: 3px;
  background: linear-gradient(90deg, transparent, #C4903F 35%, rgba(13,107,79,0.55) 70%, transparent);
}
.form-section-label {
  color: #607068;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.4px;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(196,144,63,0.16);
  text-align: right;
}

.stTextInput label, .stSelectbox label {
  color: #052A24 !important;
  font-weight: 800 !important;
  font-size: 13px !important;
}
.stTextInput input {
  min-height: 50px !important;
  border-radius: var(--r-sm) !important;
  background: rgba(255,254,252,0.96) !important;
  border: 1.5px solid rgba(196,144,63,0.28) !important;
  box-shadow: 0 6px 16px rgba(5,42,36,0.04) !important;
  font-family: 'Cairo', sans-serif !important;
  font-size: 14px !important;
  color: #052A24 !important;
}
.stTextInput input:focus {
  border-color: rgba(13,107,79,0.50) !important;
  box-shadow: 0 0 0 3px rgba(13,107,79,0.09) !important;
}
.stSelectbox > div > div {
  min-height: 50px !important;
  border-radius: var(--r-sm) !important;
  background: rgba(255,254,252,0.96) !important;
  border: 1.5px solid rgba(196,144,63,0.28) !important;
}

/* ── Dashboard: Result banner ── */
.result-banner {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 0;
  padding: 22px 32px;
  margin-bottom: 14px;
  border-radius: var(--r-xl);
  border: 1px solid rgba(196,144,63,0.34);
  background:
    radial-gradient(circle at 14% 24%, rgba(255,255,255,0.10) 0%, transparent 28%),
    linear-gradient(145deg, #031D19 0%, #064C3E 50%, #021510 100%);
  box-shadow: 0 18px 44px rgba(5,42,36,0.22), inset 0 1px 0 rgba(255,255,255,0.08);
  backdrop-filter: blur(10px);
}
/* Fine grid texture */
.result-banner::before {
  content: "";
  position: absolute;
  inset: 0;
  opacity: 0.08;
  background-image:
    linear-gradient(60deg,  rgba(255,255,255,0.25) 1px, transparent 1px),
    linear-gradient(120deg, rgba(196,144,63,0.25) 1px, transparent 1px);
  background-size: 36px 36px;
}
/* Gold arc ornament (right side) */
.result-banner::after {
  content: "";
  position: absolute;
  right: -80px; bottom: -120px;
  width: 420px; height: 260px;
  border-radius: 50%;
  border-top: 1.5px solid rgba(196,144,63,0.50);
  transform: rotate(-14deg);
  pointer-events: none;
}

.rb-inner { position: relative; z-index: 2; }

/* Date / context side */
.rb-date {
  text-align: right;
}
.rb-date-weekday {
  color: rgba(255,248,236,0.72);
  font-size: 12px;
  font-weight: 800;
  margin-bottom: 5px;
  letter-spacing: 0.2px;
}
.rb-date-hijri {
  color: #FFF8EC;
  font-size: 22px;
  font-weight: 900;
  line-height: 1.2;
  letter-spacing: -0.3px;
}
.rb-date-month {
  color: #D8B870;
  font-size: 12px;
  font-weight: 700;
  margin-top: 4px;
}

/* Level — center column */
.rb-level-col {
  text-align: center;
  padding: 0 32px;
  border-right: 1px solid rgba(255,255,255,0.10);
  border-left:  1px solid rgba(255,255,255,0.10);
}
.rb-level-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 14px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14);
  color: #EAD49A;
  font-size: 10.5px;
  font-weight: 900;
  margin-bottom: 10px;
  letter-spacing: 0.2px;
}
.rb-level-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--gold);
  box-shadow: 0 0 7px 3px rgba(196,144,63,0.36);
}
.rb-level-text {
  font-size: 56px;
  font-weight: 900;
  line-height: 0.95;
  letter-spacing: -1px;
  text-shadow: 0 10px 28px rgba(0,0,0,0.18);
}
.rb-level-label {
  color: #D8B870;
  font-size: 13px;
  font-weight: 800;
  margin-top: 8px;
}

/* Stats side */
.rb-stats {
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: flex-start;
}
.rb-stat-item {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.rb-stat-label {
  color: rgba(255,248,236,0.60);
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.2px;
}
.rb-stat-value {
  color: #FFF8EC;
  font-size: 20px;
  font-weight: 900;
  line-height: 1.15;
  letter-spacing: -0.3px;
}
.rb-stat-unit {
  color: #D8B870;
  font-size: 10px;
  font-weight: 700;
}

/* ── Dashboard: Main columns ── */

/* Chart container */
.chart-container {
  padding: 20px 22px 8px;
  border-radius: var(--r-lg);
  background: linear-gradient(145deg, rgba(255,255,255,0.90), rgba(255,251,240,0.74));
  border: 1px solid rgba(196,144,63,0.20);
  box-shadow: 0 14px 30px rgba(5,42,36,0.06);
  backdrop-filter: blur(14px);
  height: 100%;
}
.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}
.chart-title {
  color: #052A24;
  font-size: 14.5px;
  font-weight: 900;
}
.chart-legend-hint {
  display: flex;
  gap: 14px;
  align-items: center;
}
.clh-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 10.5px;
  font-weight: 800;
  color: #607068;
}
.clh-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.chart-rule {
  width: 70px;
  height: 1.5px;
  margin: 6px 0 2px auto;
  background: linear-gradient(90deg, #C4903F, transparent);
}

/* Action panel */
.action-panel { display: flex; flex-direction: column; gap: 12px; }

/* Recommendation card */
.reco-card {
  position: relative;
  overflow: hidden;
  padding: 20px 24px 22px;
  border-radius: var(--r-lg);
  background: linear-gradient(145deg, rgba(255,255,255,0.94), rgba(255,251,240,0.80));
  border: 1px solid rgba(196,144,63,0.26);
  box-shadow: 0 12px 28px rgba(5,42,36,0.06);
  backdrop-filter: blur(14px);
  text-align: center;
}
.reco-card::before {
  content: "";
  position: absolute;
  top: 0; right: 0; left: 0;
  height: 3px;
  background: linear-gradient(90deg, transparent, #C4903F 32%, rgba(13,107,79,0.52) 68%, transparent);
}
.reco-eyebrow  { color: #7A6A4C; font-size: 10.5px; font-weight: 800; letter-spacing: 0.3px; }
.reco-decision { font-size: 24px; font-weight: 900; margin: 6px 0 4px; letter-spacing: -0.3px; }
.reco-ornament {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin: 8px 0 10px;
  color: rgba(196,144,63,0.60);
  font-size: 10px;
}
.reco-ornament::before,
.reco-ornament::after {
  content: "";
  flex: 1;
  max-width: 64px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(196,144,63,0.50));
}
.reco-ornament::after {
  background: linear-gradient(90deg, rgba(196,144,63,0.50), transparent);
}
.reco-text {
  color: #2E3F3A;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.9;
}

/* Alternative day panel */
.alt-panel {
  position: relative;
  overflow: hidden;
  border-radius: var(--r-lg);
  background:
    radial-gradient(circle at 16% 18%, rgba(196,144,63,0.15) 0%, transparent 32%),
    linear-gradient(155deg, #031D19 0%, #074A3C 56%, #021510 100%);
  border: 1px solid rgba(196,144,63,0.30);
  box-shadow: 0 16px 38px rgba(5,42,36,0.18);
  backdrop-filter: blur(10px);
}
.alt-panel::before {
  content: "";
  position: absolute;
  inset: 0;
  opacity: 0.07;
  background-image:
    linear-gradient(60deg,  rgba(255,255,255,0.28) 1px, transparent 1px),
    linear-gradient(120deg, rgba(196,144,63,0.28) 1px, transparent 1px);
  background-size: 28px 28px;
}
.alt-panel-inner {
  position: relative;
  z-index: 2;
  padding: 22px 20px 18px;
}
.alt-title { color: #FFF8EC; font-size: 18px; font-weight: 900; letter-spacing: -0.3px; }
.alt-sub   { color: #D8B870; font-size: 11.5px; font-weight: 700; line-height: 1.85; margin-top: 8px; }

/* Result of best day */
.best-day-result {
  margin-top: 10px;
  padding: 16px;
  border-radius: var(--r-md);
  background: linear-gradient(145deg, rgba(238,249,243,0.98), rgba(255,255,255,0.90));
  border: 1px solid rgba(26,155,108,0.22);
  box-shadow: 0 10px 24px rgba(26,155,108,0.09);
  text-align: center;
}
.bdr-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 999px;
  color: #0D7A52;
  background: rgba(13,107,79,0.08);
  font-size: 10px;
  font-weight: 900;
  margin-bottom: 7px;
}
.bdr-day  { color: #052A24; font-size: 22px; font-weight: 900; line-height: 1.3; }
.bdr-meta { color: #4E6158; font-size: 11.5px; font-weight: 700; line-height: 1.85; margin-top: 6px; }
.bdr-meta strong { color: #052A24; font-weight: 900; }
.bdr-note { color: #0D6B4F; font-size: 10.5px; font-weight: 700; margin-top: 7px; line-height: 1.7; }

/* Seasonal info pill */
.info-pill-wrap {
  display: flex;
  justify-content: center;
  margin: 6px 0 2px;
}
.info-pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 20px;
  border-radius: 999px;
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(196,144,63,0.26);
  color: #4E3F20;
  font-size: 12px;
  font-weight: 800;
  box-shadow: 0 8px 20px rgba(5,42,36,0.05);
  backdrop-filter: blur(10px);
}
.info-pill-tag {
  padding: 4px 11px;
  border-radius: 999px;
  background: rgba(13,107,79,0.09);
  color: #0D6B4F;
  font-size: 10.5px;
  font-weight: 900;
}

/* ── Footer ── */
.site-footer {
  margin-top: 28px;
  padding: 14px 0 4px;
  border-top: 1px solid rgba(196,144,63,0.16);
  text-align: center;
  color: rgba(5,42,36,0.38);
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.2px;
}
.site-footer strong { color: rgba(5,42,36,0.55); font-weight: 900; }

/* ── Responsive ── */
@media (max-width: 900px) {
  .site-header { padding: 16px 20px; min-height: 90px; }
  .header-title { font-size: 20px; }
  .header-star  { width: 60px; height: 60px; }
  .hero-visual-side { display: none; }
  .hero-text-side { padding: 32px 24px; }
  .hero-title { font-size: 26px; }
  .result-banner { grid-template-columns: 1fr; gap: 16px; padding: 22px 20px; }
  .rb-level-col { border: none; padding: 8px 0; }
  .rb-level-text { font-size: 40px; }
  .rb-stats { align-items: flex-start; flex-direction: row; flex-wrap: wrap; gap: 16px; }
  .platform-stats { gap: 8px; }
  .pstat { min-width: 100px; }
}
</style>
""")


# ── UI building blocks ────────────────────────────────────────────────────────

def show_header():
    star_svg = """
    <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      <polygon points="50,5 61,35 94,35 68,57 79,91 50,70 21,91 32,57 6,35 39,35"
               fill="rgba(216,184,112,0.55)" stroke="rgba(216,184,112,0.30)" stroke-width="1"/>
      <polygon points="50,18 58,40 82,40 63,54 70,76 50,63 30,76 37,54 18,40 42,40"
               fill="none" stroke="rgba(216,184,112,0.22)" stroke-width="0.8"/>
    </svg>
    """
    H(f"""
    <div class="site-header">
      <div class="header-left">
        <div class="header-platform-tag">
          <span class="hpt-dot"></span>
          منصة دعم القرار — جامعي
        </div>
        <div class="header-title">المنصة الذكية لتوقع ازدحام المعتمرين</div>
        <div class="header-sub">اختيار الوقت الأنسب لأداء العمرة بناءً على نموذج XGBoost التنبؤي</div>
        <div class="header-rule"></div>
      </div>
      <div class="header-star">{star_svg}</div>
    </div>
    """)


def render_nav():
    page = st.session_state.get("page", "home")
    active_map = {"home": 1, "input": 2, "dashboard": 3}
    active_n   = active_map.get(page, 1)

    H(f"""
    <div class="nav-wrap nav-active-{active_n}">
    <div class="nav-meta">التنقل بين صفحات المنصة</div>
    """)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("الرئيسية", use_container_width=True, key="nav_home"):
            st.session_state.page = "home"
            st.rerun()
    with c2:
        if st.button("إدخال البيانات", use_container_width=True, key="nav_input"):
            st.session_state.page = "input"
            st.rerun()
    with c3:
        if st.button("لوحة النتائج", use_container_width=True, key="nav_dash"):
            st.session_state.page = "dashboard"
            st.rerun()
    H("</div>")
    H("<div style='height:12px;'></div>")


def render_footer():
    H("""
    <div class="site-footer">
      <strong>نظام ذكي للتنبؤ بمستويات ازدحام المعتمرين</strong>
      &nbsp;·&nbsp; مشروع تخرج · نموذج XGBoost · بيانات هجرية
    </div>
    """)


# ── Level gauge SVG (used on home page) ─────────────────────────────────────

def level_gauge_svg():
    return """
    <div class="level-gauge">
      <svg viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg">
        <!-- Outer ring: High -->
        <circle cx="90" cy="90" r="80" fill="none"
                stroke="rgba(208,80,69,0.18)" stroke-width="14"/>
        <circle cx="90" cy="90" r="80" fill="none"
                stroke="#D05045" stroke-width="14"
                stroke-dasharray="140 362" stroke-dashoffset="-50"
                stroke-linecap="round" opacity="0.70"/>
        <!-- Middle ring: Medium -->
        <circle cx="90" cy="90" r="58" fill="none"
                stroke="rgba(200,146,30,0.18)" stroke-width="12"/>
        <circle cx="90" cy="90" r="58" fill="none"
                stroke="#C8921E" stroke-width="12"
                stroke-dasharray="185 182" stroke-dashoffset="-30"
                stroke-linecap="round" opacity="0.80"/>
        <!-- Inner ring: Low -->
        <circle cx="90" cy="90" r="38" fill="none"
                stroke="rgba(26,155,108,0.18)" stroke-width="10"/>
        <circle cx="90" cy="90" r="38" fill="none"
                stroke="#1A9B6C" stroke-width="10"
                stroke-dasharray="220 19" stroke-dashoffset="-8"
                stroke-linecap="round" opacity="0.90"/>
        <!-- Center -->
        <circle cx="90" cy="90" r="20" fill="rgba(255,255,255,0.85)"/>
        <circle cx="90" cy="90" r="20" fill="none"
                stroke="rgba(196,144,63,0.30)" stroke-width="1"/>
        <!-- Labels -->
        <text x="90" y="20" text-anchor="middle"
              font-family="Cairo,sans-serif" font-size="9" font-weight="700"
              fill="#D05045">مرتفع</text>
        <text x="90" y="55" text-anchor="middle"
              font-family="Cairo,sans-serif" font-size="8.5" font-weight="700"
              fill="#C8921E">متوسط</text>
        <text x="90" y="78" text-anchor="middle"
              font-family="Cairo,sans-serif" font-size="8" font-weight="700"
              fill="#1A9B6C">منخفض</text>
        <!-- Center icon text -->
        <text x="90" y="94" text-anchor="middle"
              font-family="Cairo,sans-serif" font-size="14" font-weight="900"
              fill="#052A24">3</text>
        <text x="90" y="104" text-anchor="middle"
              font-family="Cairo,sans-serif" font-size="7.5" font-weight="700"
              fill="#607068">مستويات</text>
      </svg>
    </div>
    """


# ── Session state defaults ────────────────────────────────────────────────────

if "page"         not in st.session_state: st.session_state.page         = "home"
if "entered"      not in st.session_state: st.session_state.entered      = False
if "show_best_day" not in st.session_state: st.session_state.show_best_day = False


# ── Pages ─────────────────────────────────────────────────────────────────────

def home_page():
    show_header()
    render_nav()

    # Hero: 2-col layout
    H("""
    <div class="hero-wrap">
      <div class="hero-text-side">
        <div class="hero-badge">نظام دعم القرار للمعتمرين</div>
        <div class="hero-title">اختر الوقت الأنسب<br>لأداء العمرة</div>
        <div class="hero-gold">توقع مستوى الازدحام والحصول على توصية دقيقة لموعد زيارتك</div>
        <div class="hero-body">
          يستخدم النظام نموذج XGBoost للتنبؤ بعدد المعتمرين ومستوى الازدحام
          في أي يوم هجري تختاره، ويقترح البديل الأنسب من الأيام المجاورة.
        </div>
      </div>
    """ + f"""
      <div class="hero-visual-side">
        {level_gauge_svg()}
      </div>
    </div>
    """)

    # Platform stats strip
    H("""
    <div class="platform-stats">
      <div class="pstat"><div class="pstat-n">12</div><div class="pstat-l">شهر هجري مدعوم</div></div>
      <div class="pstat"><div class="pstat-n">3</div><div class="pstat-l">مستويات ازدحام</div></div>
      <div class="pstat"><div class="pstat-n">7</div><div class="pstat-l">أيام مقارنة تلقائية</div></div>
      <div class="pstat" style="font-size:13px;"><div class="pstat-n" style="font-size:18px;">XGBoost</div><div class="pstat-l">نموذج التنبؤ</div></div>
    </div>
    """)

    # Feature cards
    c1, c2, c3 = st.columns(3)
    cards = [
        ("01", "مؤشر الازدحام",
         "يعرض مستوى الازدحام المتوقع لليوم المختار على ثلاثة مستويات: منخفض، متوسط، مرتفع."),
        ("02", "توقع عددي دقيق",
         "يقدّر عدد المعتمرين المتوقع تواجدهم في اليوم والشهر الهجري الذي تحدده."),
        ("03", "توصية ذكية",
         "يقترح النظام تلقائيًا اليوم الأنسب من الأيام السبعة القريبة عند الحاجة."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            H(f"""
            <div class="feat-card">
              <div class="feat-icon">{esc(icon)}</div>
              <div class="feat-title">{esc(title)}</div>
              <div class="feat-desc">{esc(desc)}</div>
            </div>
            """)

    H("<div style='height:22px;'></div>")
    _, mid, _ = st.columns([1, 1.05, 1])
    with mid:
        if st.button("بدء التنبؤ الآن", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
    render_footer()


def input_page():
    show_header()
    render_nav()

    df_dates = load_data()

    H("""
    <div class="form-page-header">
      <div class="form-step-badge">خطوة الإدخال</div>
      <div class="form-page-title">تحديد موعد الزيارة</div>
      <div class="form-page-sub">أدخل بياناتك لمعرفة مستوى الازدحام المتوقع وأفضل وقت للزيارة</div>
      <div class="form-rule"></div>
    </div>
    """)

    with st.container():
        H('<div class="form-shell">')
        H('<div class="form-section-label">معلومات المعتمر</div>')
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("الاسم الكامل", placeholder="أدخل اسم المعتمر")
        with c2:
            nationality = st.selectbox("الجنسية", NATIONALITY_OPTIONS, index=0)

        H("<div style='height:6px;'></div>")
        H('<div class="form-section-label">التاريخ الهجري المقترح</div>')
        c3, c4 = st.columns(2)
        with c3:
            month = st.selectbox("الشهر الهجري", HIJRI_MONTHS, index=10, key="month_selector")

        available_days = get_available_days(df_dates, month)
        if not available_days:
            available_days = list(range(1, 31))

        with c4:
            day = st.selectbox("اليوم الهجري", available_days, index=0,
                               key=f"day_selector_{month}")
        H("</div>")  # close form-shell

    blocked = (month == "ذو القعدة" and nationality == "غير سعودي" and 1 <= int(day) <= 15)
    if blocked:
        st.error("لا يمكن عرض النتائج لهذا الاختيار؛ الجنسية غير سعودية خلال الفترة ١–١٥ ذو القعدة.")

    H("<div style='height:18px;'></div>")
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        submitted = st.button("عرض التوقع", use_container_width=True)

    if submitted:
        if not name.strip():
            st.warning("الرجاء إدخال الاسم الكامل.")
            return
        if blocked:
            return
        st.session_state.entered              = True
        st.session_state.show_best_day        = False
        st.session_state.selected_name        = name.strip()
        st.session_state.selected_nationality = nationality
        st.session_state.selected_month       = normalize_month_name(month)
        st.session_state.selected_day         = int(day)
        st.session_state.page                 = "dashboard"
        st.rerun()

    render_footer()


def dashboard_page():
    show_header()
    render_nav()

    if not st.session_state.entered:
        st.warning("الرجاء إدخال البيانات أولًا.")
        if st.button("الذهاب إلى إدخال البيانات"):
            st.session_state.page = "input"
            st.rerun()
        return

    month = normalize_month_name(st.session_state.selected_month)
    day   = int(st.session_state.selected_day)

    show_hajj  = should_show_hajj_count(month, day)
    show_tawaf = should_show_tawaf_ifadah(month, day)

    df  = load_data()
    df7 = get_7_days(df, month, day)

    if df7.empty:
        st.error("لا توجد بيانات مطابقة لليوم أو الشهر المختار داخل ملف المودل.")
        return

    sel_df = df7[
        (df7[MONTH_COL].astype(str).str.strip() == str(month).strip()) &
        (df7["Hijri_Day_Num"] == int(day))
    ]
    if sel_df.empty:
        st.error("لا توجد بيانات لليوم المختار داخل ملف المودل.")
        return

    row        = sel_df.iloc[0]
    weekday    = row.get("Weekday_AR", "غير محدد")
    hijri_date = row.get("Hijri_Date", f"{day} {month}")
    prediction = float(row.get("Prediction", 0))
    level      = normalize_level(row.get("Local_Crowding_Level", "متوسط"))
    decision   = get_decision(level)
    reason     = get_reason(decision)
    temp       = row.get("AvgTemp_C", np.nan)
    temp_text  = "—" if pd.isna(temp) else f"{float(temp):.1f}°م"
    hajj_count  = format_seasonal_count(row.get("Hajj", 0))
    tawaf_count = format_seasonal_count(row.get("Tawaf_Ifadah", 0))
    best_day    = get_best_day(df7, day, month)
    pred_text   = format_number(prediction)
    level_color = LEVEL_COLORS.get(level, "#1A9B6C")
    decision_color = level_color

    # ── FULL-WIDTH RESULT BANNER ─────────────────────────────────────────
    H(f"""
    <div class="result-banner">
      <div class="rb-inner rb-date">
        <div class="rb-date-weekday">اليوم المختار</div>
        <div class="rb-date-hijri">{esc(weekday)}</div>
        <div class="rb-date-month">{esc(str(hijri_date))}</div>
      </div>

      <div class="rb-inner rb-level-col">
        <div>
          <span class="rb-level-eyebrow">
            <span class="rb-level-dot"></span>
            مستوى الازدحام المتوقع
          </span>
        </div>
        <div class="rb-level-text" style="color:{level_color};">{esc(level)}</div>
        <div class="rb-level-label">{esc(decision)}</div>
      </div>

      <div class="rb-inner rb-stats">
        <div class="rb-stat-item">
          <div class="rb-stat-label">العدد المتوقع</div>
          <div class="rb-stat-value">{esc(pred_text)}</div>
          <div class="rb-stat-unit">معتمر</div>
        </div>
        <div class="rb-stat-item">
          <div class="rb-stat-label">متوسط الحرارة</div>
          <div class="rb-stat-value">{esc(temp_text)}</div>
        </div>
      </div>
    </div>
    """)

    # Seasonal pill (if applicable)
    if show_hajj or show_tawaf:
        tag  = "حج"   if show_hajj  else "طواف"
        text = f"عدد الحجاج المتوقع: {hajj_count}" if show_hajj else f"عدد طواف الإفاضة المتوقع: {tawaf_count}"
        H(f"""
        <div class="info-pill-wrap">
          <div class="info-pill">
            <span class="info-pill-tag">{esc(tag)}</span>
            <span>{esc(text)}</span>
          </div>
        </div>
        """)

    H("<div style='height:14px;'></div>")

    # ── MAIN CONTENT: Chart (left) | Action panel (right) ───────────────
    chart_col, action_col = st.columns([1.72, 1], gap="large")

    with chart_col:
        H(f"""
        <div class="chart-container">
          <div class="chart-header">
            <div class="chart-title">توقع حجم المعتمرين — الأيام السبعة القريبة</div>
            <div class="chart-legend-hint">
              <div class="clh-item">
                <div class="clh-dot" style="background:#1A9B6C;"></div>منخفض
              </div>
              <div class="clh-item">
                <div class="clh-dot" style="background:#C8921E;"></div>متوسط
              </div>
              <div class="clh-item">
                <div class="clh-dot" style="background:#D05045;"></div>مرتفع
              </div>
            </div>
          </div>
          <div class="chart-rule"></div>
        </div>
        """)
        fig = build_chart(df7, month, day)
        st.plotly_chart(fig, use_container_width=True)

    with action_col:
        # Recommendation card
        H(f"""
        <div class="action-panel">
          <div class="reco-card">
            <div class="reco-eyebrow">التوصية النهائية للزيارة</div>
            <div class="reco-decision" style="color:{decision_color};">{esc(decision)}</div>
            <div class="reco-ornament">◆</div>
            <div class="reco-text">{esc(reason)}</div>
          </div>
        """)

        # Alternative day panel (always shown, button reveals result)
        H(f"""
          <div class="alt-panel">
            <div class="alt-panel-inner">
              <div class="alt-title">اقتراح يوم بديل</div>
              <div class="alt-sub">
                يقارن النظام الأيام السبعة المجاورة ويقترح
                اليوم الأقل ازدحامًا تلقائيًا.
              </div>
            </div>
          </div>
        </div>
        """)

        H("<div style='height:6px;'></div>")
        if st.button("اعرض اليوم الأنسب", use_container_width=True):
            st.session_state.show_best_day = True

        if st.session_state.show_best_day:
            if best_day is None:
                H("""
                <div class="best-day-result">
                  <div class="bdr-day">لا يوجد يوم بديل متاح ضمن الأيام السبعة القريبة</div>
                </div>
                """)
            else:
                bday_name    = esc(best_day.get("Weekday_AR", "غير متاح"))
                bday_hijri   = esc(best_day.get("Hijri_Date", ""))
                bday_visitors = format_number(best_day.get("Prediction", 0))
                H(f"""
                <div class="best-day-result">
                  <div class="bdr-badge">اليوم الأنسب للزيارة</div>
                  <div class="bdr-day">{bday_name}</div>
                  <div class="bdr-meta">
                    التاريخ الهجري: <strong>{bday_hijri}</strong><br>
                    العدد المتوقع: <strong>{bday_visitors}</strong>
                  </div>
                  <div class="bdr-note">
                    تم اختياره لأنه الأقل ازدحامًا ضمن الأيام السبعة القريبة.
                  </div>
                </div>
                """)

    H("<div style='height:14px;'></div>")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        if st.button("إدخال تاريخ جديد", use_container_width=True):
            st.session_state.page          = "input"
            st.session_state.show_best_day = False
            st.rerun()

    render_footer()


# ── Router ────────────────────────────────────────────────────────────────────

if   st.session_state.page == "home":      home_page()
elif st.session_state.page == "input":     input_page()
else:                                       dashboard_page()
