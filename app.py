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


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def english_level(level):
    level = normalize_level(level)
    return {"منخفض": "Low", "متوسط": "Medium", "مرتفع": "High"}.get(level, "Medium")


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
    else:
        return "مناسب مع الحذر"


def get_reason(decision):
    if decision == "مناسب للزيارة":
        return "من المتوقع أن يكون مستوى الازدحام منخفضًا مقارنة بالأيام القريبة، لذلك يعد هذا اليوم خيارًا مناسبًا لأداء العمرة."
    if decision == "يفضل اختيار يوم آخر":
        return "الازدحام المتوقع مرتفع مقارنةً بالأيام القريبة، لذلك يفضل اختيار يوم أقل ازدحامًا."
    return "الازدحام المتوقع ضمن المستوى المتوسط، لذلك يمكن أداء العمرة مع الحذر واختيار الوقت المناسب."


# ── Chart ───────────────────────────────────────────────────────────────────

def build_chart(df7, selected_month=None, selected_day=None):
    cdf = df7.copy().reset_index(drop=True)
    cdf["Prediction"] = pd.to_numeric(cdf["Prediction"], errors="coerce")
    cdf["Local_Crowding_Level"] = cdf["Local_Crowding_Level"].apply(normalize_level)
    cdf["x_label"] = (
        cdf["Weekday_AR"].astype(str) + "<br>" +
        cdf["Hijri_Day_Num"].astype(int).astype(str) + "-" +
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
    pad = max((y_max - y_min) * 0.28, 500)
    base_y = max(0, y_min - pad)

    fig = go.Figure()

    # Fill areas
    for i in range(len(cdf) - 1):
        x0, x1 = cdf.loc[i, "x_label"], cdf.loc[i + 1, "x_label"]
        y0, y1 = cdf.loc[i, "Prediction"], cdf.loc[i + 1, "Prediction"]
        l1 = normalize_level(cdf.loc[i, "Local_Crowding_Level"])
        l2 = normalize_level(cdf.loc[i + 1, "Local_Crowding_Level"])
        seg_color = LEVEL_COLORS["مرتفع"] if "مرتفع" in [l1, l2] else (
            LEVEL_COLORS["متوسط"] if "متوسط" in [l1, l2] else LEVEL_COLORS["منخفض"]
        )
        fig.add_trace(go.Scatter(
            x=[x0, x0, x1, x1], y=[base_y, y0, y1, base_y],
            mode="lines", fill="toself",
            fillcolor=hex_to_rgba(seg_color, 0.14),
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip", showlegend=False
        ))

    # Line segments
    for i in range(len(cdf) - 1):
        x0, x1 = cdf.loc[i, "x_label"], cdf.loc[i + 1, "x_label"]
        y0, y1 = cdf.loc[i, "Prediction"], cdf.loc[i + 1, "Prediction"]
        l1 = normalize_level(cdf.loc[i, "Local_Crowding_Level"])
        l2 = normalize_level(cdf.loc[i + 1, "Local_Crowding_Level"])
        seg_color = LEVEL_COLORS["مرتفع"] if "مرتفع" in [l1, l2] else (
            LEVEL_COLORS["متوسط"] if "متوسط" in [l1, l2] else LEVEL_COLORS["منخفض"]
        )
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=seg_color, width=4, shape="spline"),
            hoverinfo="skip", showlegend=False
        ))

    # Points by level
    for level in ["منخفض", "متوسط", "مرتفع"]:
        level_df = cdf[cdf["Local_Crowding_Level"] == level].copy()
        fig.add_trace(go.Scatter(
            x=level_df["x_label"], y=level_df["Prediction"],
            mode="markers+text",
            marker=dict(size=14, color=LEVEL_COLORS[level], line=dict(color="white", width=2.5)),
            text=level_df["Prediction"].apply(
                lambda x: f"{int(round(float(x))):,}" if not pd.isna(x) else ""
            ),
            textposition="top center",
            textfont=dict(size=11, color="#053D33", family="Cairo"),
            hovertemplate="<b>%{x}</b><br>العدد المتوقع: %{y:,.0f}<extra></extra>",
            name=level
        ))

    # Selected day highlight
    if selected_mask.any():
        sp = cdf[selected_mask].iloc[[0]]
        fig.add_trace(go.Scatter(
            x=sp["x_label"], y=sp["Prediction"],
            mode="markers",
            marker=dict(size=44, color="rgba(196,144,63,0.14)",
                        line=dict(color="rgba(196,144,63,0.22)", width=2)),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=sp["x_label"], y=sp["Prediction"],
            mode="markers+text",
            marker=dict(size=22, color="#FFF8EC", line=dict(color="#C4903F", width=4)),
            text=["اليوم المختار"],
            textposition="bottom center",
            textfont=dict(size=11, color="#062B25", family="Cairo"),
            hovertemplate="<b>اليوم المختار</b><br>%{x}<br>العدد المتوقع: %{y:,.0f}<extra></extra>",
            showlegend=False
        ))

    q1 = cdf["Prediction"].quantile(0.33)
    q2 = cdf["Prediction"].quantile(0.66)
    fig.add_hline(y=q1, line_dash="dot", line_color="rgba(26,155,108,0.25)", line_width=1)
    fig.add_hline(y=q2, line_dash="dot", line_color="rgba(200,146,30,0.25)", line_width=1)

    fig.update_layout(
        height=380,
        margin=dict(l=16, r=16, t=32, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(family="Cairo", size=12, color="#082E28"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0,
            title_text="مستوى الازدحام:",
            title_font=dict(size=11),
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(showgrid=False, title="",
                   tickfont=dict(family="Cairo", size=11)),
        yaxis=dict(
            range=[base_y, y_max + pad],
            showgrid=True, gridcolor="rgba(120,100,60,0.07)",
            tickformat=",",
            title=dict(text="عدد المعتمرين", font=dict(size=10))
        ),
        hoverlabel=dict(bgcolor="#052A24", bordercolor="#C4903F",
                        font=dict(color="#FFF8EC", family="Cairo", size=12))
    )
    return fig


# ── CSS ──────────────────────────────────────────────────────────────────────

H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800;900&display=swap');

:root {
  --primary: #006C4E;
  --primary-dark: #014D40;
  --primary-soft: #EAF5F1;
  --mint: #F6FBF9;
  --bg: #F7F8F6;
  --surface: #FFFFFF;
  --surface-2: #FBFCFB;
  --border: #DDE7E2;
  --border-2: #EDF1EF;
  --text: #172B24;
  --muted: #61736B;
  --muted-2: #8A9992;
  --warning: #B7791F;
  --danger: #B42318;
  --shadow: 0 10px 28px rgba(21, 35, 31, 0.055);
  --shadow-2: 0 18px 42px rgba(21, 35, 31, 0.075);
}

html, body, [class*="css"] {
  font-family: 'Cairo', sans-serif !important;
  direction: rtl;
}

#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
.js-plotly-plot .plotly .modebar { display: none !important; }

.stApp {
  background:
    linear-gradient(180deg, rgba(0,108,78,0.035) 0%, rgba(247,248,246,0.0) 260px),
    var(--bg);
  color: var(--text);
}

.block-container {
  max-width: 1220px !important;
  padding-top: 1.1rem !important;
  padding-bottom: 2.2rem !important;
}

div[data-testid="stHorizontalBlock"] {
  gap: 1rem !important;
  align-items: stretch !important;
}

/* Streamlit controls */
.stButton button {
  background: var(--surface) !important;
  color: var(--primary-dark) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  min-height: 42px !important;
  font-size: 13px !important;
  font-weight: 800 !important;
  box-shadow: none !important;
  transition: all .16s ease !important;
}
.stButton button:hover {
  background: var(--primary) !important;
  color: white !important;
  border-color: var(--primary) !important;
  transform: translateY(-1px);
}
.stButton button:focus, .stButton button:active {
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(0,108,78,.12) !important;
}

.stTextInput label, .stSelectbox label {
  color: var(--text) !important;
  font-weight: 800 !important;
  font-size: 13px !important;
}
.stTextInput input {
  min-height: 48px !important;
  border-radius: 10px !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
  font-family: 'Cairo', sans-serif !important;
  color: var(--text) !important;
}
.stTextInput input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(0,108,78,.10) !important;
}
.stSelectbox > div > div {
  min-height: 48px !important;
  border-radius: 10px !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}

/* Government / enterprise shell */
.gov-shell {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 18px;
  box-shadow: var(--shadow);
  overflow: hidden;
  margin-bottom: 16px;
}
.gov-topbar {
  min-height: 76px;
  padding: 18px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  border-bottom: 1px solid var(--border-2);
  background: linear-gradient(90deg, #FFFFFF 0%, #F8FBF9 100%);
}
.gov-brand {
  display: flex;
  align-items: center;
  gap: 14px;
}
.gov-logo {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
  display: grid;
  place-items: center;
  color: #fff;
  font-size: 18px;
  font-weight: 900;
  box-shadow: 0 8px 18px rgba(0,108,78,0.18);
}
.gov-title {
  color: var(--text);
  font-size: 20px;
  font-weight: 900;
  letter-spacing: -0.3px;
  line-height: 1.35;
}
.gov-subtitle {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  margin-top: 3px;
}
.gov-status {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--primary-dark);
  font-size: 12px;
  font-weight: 800;
  background: var(--primary-soft);
  border: 1px solid #D6EAE3;
  border-radius: 999px;
  padding: 8px 14px;
  white-space: nowrap;
}
.gov-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--primary);
}

.nav-wrap {
  padding: 12px 14px;
  background: var(--surface);
  border-bottom: 1px solid var(--border-2);
}
.nav-wrap .stButton button {
  background: #FAFCFB !important;
  border-color: var(--border-2) !important;
}
.nav-wrap .stButton button:hover {
  background: var(--primary-dark) !important;
  color: white !important;
}

.page-section {
  padding: 20px;
}
.panel {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 22px;
}
.panel-soft {
  background: linear-gradient(180deg, #FFFFFF 0%, #FAFCFB 100%);
}
.section-title {
  color: var(--text);
  font-size: 18px;
  font-weight: 900;
  margin-bottom: 6px;
}
.section-subtitle {
  color: var(--muted);
  font-size: 12.5px;
  font-weight: 650;
  line-height: 1.8;
}
.thin-line {
  height: 1px;
  background: var(--border-2);
  margin: 16px 0;
}

/* Home */
.hero-clean {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #FFFFFF 0%, #F5FBF8 100%);
  border: 1px solid var(--border-2);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 34px 32px;
  margin-bottom: 16px;
}
.hero-clean:before {
  content: "";
  position: absolute;
  top: 0; right: 0; bottom: 0;
  width: 7px;
  background: var(--primary);
}
.hero-kicker {
  display: inline-block;
  color: var(--primary-dark);
  background: var(--primary-soft);
  border: 1px solid #D6EAE3;
  border-radius: 999px;
  padding: 7px 14px;
  font-size: 12px;
  font-weight: 900;
  margin-bottom: 14px;
}
.hero-title {
  color: var(--text);
  font-size: 34px;
  font-weight: 900;
  letter-spacing: -0.8px;
  line-height: 1.35;
}
.hero-gold {
  color: var(--primary);
  font-size: 14px;
  font-weight: 900;
  margin-top: 6px;
}
.hero-body {
  color: var(--muted);
  font-size: 14px;
  font-weight: 650;
  max-width: 760px;
  line-height: 2;
  margin-top: 12px;
}
.feat-card {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 20px;
  min-height: 146px;
  height: 100%;
}
.feat-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  color: var(--primary);
  background: var(--primary-soft);
  font-size: 13px;
  font-weight: 900;
  margin-bottom: 14px;
}
.feat-title { color: var(--text); font-size: 16px; font-weight: 900; }
.feat-desc { color: var(--muted); font-size: 12.5px; font-weight: 650; line-height: 1.9; margin-top: 8px; }

/* Form */
.form-shell {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 24px;
  margin-bottom: 16px;
}
.form-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-2);
  margin-bottom: 18px;
}
.form-title { color: var(--text); font-size: 22px; font-weight: 900; }
.form-sub { color: var(--muted); font-size: 12.5px; font-weight: 700; margin-top: 4px; }
.form-badge {
  color: var(--primary-dark);
  background: var(--primary-soft);
  border: 1px solid #D6EAE3;
  border-radius: 999px;
  padding: 7px 13px;
  font-size: 11px;
  font-weight: 900;
  white-space: nowrap;
}

/* Dashboard */
.summary-card {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 22px;
  height: 100%;
}
.summary-label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  margin-bottom: 8px;
}
.summary-level {
  font-size: 46px;
  font-weight: 900;
  line-height: 1.1;
  letter-spacing: -0.8px;
}
.summary-desc {
  color: var(--muted);
  font-size: 12.5px;
  font-weight: 650;
  line-height: 1.8;
  margin-top: 10px;
}
.summary-divider {
  height: 1px;
  background: var(--border-2);
  margin: 18px 0;
}
.summary-mini-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.summary-mini {
  background: #FAFCFB;
  border: 1px solid var(--border-2);
  border-radius: 12px;
  padding: 12px;
}
.summary-mini-label { color: var(--muted-2); font-size: 11px; font-weight: 800; margin-bottom: 4px; }
.summary-mini-value { color: var(--text); font-size: 16px; font-weight: 900; }

.metric-card {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: 16px;
  min-height: 112px;
  height: 100%;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.metric-icon {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  font-size: 16px;
  color: var(--primary);
  background: var(--primary-soft);
  flex: 0 0 auto;
}
.metric-body { flex: 1; }
.metric-label { color: var(--muted); font-size: 11.5px; font-weight: 850; margin-bottom: 6px; }
.metric-value { color: var(--text); font-size: 24px; font-weight: 900; line-height: 1.2; letter-spacing: -0.3px; }
.metric-sub { color: var(--muted-2); font-size: 11px; font-weight: 700; margin-top: 5px; }

.reco-card {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 20px 22px;
  margin-top: 14px;
}
.reco-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}
.reco-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--primary-dark);
  background: var(--primary-soft);
  border: 1px solid #D6EAE3;
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 11px;
  font-weight: 900;
}
.cc-badge-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--primary);
  display: inline-block;
}
.reco-decision { font-size: 24px; font-weight: 900; letter-spacing: -0.3px; }
.reco-text { color: var(--muted); font-size: 13.5px; font-weight: 650; line-height: 1.9; }

.suggest-card {
  background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
  border-radius: 16px;
  box-shadow: var(--shadow-2);
  padding: 22px;
  min-height: 164px;
  color: white;
}
.suggest-title { color: #fff; font-size: 20px; font-weight: 900; }
.suggest-sub { color: rgba(255,255,255,0.82); font-size: 12.5px; font-weight: 650; line-height: 1.9; margin-top: 8px; }
.best-day-card {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: 16px;
  margin-top: 10px;
}
.bdc-badge { display: inline-block; color: var(--primary-dark); background: var(--primary-soft); border-radius: 999px; padding: 5px 10px; font-size: 10px; font-weight: 900; margin-bottom: 8px; }
.bdc-day { color: var(--text); font-size: 22px; font-weight: 900; }
.bdc-meta { color: var(--muted); font-size: 12px; font-weight: 700; line-height: 1.9; margin-top: 6px; }
.bdc-meta strong { color: var(--text); }
.bdc-note { color: var(--primary); font-size: 11px; font-weight: 800; margin-top: 8px; line-height: 1.7; }

.chart-card {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 18px 18px 4px;
}
.chart-title { color: var(--text); font-size: 16px; font-weight: 900; }
.chart-subtitle { color: var(--muted); font-size: 11.5px; font-weight: 700; margin-top: 4px; }
.info-pill {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 12px;
  background: var(--surface);
  border: 1px solid var(--border-2);
  color: var(--text);
  font-size: 12px;
  font-weight: 800;
  box-shadow: var(--shadow);
}
.pill-tag {
  color: var(--primary-dark);
  background: var(--primary-soft);
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 10px;
  font-weight: 900;
}

@media (max-width: 900px) {
  .gov-topbar { align-items: flex-start; flex-direction: column; }
  .hero-title { font-size: 26px; }
  .summary-level { font-size: 36px; }
  .summary-mini-grid { grid-template-columns: 1fr; }
  .metric-value { font-size: 20px; }
}
</style>
""")


# ── Page components ──────────────────────────────────────────────────────────

def show_header():
    H("""
    <div class="gov-shell">
      <div class="gov-topbar">
        <div class="gov-brand">
          <div class="gov-logo">ح</div>
          <div>
            <div class="gov-title">المنصة الذكية لتوقع الازدحام</div>
            <div class="gov-subtitle">اختيار الوقت الأنسب لأداء العمرة</div>
          </div>
        </div>
        <div class="gov-status"><span class="gov-dot"></span> نظام تنبؤ تفاعلي</div>
      </div>
    </div>
    """)


def render_nav():
    H('<div class="nav-wrap">')
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("الرئيسية", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with c2:
        if st.button("إدخال البيانات", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
    with c3:
        if st.button("لوحة النتائج", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    H('</div><div style="height:14px;"></div>')


def crowding_hero_card(level, prediction_text, weekday, hijri_date):
    level = normalize_level(level)
    color = LEVEL_COLORS.get(level, "#006C4E")
    H(f"""
    <div class="summary-card panel-soft">
      <div class="summary-label">مستوى الازدحام المتوقع</div>
      <div class="summary-level" style="color:{color};">{esc(level)}</div>
      <div class="summary-desc">يعرض هذا المؤشر مستوى الازدحام المتوقع لليوم المحدد بناءً على بيانات النموذج.</div>
      <div class="summary-divider"></div>
      <div class="summary-mini-grid">
        <div class="summary-mini">
          <div class="summary-mini-label">العدد المتوقع</div>
          <div class="summary-mini-value">{esc(prediction_text)}</div>
        </div>
        <div class="summary-mini">
          <div class="summary-mini-label">اليوم المختار</div>
          <div class="summary-mini-value">{esc(weekday)}</div>
        </div>
      </div>
    </div>
    """)


def metric_card(label, value, sub, icon):
    H(f"""
    <div class="metric-card">
      <div class="metric-body">
        <div class="metric-label">{esc(label)}</div>
        <div class="metric-value">{esc(value)}</div>
        <div class="metric-sub">{esc(sub)}</div>
      </div>
      <div class="metric-icon">{esc(icon)}</div>
    </div>
    """)


def reco_card(decision, reason, level):
    level = normalize_level(level)
    color = LEVEL_COLORS.get(level, "#006C4E")
    H(f"""
    <div class="reco-card">
      <div class="reco-top">
        <div class="reco-badge"><span class="cc-badge-dot"></span> التوصية النهائية</div>
        <div class="reco-decision" style="color:{color};">{esc(decision)}</div>
      </div>
      <div class="reco-text">{esc(reason)}</div>
    </div>
    """)


def best_day_card(row):
    if row is None:
        H("""
        <div class="best-day-card">
          <div class="bdc-day">لا يوجد يوم بديل ضمن الأيام السبعة القريبة</div>
        </div>
        """)
        return
    day_name = esc(row.get("Weekday_AR", "غير متاح"))
    hijri_date = esc(row.get("Hijri_Date", ""))
    visitors = format_number(row.get("Prediction", 0))
    H(f"""
    <div class="best-day-card">
      <div class="bdc-badge">اليوم الأنسب للزيارة</div>
      <div class="bdc-day">{day_name}</div>
      <div class="bdc-meta">
        التاريخ الهجري: <strong>{hijri_date}</strong><br>
        العدد المتوقع: <strong>{visitors}</strong>
      </div>
      <div class="bdc-note">تم اختياره لأنه الأقل ازدحامًا ضمن الأيام السبعة القريبة.</div>
    </div>
    """)


def info_pill(text, tag):
    H(f"""
    <div class="info-pill">
      <span class="pill-tag">{esc(tag)}</span>
      <span>{esc(text)}</span>
    </div>
    """)


# ── Session state ────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "home"
if "entered" not in st.session_state:
    st.session_state.entered = False
if "show_best_day" not in st.session_state:
    st.session_state.show_best_day = False


# ── Pages ────────────────────────────────────────────────────────────────────

def home_page():
    show_header()
    render_nav()

    H("""
    <div class="hero-clean">
      <div class="hero-kicker">نظام تنبؤ تفاعلي</div>
      <div class="hero-title">مرحباً بك في نظام التنبؤ بازدحام المعتمرين</div>
      <div class="hero-gold">Umrah Visitors Smart Forecasting System</div>
      <div class="hero-body">
        يعرض النظام العدد المتوقع للمعتمرين ومستوى الازدحام والتوصية المناسبة
        لمساعدتك في اختيار أنسب وقت لأداء العمرة.
      </div>
    </div>
    """)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("01", "مؤشر الازدحام", "عرض مستوى الازدحام لليوم المختار بشكل واضح وفوري."),
        ("02", "توقع عددي",    "تقدير العدد المتوقع للمعتمرين في اليوم المحدد."),
        ("03", "توصية ذكية",   "اقتراح يوم بديل أقل ازدحامًا عند الحاجة."),
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

    H("<div style='height:20px;'></div>")
    _, mid, _ = st.columns([1, 1.05, 1])
    with mid:
        if st.button("بدء التنبؤ", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()


def input_page():
    show_header()
    render_nav()

    df_dates = load_data()

    H("""
    <div class="form-shell">
      <div class="form-header">
        <div>
          <div class="form-title">تحديد موعد العمرة</div>
          <div class="form-sub">اختر تاريخ الزيارة لمعرفة مستوى الازدحام المتوقع</div>
        </div>
        <div class="form-badge">بيانات الزيارة</div>
      </div>
    </div>
    """)

    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("الاسم الكامل", placeholder="أدخل اسم المعتمر")
    with c2:
        nationality = st.selectbox("الجنسية", NATIONALITY_OPTIONS, index=0)

    c3, c4 = st.columns(2)
    with c3:
        month = st.selectbox("الشهر الهجري", HIJRI_MONTHS, index=10, key="month_selector")

    available_days = get_available_days(df_dates, month)
    if not available_days:
        available_days = list(range(1, 31))

    with c4:
        day = st.selectbox("التاريخ الهجري", available_days, index=0, key=f"day_selector_{month}")

    blocked = (month == "ذو القعدة" and nationality == "غير سعودي" and 1 <= int(day) <= 15)
    if blocked:
        st.error("لا يمكن عرض النتائج لهذا الاختيار؛ لأن الجنسية غير سعودي خلال الفترة من 1 إلى 15 ذو القعدة.")

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
        st.session_state.entered = True
        st.session_state.show_best_day = False
        st.session_state.selected_name = name.strip()
        st.session_state.selected_nationality = nationality
        st.session_state.selected_month = normalize_month_name(month)
        st.session_state.selected_day = int(day)
        st.session_state.page = "dashboard"
        st.rerun()


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

    show_hajj   = should_show_hajj_count(month, day)
    show_tawaf  = should_show_tawaf_ifadah(month, day)

    df   = load_data()
    df7  = get_7_days(df, month, day)

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
    hijri_date = row.get("Hijri_Date", f"{day}-{month}")
    prediction = float(row.get("Prediction", 0))
    level      = normalize_level(row.get("Local_Crowding_Level", "متوسط"))
    decision   = get_decision(level)
    reason     = get_reason(decision)

    temp       = row.get("AvgTemp_C", np.nan)
    temp_text  = "غير متاحة" if pd.isna(temp) else f"{float(temp):.1f}°C"

    hajj_count  = format_seasonal_count(row.get("Hajj", 0))
    tawaf_count = format_seasonal_count(row.get("Tawaf_Ifadah", 0))
    best_day    = get_best_day(df7, day, month)
    pred_text   = format_number(prediction)

    top_col, metrics_col = st.columns([1.25, 1.75], gap="large")
    with top_col:
        crowding_hero_card(level, pred_text, weekday, hijri_date)
    with metrics_col:
        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("اليوم المختار",  weekday,   hijri_date, "📅")
        with m2:
            metric_card("العدد المتوقع",  pred_text, "معتمر",    "👥")
        with m3:
            metric_card("درجة الحرارة",   temp_text, "متوسط اليوم", "🌡")
        reco_card(decision, reason, level)

    if show_hajj or show_tawaf:
        H("<div style='height:12px;'></div>")
        _, s1, _ = st.columns([1, 1.4, 1])
        with s1:
            if show_hajj:
                info_pill(f"عدد الحجاج المتوقع: {hajj_count}", "حج")
            elif show_tawaf:
                info_pill(f"عدد طواف الإفاضة المتوقع: {tawaf_count}", "طواف")

    H("<div style='height:14px;'></div>")
    sug_col, chart_col = st.columns([0.85, 2.15], gap="large")

    with sug_col:
        H("""
        <div class="suggest-card">
          <div class="suggest-title">اقتراح يوم بديل</div>
          <div class="suggest-sub">
            يقارن النظام الأيام السبعة القريبة ويقترح
            الأقل ازدحامًا عند الحاجة.
          </div>
        </div>
        """)
        H("<div style='height:10px;'></div>")
        if st.button("اقتراح يوم أنسب", use_container_width=True):
            st.session_state.show_best_day = True
        if st.session_state.show_best_day:
            best_day_card(best_day)

    with chart_col:
        H("""
        <div class="chart-card">
          <div class="chart-title">الاتجاه المتوقع خلال الأيام السبعة القريبة</div>
          <div class="chart-subtitle">مقارنة بصرية لمستوى الازدحام المتوقع حول التاريخ المختار</div>
        """)
        fig = build_chart(df7, month, day)
        st.plotly_chart(fig, use_container_width=True)
        H("</div>")

    H("<div style='height:14px;'></div>")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        if st.button("إدخال تاريخ جديد", use_container_width=True):
            st.session_state.page = "input"
            st.session_state.show_best_day = False
            st.rerun()


# ── Router ───────────────────────────────────────────────────────────────────

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "input":
    input_page()
else:
    dashboard_page()
