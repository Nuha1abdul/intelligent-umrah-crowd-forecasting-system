%%writefile app.py
from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="نظام التنبؤ بازدحام المعتمرين",
    page_icon="🕋",
    layout="wide",
    initial_sidebar_state="expanded"
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
    "منخفض": "#159a5c",
    "متوسط": "#c8921e",
    "مرتفع": "#d2554a"
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


def H(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def esc(x):
    if pd.isna(x):
        return ""
    return html.escape(str(x))


def normalize_month_name(month_name):
    text = str(month_name).strip()

    mapping = {
        "محرم": "محرم",
        "صفر": "صفر",
        "ربيع الأول": "ربيع الأول",
        "ربيع الاول": "ربيع الأول",
        "ربيع اول": "ربيع الأول",
        "ربيع الآخر": "ربيع الآخر",
        "ربيع الاخر": "ربيع الآخر",
        "ربيع الثاني": "ربيع الآخر",
        "جمادى الأولى": "جماد الأول",
        "جمادى الاولى": "جماد الأول",
        "جماد الأولى": "جماد الأول",
        "جماد الاولى": "جماد الأول",
        "جماد الأول": "جماد الأول",
        "جماد الاول": "جماد الأول",
        "جمادى الآخرة": "جماد الآخر",
        "جمادى الاخرة": "جماد الآخر",
        "جمادى الثانية": "جماد الآخر",
        "جمادى الثاني": "جماد الآخر",
        "جماد الآخرة": "جماد الآخر",
        "جماد الاخرة": "جماد الآخر",
        "جماد الآخر": "جماد الآخر",
        "جماد الاخر": "جماد الآخر",
        "جماد الثاني": "جماد الآخر",
        "رجب": "رجب",
        "شعبان": "شعبان",
        "رمضان": "رمضان",
        "شوال": "شوال",
        "ذو القعدة": "ذو القعدة",
        "ذو القعده": "ذو القعدة",
        "ذو الحجة": "ذو الحجة",
        "ذو الحجه": "ذو الحجة",
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

    out = df[
        (df[DATE_COL] >= start_date) &
        (df[DATE_COL] <= end_date)
    ].copy()

    out = out.sort_values(DATE_COL).reset_index(drop=True)
    out["Local_Crowding_Level"] = classify_by_quantiles(out["Prediction"])
    out["Local_Crowding_Level"] = out["Local_Crowding_Level"].apply(normalize_level)

    return out


def get_available_days(df, month):
    month = normalize_month_name(month)

    days = (
        df[
            df[MONTH_COL].astype(str).str.strip() == str(month).strip()
        ]["Hijri_Day_Num"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
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


def build_chart(df7):
    cdf = df7.copy().reset_index(drop=True)
    cdf["Prediction"] = pd.to_numeric(cdf["Prediction"], errors="coerce")
    cdf["Local_Crowding_Level"] = cdf["Local_Crowding_Level"].apply(normalize_level)

    cdf["x_label"] = (
        cdf["Weekday_AR"].astype(str) + "<br>" +
        cdf["Hijri_Day_Num"].astype(int).astype(str) + "-" +
        cdf[MONTH_COL].astype(str)
    )

    y_min = cdf["Prediction"].min()
    y_max = cdf["Prediction"].max()
    pad = max((y_max - y_min) * 0.25, 450)
    base_y = max(0, y_min - pad)

    fig = go.Figure()

    for i in range(len(cdf) - 1):
        x0 = cdf.loc[i, "x_label"]
        x1 = cdf.loc[i + 1, "x_label"]
        y0 = cdf.loc[i, "Prediction"]
        y1 = cdf.loc[i + 1, "Prediction"]

        level_1 = normalize_level(cdf.loc[i, "Local_Crowding_Level"])
        level_2 = normalize_level(cdf.loc[i + 1, "Local_Crowding_Level"])

        if "مرتفع" in [level_1, level_2]:
            seg_color = LEVEL_COLORS["مرتفع"]
        elif "متوسط" in [level_1, level_2]:
            seg_color = LEVEL_COLORS["متوسط"]
        else:
            seg_color = LEVEL_COLORS["منخفض"]

        fig.add_trace(go.Scatter(
            x=[x0, x0, x1, x1],
            y=[base_y, y0, y1, base_y],
            mode="lines",
            fill="toself",
            fillcolor=hex_to_rgba(seg_color, 0.12),
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip",
            showlegend=False
        ))

    for i in range(len(cdf) - 1):
        x0 = cdf.loc[i, "x_label"]
        x1 = cdf.loc[i + 1, "x_label"]
        y0 = cdf.loc[i, "Prediction"]
        y1 = cdf.loc[i + 1, "Prediction"]

        level_1 = normalize_level(cdf.loc[i, "Local_Crowding_Level"])
        level_2 = normalize_level(cdf.loc[i + 1, "Local_Crowding_Level"])

        if "مرتفع" in [level_1, level_2]:
            seg_color = LEVEL_COLORS["مرتفع"]
        elif "متوسط" in [level_1, level_2]:
            seg_color = LEVEL_COLORS["متوسط"]
        else:
            seg_color = LEVEL_COLORS["منخفض"]

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=seg_color, width=4, shape="spline"),
            hoverinfo="skip",
            showlegend=False
        ))

    for level in ["منخفض", "متوسط", "مرتفع"]:
        level_df = cdf[cdf["Local_Crowding_Level"] == level].copy()

        fig.add_trace(go.Scatter(
            x=level_df["x_label"],
            y=level_df["Prediction"],
            mode="markers+text",
            marker=dict(
                size=16,
                color=LEVEL_COLORS[level],
                line=dict(color="white", width=3)
            ),
            text=level_df["Prediction"].apply(
                lambda x: f"{int(round(float(x))):,}" if not pd.isna(x) else ""
            ),
            textposition="top center",
            textfont=dict(size=12, color="#064b3b"),
            hovertemplate="<b>%{x}</b><br>المعتمرون المتوقعون: %{y:,.0f}<extra></extra>",
            name=level
        ))

    q1 = cdf["Prediction"].quantile(0.33)
    q2 = cdf["Prediction"].quantile(0.66)

    fig.add_hline(y=q1, line_dash="dot", line_color="rgba(21,154,92,0.30)", line_width=1)
    fig.add_hline(y=q2, line_dash="dot", line_color="rgba(200,146,30,0.30)", line_width=1)

    fig.update_layout(
        height=360,
        margin=dict(l=15, r=15, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,253,247,0.94)",
        font=dict(family="Cairo", size=12, color="#064b3b"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0.01,
            title="مستوى الازدحام",
            font=dict(size=11)
        ),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(
            range=[base_y, y_max + pad],
            showgrid=True,
            gridcolor="rgba(120,100,60,0.10)",
            tickformat=",",
            title=dict(text="عدد المعتمرين", font=dict(size=11))
        )
    )

    return fig


def top_card(title, value, subtitle, icon, level=None):
    if level is None:
        color = "#064b3b"
        bg = "rgba(255,255,255,0.90)"
        border = "#e5d4aa"
        badge_bg = "#eef3ea"
        badge_text = "#0a6b52"
        shadow = "rgba(105,84,35,0.04)"
    else:
        level = normalize_level(level)
        color = LEVEL_COLORS.get(level, "#064b3b")
        bg = LEVEL_BG.get(level, "#eef8f1")
        border = LEVEL_BORDER.get(level, "#b9dec6")
        badge_bg = bg
        badge_text = color
        shadow = hex_to_rgba(color, 0.20)

    H(f"""
    <div class="top-card" style="background:{bg}; border-color:{border}; box-shadow:0 8px 18px {shadow};">
        <div class="top-icon">{esc(icon)}</div>
        <div class="top-title">{esc(title)}</div>
        <div class="top-value" style="color:{color};">{esc(value)}</div>
        <div class="top-badge" style="background:{badge_bg}; color:{badge_text}; box-shadow:0 5px 12px {shadow};">{esc(subtitle)}</div>
    </div>
    """)


def info_pill(text, icon="👥"):
    H(f"""
    <div class="small-pill">
        <span class="pill-icon">{esc(icon)}</span>
        <span>{esc(text)}</span>
    </div>
    """)


def best_day_card(row):
    if row is None:
        H("""
        <div class="best-inline-card">
            <div class="best-inline-head">لا يوجد يوم مقترح ضمن الأيام السبعة</div>
        </div>
        """)
        return

    day_name = esc(row.get("Weekday_AR", "غير متاح"))
    hijri_date = esc(row.get("Hijri_Date", ""))
    visitors = format_number(row.get("Prediction", 0))

    H(f"""
    <div class="best-inline-card">
        <div class="best-inline-head">
            <span class="best-check">✅</span>
            <span>اليوم الأفضل المقترح: <strong>{day_name}</strong></span>
        </div>
        <div class="best-inline-meta">
            <span>التاريخ الهجري: <strong>{hijri_date}</strong></span>
            <span class="best-dot">•</span>
            <span>المعتمرون المتوقعون: <strong>{visitors}</strong></span>
        </div>
    </div>
    """)


H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif !important;
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
    padding-bottom: 1rem !important;
}

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
    outline: none !important;
    box-shadow: none !important;
    border-radius: 14px !important;
    height: 40px !important;
    font-weight: 800 !important;
    font-size: 13px !important;
}

.stButton button:focus,
.stButton button:active {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
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
    padding: 24px;
    text-align:center;
    margin-bottom: 14px;
}

.hero-icon {
    font-size:34px;
}

.hero-title {
    color:#064b3b;
    font-size:22px;
    font-weight:900;
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
    font-size:13px;
    font-weight:700;
    margin-top:10px;
    line-height:1.8;
}

.feature-card {
    background: rgba(255,255,255,0.76);
    border: 1px solid rgba(180,155,96,0.28);
    border-radius: 16px;
    padding: 18px 12px;
    text-align:center;
    box-shadow: 0 4px 10px rgba(105,84,35,0.03);
    min-height: 120px;
}

.feature-icon {
    font-size:24px;
}

.feature-title {
    color:#064b3b;
    font-weight:900;
    font-size:15px;
    margin-top:8px;
}

.feature-desc {
    color:#705f39;
    font-weight:700;
    font-size:12px;
    margin-top:6px;
}

.top-card {
    background: rgba(255,255,255,0.90);
    border: 1px solid #e5d4aa;
    border-radius: 18px;
    padding: 12px 12px;
    min-height: 112px;
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
    gap:6px;
    font-size:10px;
    font-weight:800;
}

.pill-icon {
    width:22px;
    height:22px;
    border-radius:50%;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    background:#c79a2b;
    color:white;
    font-size:11px;
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
    font-size:14px;
    font-weight:800;
}

.section-line {
    width:68px;
    height:2px;
    background: linear-gradient(90deg, transparent, #d0a347, transparent);
    margin: 4px auto 8px auto;
}

.suggest-box {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(21,154,92,0.20);
    border-radius: 18px;
    padding: 18px 16px;
    text-align:center;
    box-shadow: 0 6px 18px rgba(21,154,92,0.07);
}

.side-suggest-box {
    min-height: 170px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-top: 42px;
}

.suggest-title {
    color:#064b3b;
    font-size:19px;
    font-weight:900;
    text-align:center;
}

.suggest-sub {
    color:#50422c;
    font-size:12px;
    font-weight:700;
    margin-top:6px;
    line-height:1.8;
}

.best-inline-card {
    background: linear-gradient(90deg, rgba(231,245,234,0.95), rgba(244,251,245,0.95));
    border: 1px solid #cfe7d4;
    border-radius: 16px;
    padding: 14px 12px;
    margin-top: 14px;
    text-align: center;
    box-shadow: 0 8px 18px rgba(21,154,92,0.08);
}

.best-inline-head {
    color: #159a5c;
    font-size: 17px;
    font-weight: 900;
    line-height: 1.7;
}

.best-check {
    font-size: 16px;
    margin-left: 6px;
}

.best-inline-meta {
    margin-top: 6px;
    color: #4e5a4f;
    font-size: 12px;
    font-weight: 700;
    line-height: 1.8;
}

.best-inline-meta strong {
    color: #064b3b;
    font-weight: 900;
}

.best-dot {
    margin: 0 6px;
    color: #90b29a;
    font-weight: 900;
}

.form-shell {
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


if "page" not in st.session_state:
    st.session_state.page = "home"

if "entered" not in st.session_state:
    st.session_state.entered = False

if "show_best_day" not in st.session_state:
    st.session_state.show_best_day = False


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


def home_page():
    show_header()

    H("""
    <div class="hero-box">
        <div class="hero-icon">🕋</div>
        <div class="hero-title">مرحبًا بك في نظام التنبؤ بازدحام المعتمرين</div>
        <div class="hero-sub">Umrah Visitors Smart Forecasting System</div>
        <div class="hero-text">
            يساعد هذا النظام في عرض توقعات ازدحام المعتمرين للأيام القادمة،
            وتحديد مستوى الازدحام، وتقديم توصية واضحة لاختيار اليوم الأنسب لأداء العمرة.
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
            <div class="feature-card">
                <div class="feature-icon">{esc(icon)}</div>
                <div class="feature-title">{esc(title)}</div>
                <div class="feature-desc">{esc(desc)}</div>
            </div>
            """)

    st.markdown("<br>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.05, 1])

    with mid:
        if st.button("ابدأ إدخال البيانات", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()


def input_page():
    show_header()

    df_dates = load_data()

    H("""
    <div class="form-shell">
        <div class="form-title">نظام التنبؤ بمستويات ازدحام المعتمرين</div>
        <div class="form-sub">المسجد الحرام الشريف</div>
        <div class="form-line"></div>
        <div class="form-section">✎ إدخال بيانات</div>
    </div>
    """)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        name = st.text_input("الاسم الكامل", placeholder="أدخل اسم المعتمر")

    with c2:
        nationality = st.selectbox("الجنسية", NATIONALITY_OPTIONS, index=0)

    c3, c4 = st.columns(2)

    with c3:
        month = st.selectbox(
            "الشهر الهجري",
            HIJRI_MONTHS,
            index=10,
            key="month_selector"
        )

    available_days = get_available_days(df_dates, month)

    if not available_days:
        available_days = list(range(1, 31))

    day_key = f"day_selector_{month}"

    with c4:
        day = st.selectbox(
            "التاريخ الهجري",
            available_days,
            index=0,
            key=day_key
        )

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
        submitted = st.button("عرض النتائج", use_container_width=True)

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

    if not st.session_state.entered:
        st.warning("الرجاء إدخال البيانات أولًا.")
        if st.button("الذهاب إلى إدخال البيانات"):
            st.session_state.page = "input"
            st.rerun()
        return

    month = normalize_month_name(st.session_state.selected_month)
    day = int(st.session_state.selected_day)

    show_hajj_count = should_show_hajj_count(month, day)
    show_tawaf_ifadah = should_show_tawaf_ifadah(month, day)

    df = load_data()
    df7 = get_7_days(df, month, day)

    if df7.empty:
        st.error("لا توجد بيانات مطابقة لليوم أو الشهر المختار داخل ملف المودل.")
        return

    selected_df = df7[
        (df7[MONTH_COL].astype(str).str.strip() == str(month).strip()) &
        (df7["Hijri_Day_Num"] == int(day))
    ]

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

    best_day = get_best_day(df7, day, month)

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
        <div class="reco-title" style="color:{LEVEL_COLORS.get(crowd_level, '#064b3b')};">{esc(decision)}</div>
        <div class="reco-line"></div>
        <div class="reco-text">{esc(reason)}</div>
    </div>
    """)

    if show_hajj_count or show_tawaf_ifadah:
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        if show_hajj_count:
            _, s1, _ = st.columns([1, 1.3, 1])
            with s1:
                info_pill(f"عدد الحجاج المتوقع: {hajj_count}", "👥")

        elif show_tawaf_ifadah:
            _, s1, _ = st.columns([1, 1.3, 1])
            with s1:
                info_pill(f"عدد طواف الإفاضة المتوقع: {tawaf_count}", "🕋")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    suggest_col, chart_col = st.columns([1, 2.25], gap="large")

    with suggest_col:
        H("""
        <div class="suggest-box side-suggest-box">
            <div class="suggest-title">هل تبحث عن يوم أفضل؟</div>
            <div class="suggest-sub">يمكنك عرض اليوم الأقل ازدحامًا ضمن الأيام السبعة القريبة.</div>
        </div>
        """)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        if st.button("عرض اليوم الأفضل", use_container_width=True):
            st.session_state.show_best_day = True

        if st.session_state.show_best_day:
            best_day_card(best_day)

    with chart_col:
        H("""
        <div class="section-box">
            <div class="section-title">توقعات ازدحام المعتمرين – الأيام القادمة (7 أيام)</div>
            <div class="section-line"></div>
        </div>
        """)

        fig = build_chart(df7)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])

    with mid:
        if st.button("رجوع لإدخال بيانات جديدة", use_container_width=True):
            st.session_state.page = "input"
            st.session_state.show_best_day = False
            st.rerun()


if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "input":
    input_page()
else:
    dashboard_page()
