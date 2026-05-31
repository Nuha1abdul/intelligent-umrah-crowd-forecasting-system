from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="نظام ذكي للتنبؤ بمستويات ازدحام المعتمرين",
    page_icon="◇",
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
            fillcolor=hex_to_rgba(seg_color, 0.18),
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
            line=dict(color=seg_color, width=5, shape="spline"),
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

    if selected_mask.any():
        selected_point = cdf[selected_mask].iloc[[0]]
        fig.add_trace(go.Scatter(
            x=selected_point["x_label"],
            y=selected_point["Prediction"],
            mode="markers",
            marker=dict(
                size=48,
                color="rgba(199,163,90,0.18)",
                line=dict(color="rgba(199,163,90,0.24)", width=2)
            ),
            hoverinfo="skip",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=selected_point["x_label"],
            y=selected_point["Prediction"],
            mode="markers+text",
            marker=dict(
                size=25,
                color="#FFF8E8",
                line=dict(color="#C7A35A", width=5),
                symbol="circle"
            ),
            text=["اليوم المختار"],
            textposition="bottom center",
            textfont=dict(size=12, color="#062B25"),
            hovertemplate="<b>اليوم المختار</b><br>%{x}<br>العدد المتوقع: %{y:,.0f}<extra></extra>",
            name="اليوم المختار",
            showlegend=False
        ))

    q1 = cdf["Prediction"].quantile(0.33)
    q2 = cdf["Prediction"].quantile(0.66)

    fig.add_hline(y=q1, line_dash="dot", line_color="rgba(21,154,92,0.30)", line_width=1)
    fig.add_hline(y=q2, line_dash="dot", line_color="rgba(200,146,30,0.30)", line_width=1)

    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=28, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.62)",
        font=dict(family="Cairo", size=12, color="#082E28"),
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
            gridcolor="rgba(120,100,60,0.08)",
            tickformat=",",
            title=dict(text="عدد المعتمرين", font=dict(size=11))
        ),
        hoverlabel=dict(
            bgcolor="#062B25",
            bordercolor="#C7A35A",
            font=dict(color="#FFF8E8", family="Cairo", size=12)
        )
    )

    return fig


def top_card(title, value, subtitle, icon="", level=None):
    if level is None:
        color = "#123F35"
        bg = "rgba(255,255,255,0.76)"
        border = "rgba(201,169,95,0.30)"
        accent = "#C7A35A"
        shadow = "rgba(8,46,40,0.06)"
    else:
        level = normalize_level(level)
        color = LEVEL_COLORS.get(level, "#123F35")
        bg = LEVEL_BG.get(level, "rgba(255,255,255,0.76)")
        border = LEVEL_BORDER.get(level, "rgba(201,169,95,0.30)")
        accent = color
        shadow = hex_to_rgba(color, 0.18)

    H(f"""
    <div class="metric-card" style="background:{bg}; border-color:{border}; box-shadow:0 18px 35px {shadow};">
        <div class="metric-label">{esc(title)}</div>
        <div class="metric-value" style="color:{color};">{esc(value)}</div>
        <div class="metric-subtitle">{esc(subtitle)}</div>
        <div class="metric-accent" style="background:{accent};"></div>
    </div>
    """)


def main_prediction_card(level, prediction_value, day_name, hijri_date):
    level = normalize_level(level)
    color = LEVEL_COLORS.get(level, "#0F6E52")
    H(f"""
    <div class="main-kpi-card">
        <div class="main-kpi-kicker">التوقع الرئيسي</div>
        <div class="main-kpi-number" style="color:{color};">{esc(level)}</div>
        <div class="main-kpi-label">مستوى الازدحام المتوقع</div>
        <div class="main-kpi-meta">
            <span>العدد المتوقع للمعتمرين: <strong>{esc(prediction_value)}</strong></span>
            <span>اليوم المختار: <strong>{esc(day_name)}</strong></span>
            <span>التاريخ الهجري: <strong>{esc(hijri_date)}</strong></span>
        </div>
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
        <div class="best-card">
            <div class="best-title">لا يوجد يوم بديل ضمن الأيام السبعة القريبة</div>
        </div>
        """)
        return

    day_name = esc(row.get("Weekday_AR", "غير متاح"))
    hijri_date = esc(row.get("Hijri_Date", ""))
    visitors = format_number(row.get("Prediction", 0))

    H(f"""
    <div class="best-card">
        <div class="best-kicker">اليوم الأنسب للزيارة</div>
        <div class="best-title">{day_name}</div>
        <div class="best-meta">
            <span>التاريخ الهجري: <strong>{hijri_date}</strong></span>
            <span>العدد المتوقع: <strong>{visitors}</strong></span>
        </div>
        <div class="best-note">تم اختياره لأنه الأقل ازدحامًا ضمن الأيام السبعة القريبة.</div>
    </div>
    """)


H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800;900&display=swap');

:root {
    --green-950:#062B25;
    --green-900:#0A352E;
    --green-800:#104B40;
    --green-700:#0F6E52;
    --green-600:#0F8B5F;
    --gold-700:#A87312;
    --gold-500:#C7A35A;
    --gold-200:#EFE1B8;
    --cream-50:#FCFAF4;
    --cream-100:#F5EFE3;
    --ink:#10241F;
    --muted:#66736D;
    --line:rgba(199,163,90,0.28);
}

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
}

.stApp {
    background:
      radial-gradient(circle at 12% 7%, rgba(199,163,90,0.13), transparent 24%),
      radial-gradient(circle at 90% 10%, rgba(15,110,82,0.12), transparent 24%),
      radial-gradient(circle at 50% 100%, rgba(15,110,82,0.06), transparent 30%),
      linear-gradient(180deg, #FBFAF5 0%, #F3EDE1 100%);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.055;
    background-image:
      linear-gradient(30deg, rgba(8,46,40,0.45) 12%, transparent 12.5%, transparent 87%, rgba(8,46,40,0.45) 87.5%, rgba(8,46,40,0.45)),
      linear-gradient(150deg, rgba(8,46,40,0.45) 12%, transparent 12.5%, transparent 87%, rgba(8,46,40,0.45) 87.5%, rgba(8,46,40,0.45)),
      linear-gradient(30deg, rgba(8,46,40,0.45) 12%, transparent 12.5%, transparent 87%, rgba(8,46,40,0.45) 87.5%, rgba(8,46,40,0.45)),
      linear-gradient(150deg, rgba(8,46,40,0.45) 12%, transparent 12.5%, transparent 87%, rgba(8,46,40,0.45) 87.5%, rgba(8,46,40,0.45));
    background-size: 82px 142px;
    background-position: 0 0, 0 0, 41px 71px, 41px 71px;
    z-index: 0;
}

.stApp > * {
    position: relative;
    z-index: 1;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* Hide Streamlit sidebar completely for a full executive layout */
section[data-testid="stSidebar"] {
    display: none !important;
}

.block-container {
    max-width: 1360px !important;
    padding-top: 1rem !important;
    padding-bottom: 1.6rem !important;
}

/* Premium top navigation */
.top-nav {
    display:flex;
    justify-content:center;
    align-items:center;
    gap:10px;
    margin: -4px 0 18px 0;
}

div[data-testid="stHorizontalBlock"] {
    gap: 0.85rem !important;
}

.nav-shell {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(199,163,90,0.24);
    border-radius: 22px;
    padding: 8px;
    box-shadow: 0 14px 32px rgba(8,46,40,0.055);
    backdrop-filter: blur(14px);
}

.stButton button {
    background: linear-gradient(135deg, #0F6E52, #062B25) !important;
    color: white !important;
    border: 1px solid rgba(199,163,90,0.22) !important;
    outline: none !important;
    box-shadow: 0 16px 30px rgba(8,46,40,0.13) !important;
    border-radius: 16px !important;
    height: 44px !important;
    font-weight: 900 !important;
    font-size: 13px !important;
    transition: all 0.18s ease !important;
}

.stButton button:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
    box-shadow: 0 20px 36px rgba(8,46,40,0.16) !important;
}

.stButton button:focus,
.stButton button:active {
    outline: none !important;
    box-shadow: 0 16px 30px rgba(8,46,40,0.13) !important;
}

.main-header {
    position: relative;
    overflow: hidden;
    background:
      linear-gradient(135deg, rgba(6,43,37,0.98), rgba(16,75,64,0.96)),
      radial-gradient(circle at 18% 30%, rgba(199,163,90,0.18), transparent 26%);
    border: 1px solid rgba(199,163,90,0.38);
    border-radius: 30px;
    padding: 20px 28px;
    min-height: 108px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 28px 58px rgba(8,46,40,0.15);
    margin-bottom: 16px;
}

.main-header::before {
    content:"";
    position:absolute;
    inset:0;
    background:
      linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.06) 44%, transparent 72%);
    pointer-events:none;
}

.header-main-title {
    color: #FFF8E8;
    font-size: 28px;
    font-weight: 900;
    text-align: center;
    letter-spacing:-0.7px;
}

.header-subtitle {
    color: #D8BD78;
    font-size: 14px;
    font-weight: 800;
    text-align: center;
    margin-top: 8px;
}

.header-decor {
    width: 300px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #D8BD78, transparent);
    margin: 12px auto 0 auto;
}

.header-logo-box {
    width: 54px;
    height: 54px;
    border-radius: 18px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(216,189,120,0.45);
    display: grid;
    place-items: center;
    color:#F6E7BE;
    font-size: 21px;
    font-weight:900;
    box-shadow: inset 0 0 26px rgba(216,189,120,0.07);
}

.hero-box {
    position:relative;
    overflow:hidden;
    background:
      linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,251,242,0.80));
    border: 1px solid var(--line);
    border-radius: 30px;
    padding: 38px 36px;
    text-align:center;
    margin-bottom: 18px;
    box-shadow: 0 26px 60px rgba(8,46,40,0.075);
    backdrop-filter: blur(14px);
}

.hero-box::after {
    content:"";
    position:absolute;
    width:280px;
    height:280px;
    border-radius:50%;
    background:rgba(15,110,82,0.055);
    left:-100px;
    bottom:-135px;
}

.hero-kicker {
    width:fit-content;
    margin:0 auto 12px auto;
    padding:7px 16px;
    border-radius:999px;
    background:rgba(15,110,82,0.075);
    color:#0F6E52;
    border:1px solid rgba(15,110,82,0.12);
    font-size:12px;
    font-weight:900;
}

.hero-title {
    color:#062B25;
    font-size:34px;
    font-weight:900;
    letter-spacing:-1px;
}

.hero-sub {
    color:#A87312;
    font-size:15px;
    font-weight:900;
    margin-top:10px;
}

.hero-text {
    color:#465651;
    font-size:14px;
    font-weight:700;
    margin:16px auto 0 auto;
    line-height:1.95;
    max-width:780px;
}

.feature-card {
    position:relative;
    overflow:hidden;
    background: rgba(255,255,255,0.74);
    border: 1px solid rgba(201,169,95,0.24);
    border-radius: 24px;
    padding: 24px 20px;
    text-align:right;
    box-shadow: 0 22px 45px rgba(8,46,40,0.055);
    min-height: 136px;
    backdrop-filter: blur(12px);
    transition:0.18s ease;
}

.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 28px 58px rgba(8,46,40,0.075);
}

.feature-icon {
    width:40px;
    height:40px;
    border-radius:15px;
    display:grid;
    place-items:center;
    background:rgba(8,46,40,0.06);
    border:1px solid rgba(199,163,90,0.28);
    color:#A87312;
    font-size:13px;
    font-weight:900;
    margin-bottom:12px;
}

.feature-title {
    color:#062B25;
    font-weight:900;
    font-size:17px;
}

.feature-desc {
    color:#64706B;
    font-weight:700;
    font-size:12px;
    margin-top:8px;
    line-height:1.75;
}

/* Main prediction KPI */
.main-kpi-card {
    position:relative;
    overflow:hidden;
    background:
      linear-gradient(135deg, rgba(6,43,37,0.98), rgba(15,110,82,0.94));
    border: 1px solid rgba(216,189,120,0.38);
    border-radius: 30px;
    padding: 28px 34px;
    min-height: 222px;
    text-align: center;
    box-shadow: 0 30px 66px rgba(8,46,40,0.18);
}

.main-kpi-card::before {
    content:"";
    position:absolute;
    inset:0;
    background:
      radial-gradient(circle at 16% 18%, rgba(216,189,120,0.18), transparent 26%),
      linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.055) 48%, transparent 72%);
    pointer-events:none;
}

.main-kpi-kicker {
    width:fit-content;
    margin:0 auto 12px auto;
    padding:7px 16px;
    border-radius:999px;
    background:rgba(255,255,255,0.08);
    color:#F5E7C2;
    border:1px solid rgba(245,231,194,0.18);
    font-size:12px;
    font-weight:900;
}

.main-kpi-number {
    color:#FFFFFF;
    font-size:64px;
    font-weight:900;
    line-height:1.05;
    letter-spacing:-2px;
}

.main-kpi-label {
    color:#D8BD78;
    font-size:15px;
    font-weight:900;
    margin-top:10px;
}

.main-kpi-meta {
    margin-top:18px;
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    justify-content:center;
    gap:10px;
}

.main-kpi-meta span {
    padding:7px 12px;
    border-radius:999px;
    background:rgba(255,255,255,0.08);
    color:#F7F1E3;
    border:1px solid rgba(255,255,255,0.09);
    font-size:11px;
    font-weight:800;
}

.metric-card {
    position:relative;
    overflow:hidden;
    border: 1px solid rgba(201,169,95,0.28);
    border-radius: 22px;
    padding: 14px 16px;
    min-height: 112px;
    text-align: right;
    backdrop-filter: blur(14px);
    transition:0.18s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 24px 46px rgba(8,46,40,0.08) !important;
}

.metric-label {
    color:#68766F;
    font-size:12px;
    font-weight:900;
}

.metric-value {
    font-size:25px;
    font-weight:900;
    line-height:1.18;
    margin-top:12px;
    letter-spacing:-0.5px;
}

.metric-subtitle {
    color:#6D756F;
    font-size:11px;
    font-weight:800;
    margin-top:6px;
}

.metric-accent {
    position:absolute;
    right:0;
    bottom:0;
    height:4px;
    width:100%;
    opacity:0.65;
}

.reco-box {
    position:relative;
    overflow:hidden;
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,251,242,0.82));
    border: 1px solid rgba(201,169,95,0.34);
    border-radius: 24px;
    padding: 14px 24px;
    box-shadow: 0 18px 38px rgba(8,46,40,0.065);
    backdrop-filter: blur(14px);
}

.reco-label {
    text-align:center;
    color:#786A4E;
    font-size:11px;
    font-weight:900;
}

.reco-title {
    text-align:center;
    font-size:23px;
    font-weight:900;
    margin-top:3px;
    letter-spacing:-0.5px;
}

.reco-line {
    width:132px;
    height:2px;
    background: linear-gradient(90deg, transparent, #C7A35A, transparent);
    margin:7px auto 8px auto;
}

.reco-text {
    text-align:center;
    color:#31413B;
    font-size:13px;
    font-weight:700;
    line-height:1.75;
    max-width:1000px;
    margin:auto;
}

.small-pill {
    background: rgba(255,255,255,0.84);
    border: 1px solid rgba(201,169,95,0.32);
    color:#5D5138;
    border-radius: 999px;
    padding: 8px 14px;
    min-height: 34px;
    display:flex;
    align-items:center;
    justify-content:center;
    gap:8px;
    font-size:11px;
    font-weight:900;
    box-shadow:0 12px 28px rgba(8,46,40,0.05);
    backdrop-filter: blur(12px);
}

.pill-icon {
    min-width:30px;
    height:24px;
    border-radius:999px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    background:rgba(15,110,82,0.08);
    color:#0F6E52;
    font-size:10px;
    font-weight:900;
    padding:0 8px;
}

.section-box {
    background: rgba(255,255,255,0.74);
    border: 1px solid rgba(201,169,95,0.23);
    border-radius: 24px;
    padding: 14px 18px;
    box-shadow: 0 18px 40px rgba(8,46,40,0.055);
    height: 100%;
    backdrop-filter: blur(12px);
}

.section-title {
    text-align:right;
    color:#062B25;
    font-size:16px;
    font-weight:900;
}

.section-line {
    width:92px;
    height:2px;
    background: linear-gradient(90deg, #C7A35A, transparent);
    margin: 8px 0 8px auto;
}

.suggest-box {
    background: linear-gradient(180deg, rgba(6,43,37,0.98), rgba(16,75,64,0.96));
    border: 1px solid rgba(199,163,90,0.34);
    border-radius: 28px;
    padding: 26px 22px;
    text-align:center;
    box-shadow: 0 24px 52px rgba(8,46,40,0.18);
}

.side-suggest-box {
    min-height: 190px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-top: 18px;
}

.suggest-title {
    color:#FFF8E8;
    font-size:24px;
    font-weight:900;
    text-align:center;
    letter-spacing:-0.5px;
}

.suggest-sub {
    color:#D8BD78;
    font-size:12px;
    font-weight:800;
    margin-top:12px;
    line-height:1.95;
}

.best-card {
    background: linear-gradient(135deg, rgba(236,247,240,0.98), rgba(255,255,255,0.92));
    border: 1px solid rgba(15,139,95,0.22);
    border-radius: 26px;
    padding: 20px 16px;
    margin-top: 14px;
    text-align: center;
    box-shadow: 0 20px 42px rgba(15,139,95,0.10);
    backdrop-filter: blur(12px);
}

.best-kicker {
    width:fit-content;
    margin:0 auto 8px auto;
    padding:5px 12px;
    border-radius:999px;
    background:rgba(15,139,95,0.08);
    color:#0F8B5F;
    font-size:10px;
    font-weight:900;
}

.best-title {
    color: #062B25;
    font-size: 25px;
    font-weight: 900;
    line-height: 1.5;
}

.best-meta {
    margin-top: 8px;
    color: #53615B;
    font-size: 12px;
    font-weight: 800;
    line-height: 1.9;
    display:flex;
    flex-direction:column;
    gap:2px;
}

.best-meta strong {
    color: #062B25;
    font-weight: 900;
}

.best-note {
    margin-top:10px;
    color:#0F6E52;
    font-size:11px;
    font-weight:800;
    line-height:1.7;
}

.form-shell {
    background: linear-gradient(135deg, rgba(255,255,255,0.90), rgba(255,251,242,0.80)) !important;
    border:1px solid rgba(201,169,95,0.30) !important;
    border-radius: 30px !important;
    padding: 30px 34px !important;
    box-shadow: 0 26px 58px rgba(8,46,40,0.075);
    backdrop-filter: blur(14px);
}

.form-title {
    text-align:center;
    color:#062B25;
    font-size:28px;
    font-weight:900;
    letter-spacing:-0.6px;
}

.form-sub {
    text-align:center;
    color:#A87312;
    font-size:13px;
    font-weight:900;
    margin-top:8px;
}

.form-line {
    width:280px;
    height:2px;
    background:linear-gradient(90deg, transparent, #C7A35A, transparent);
    margin:14px auto 18px auto;
}

.form-section {
    width:fit-content;
    margin: 0 auto 5px auto;
    color:#0F6E52;
    background:rgba(15,110,82,0.075);
    border:1px solid rgba(15,110,82,0.12);
    border-radius:999px;
    padding:7px 18px;
    font-size:13px;
    font-weight:900;
}

.stTextInput input, .stSelectbox > div > div {
    background:#FFFEFA !important;
    border:1px solid rgba(201,169,95,0.36) !important;
    border-radius:16px !important;
    min-height:46px !important;
    box-shadow:0 12px 28px rgba(8,46,40,0.035);
}


.nav-caption {
    width: fit-content;
    margin: -4px auto 8px auto;
    padding: 4px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.50);
    border: 1px solid rgba(199,163,90,0.18);
    color: rgba(8,46,40,0.62);
    font-size: 10px;
    font-weight: 900;
    backdrop-filter: blur(10px);
}

@media (max-width: 900px) {
    .header-main-title { font-size: 21px; }
    .hero-title { font-size: 26px; }
    .main-kpi-number { font-size: 46px; }
    .metric-value { font-size: 24px; }
}
</style>
""")



def show_header():
    H("""
    <div class="main-header">
        <div class="header-logo-box">◇</div>
        <div style="flex:1;">
            <div class="header-main-title">نظام ذكي للتنبؤ بمستويات ازدحام المعتمرين</div>
            <div class="header-subtitle">تحليل تنبؤي وتوصية بأفضل أوقات أداء العمرة</div>
            <div class="header-decor"></div>
        </div>
    </div>
    """)


def render_nav():
    H('<div class="nav-caption">التنقل بين أقسام النظام</div>')
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
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "home"

if "entered" not in st.session_state:
    st.session_state.entered = False

if "show_best_day" not in st.session_state:
    st.session_state.show_best_day = False




def home_page():
    show_header()
    render_nav()

    H("""
    <div class="hero-box">
        <div class="hero-kicker">نظام تنبؤ وتوصية</div>
        <div class="hero-title">توقع الازدحام قبل الوصول</div>
        <div class="hero-sub">قراءة ذكية للأيام القادمة لاختيار وقت أداء العمرة الأنسب</div>
        <div class="hero-text">
            يعرض النظام العدد المتوقع للمعتمرين، ومستوى الازدحام، والتوصية المناسبة بناءً على التنبؤات اليومية
            والمقارنة بين الأيام القريبة في المسجد الحرام.
        </div>
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    cards = [
        ("01", "تنبؤ يومي", "تقدير العدد المتوقع للمعتمرين."),
        ("02", "تصنيف الازدحام", "تحويل التوقعات إلى مستويات واضحة."),
        ("03", "توصية عملية", "اقتراح يوم بديل عند ارتفاع الازدحام."),
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
        if st.button("بدء التنبؤ", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()


def input_page():
    show_header()
    render_nav()

    df_dates = load_data()

    H("""
    <div class="form-shell">
        <div class="form-title">إدخال بيانات الزيارة</div>
        <div class="form-sub">اختيار التاريخ الهجري لتوليد التوقع والتوصية</div>
        <div class="form-line"></div>
        <div class="form-section">بيانات التوقع</div>
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

    main_col, side_col = st.columns([1.55, 1], gap="large")

    with main_col:
        main_prediction_card(crowd_level, format_number(prediction), weekday, hijri_date)

    with side_col:
        s1, s2 = st.columns(2)
        with s1:
            top_card("العدد المتوقع للمعتمرين", format_number(prediction), "معتمر", "")
        with s2:
            top_card("اليوم المختار", weekday, hijri_date, "")
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        top_card("درجة الحرارة", temp_text, "متوسط اليوم", "")
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
                info_pill(f"عدد الحجاج المتوقع: {hajj_count}", "حج")

        elif show_tawaf_ifadah:
            _, s1, _ = st.columns([1, 1.3, 1])
            with s1:
                info_pill(f"عدد طواف الإفاضة المتوقع: {tawaf_count}", "طواف")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    suggest_col, chart_col = st.columns([1, 2.25], gap="large")

    with suggest_col:
        H("""
        <div class="suggest-box side-suggest-box">
            <div class="suggest-title">اقتراح يوم بديل</div>
            <div class="suggest-sub">يقارن النظام الأيام السبعة القريبة ويقترح الأقل ازدحامًا عند الحاجة.</div>
        </div>
        """)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        if st.button("اقتراح يوم أنسب", use_container_width=True):
            st.session_state.show_best_day = True

        if st.session_state.show_best_day:
            best_day_card(best_day)

    with chart_col:
        H("""
        <div class="section-box">
            <div class="section-title">الاتجاه المتوقع للعدد اليومي خلال 7 أيام</div>
            <div class="section-line"></div>
        </div>
        """)

        fig = build_chart(df7, month, day)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])

    with mid:
        if st.button("إدخال تاريخ جديد", use_container_width=True):
            st.session_state.page = "input"
            st.session_state.show_best_day = False
            st.rerun()


if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "input":
    input_page()
else:
    dashboard_page()
