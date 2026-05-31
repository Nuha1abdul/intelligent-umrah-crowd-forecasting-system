from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="نظام ذكي للتنبؤ بمستويات ازدحام المعتمرين",
    page_icon="م",
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



def main_prediction_card(value, level, day_name, hijri_date):
    level = normalize_level(level)
    color = LEVEL_COLORS.get(level, "#0F6E52")
    H(f"""
    <div class="decision-hero-card" style="--level-color:{color};">
        <div class="decision-orb decision-orb-one"></div>
        <div class="decision-orb decision-orb-two"></div>

        <div class="decision-top-row">
            <div class="decision-eyebrow">مؤشر الزيارة اليومي</div>
            <div class="decision-status-chip" style="color:{color}; border-color:{color};">
                مستوى الازدحام: {esc(level)}
            </div>
        </div>

        <div class="decision-main">
            <div class="decision-label">الحالة المتوقعة</div>
            <div class="decision-level" style="color:{color};">{esc(level)}</div>
            <div class="decision-text">
                قراءة فورية لمستوى الازدحام بناءً على التوقعات اليومية ومقارنة الأيام القريبة.
            </div>
        </div>

        <div class="decision-meta-grid">
            <div class="decision-meta-item">
                <span>العدد المتوقع</span>
                <strong>{esc(value)}</strong>
            </div>
            <div class="decision-meta-item">
                <span>اليوم المختار</span>
                <strong>{esc(day_name)}</strong>
            </div>
            <div class="decision-meta-item">
                <span>التاريخ الهجري</span>
                <strong>{esc(hijri_date)}</strong>
            </div>
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
    --ink:#0B1F1A;
    --deep:#062B25;
    --deep-2:#0B3B32;
    --mint:#0F8B5F;
    --gold:#C9A45D;
    --sand:#F7F1E7;
    --panel:rgba(255,255,255,0.72);
    --panel-strong:rgba(255,255,255,0.88);
    --line:rgba(20,54,46,0.10);
    --gold-line:rgba(201,164,93,0.22);
    --muted:#66746F;
}

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
}

.stApp {
    background:
        radial-gradient(circle at 12% 4%, rgba(201,164,93,0.16), transparent 24%),
        radial-gradient(circle at 88% 8%, rgba(15,139,95,0.13), transparent 26%),
        linear-gradient(180deg, #FCFAF5 0%, #F4EFE6 55%, #F0E8DA 100%);
}

.stApp::before {
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    opacity:0.035;
    background-image:
      linear-gradient(30deg, rgba(6,43,37,0.75) 12%, transparent 12.5%, transparent 87%, rgba(6,43,37,0.75) 87.5%, rgba(6,43,37,0.75)),
      linear-gradient(150deg, rgba(6,43,37,0.75) 12%, transparent 12.5%, transparent 87%, rgba(6,43,37,0.75) 87.5%, rgba(6,43,37,0.75));
    background-size:92px 160px;
    z-index:0;
}

.stApp > * {
    position:relative;
    z-index:1;
}

#MainMenu, footer, header {
    visibility:hidden;
}

section[data-testid="stSidebar"] {
    display:none !important;
}

.block-container {
    max-width: 1320px !important;
    padding-top: 1rem !important;
    padding-bottom: 1.6rem !important;
}

div[data-testid="stHorizontalBlock"] {
    gap: 0.9rem !important;
}

/* Header */
.main-header {
    position:relative;
    overflow:hidden;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:22px;
    min-height:112px;
    padding:24px 28px;
    margin-bottom:14px;
    border-radius:32px;
    background:
      linear-gradient(135deg, rgba(6,43,37,0.98), rgba(13,67,56,0.96)),
      radial-gradient(circle at 18% 20%, rgba(201,164,93,0.22), transparent 30%);
    border:1px solid rgba(201,164,93,0.34);
    box-shadow:0 30px 70px rgba(6,43,37,0.16);
}

.main-header::after {
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(115deg, transparent 0%, rgba(255,255,255,0.065) 42%, transparent 68%);
    pointer-events:none;
}

.header-brand {
    display:flex;
    align-items:center;
    gap:16px;
    position:relative;
    z-index:2;
}

.header-mark {
    width:58px;
    height:58px;
    border-radius:22px;
    display:grid;
    place-items:center;
    color:#F8E9C5;
    font-size:26px;
    font-weight:900;
    background:linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.035));
    border:1px solid rgba(248,233,197,0.25);
    box-shadow: inset 0 0 30px rgba(201,164,93,0.10);
}

.header-kicker {
    color:rgba(248,233,197,0.72);
    font-size:11px;
    font-weight:800;
    letter-spacing:0.4px;
    text-align:right;
}

.header-main-title {
    color:#FFF9EC;
    font-size:27px;
    font-weight:900;
    letter-spacing:-0.7px;
    margin-top:4px;
    text-align:right;
}

.header-subtitle {
    position:relative;
    z-index:2;
    color:#E5CA8E;
    font-size:13px;
    font-weight:800;
    padding:9px 14px;
    border-radius:999px;
    background:rgba(255,255,255,0.07);
    border:1px solid rgba(248,233,197,0.13);
}

/* Nav */
.nav-frame {
    width:min(760px, 100%);
    margin:0 auto;
    padding:8px;
    border-radius:24px;
    background:rgba(255,255,255,0.58);
    border:1px solid rgba(201,164,93,0.20);
    box-shadow:0 18px 42px rgba(6,43,37,0.06);
    backdrop-filter:blur(18px);
}

.nav-frame + div {
    margin-top:0 !important;
}

.stButton button {
    background:linear-gradient(135deg, #0F6E52, #062B25) !important;
    color:#FFFFFF !important;
    border:1px solid rgba(201,164,93,0.22) !important;
    box-shadow:0 16px 32px rgba(6,43,37,0.13) !important;
    border-radius:18px !important;
    min-height:45px !important;
    font-weight:900 !important;
    font-size:13px !important;
    transition:all 0.18s ease !important;
}

.stButton button:hover {
    transform:translateY(-2px);
    filter:brightness(1.06);
    box-shadow:0 20px 42px rgba(6,43,37,0.17) !important;
}

/* Home */
.hero-box {
    position:relative;
    overflow:hidden;
    padding:42px 38px;
    text-align:center;
    margin-bottom:18px;
    border-radius:34px;
    background:linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,250,239,0.70));
    border:1px solid rgba(201,164,93,0.20);
    box-shadow:0 28px 68px rgba(6,43,37,0.075);
    backdrop-filter:blur(16px);
}

.hero-box::before {
    content:"";
    position:absolute;
    width:360px;
    height:360px;
    border-radius:50%;
    right:-180px;
    top:-210px;
    background:radial-gradient(circle, rgba(15,139,95,0.12), transparent 68%);
}

.hero-box::after {
    content:"";
    position:absolute;
    width:280px;
    height:280px;
    border-radius:50%;
    left:-140px;
    bottom:-160px;
    background:radial-gradient(circle, rgba(201,164,93,0.14), transparent 70%);
}

.hero-kicker {
    width:fit-content;
    margin:0 auto 14px auto;
    padding:7px 16px;
    border-radius:999px;
    background:rgba(15,139,95,0.08);
    color:#0F6E52;
    border:1px solid rgba(15,139,95,0.13);
    font-size:12px;
    font-weight:900;
}

.hero-title {
    color:var(--deep);
    font-size:38px;
    font-weight:900;
    letter-spacing:-1.2px;
}

.hero-sub {
    color:#A87312;
    font-size:15px;
    font-weight:900;
    margin-top:10px;
}

.hero-text {
    color:#4C5D56;
    max-width:790px;
    margin:16px auto 0 auto;
    line-height:1.95;
    font-size:14px;
    font-weight:700;
}

.feature-card {
    position:relative;
    overflow:hidden;
    min-height:150px;
    padding:26px 22px;
    border-radius:28px;
    text-align:right;
    background:rgba(255,255,255,0.70);
    border:1px solid rgba(201,164,93,0.20);
    box-shadow:0 22px 54px rgba(6,43,37,0.055);
    backdrop-filter:blur(14px);
    transition:all 0.18s ease;
}

.feature-card:hover {
    transform:translateY(-4px);
    box-shadow:0 28px 70px rgba(6,43,37,0.075);
}

.feature-icon {
    width:44px;
    height:44px;
    border-radius:18px;
    display:grid;
    place-items:center;
    color:#A87312;
    background:rgba(201,164,93,0.10);
    border:1px solid rgba(201,164,93,0.20);
    font-size:13px;
    font-weight:900;
    margin-bottom:14px;
}

.feature-title {
    color:var(--deep);
    font-weight:900;
    font-size:18px;
}

.feature-desc {
    color:#66746F;
    font-weight:700;
    font-size:12px;
    margin-top:9px;
    line-height:1.8;
}

/* Form */
.form-shell {
    background:rgba(255,255,255,0.76) !important;
    border:1px solid rgba(201,164,93,0.22) !important;
    border-radius:34px !important;
    padding:34px 38px !important;
    box-shadow:0 28px 68px rgba(6,43,37,0.075);
    backdrop-filter:blur(16px);
}

.form-title {
    text-align:center;
    color:var(--deep);
    font-size:30px;
    font-weight:900;
    letter-spacing:-0.8px;
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
    background:linear-gradient(90deg, transparent, var(--gold), transparent);
    margin:14px auto 18px auto;
}

.form-section {
    width:fit-content;
    margin:0 auto;
    color:#0F6E52;
    background:rgba(15,139,95,0.08);
    border:1px solid rgba(15,139,95,0.13);
    border-radius:999px;
    padding:7px 18px;
    font-size:13px;
    font-weight:900;
}

.stTextInput input, .stSelectbox > div > div {
    background:#FFFEFA !important;
    border:1px solid rgba(201,164,93,0.32) !important;
    border-radius:18px !important;
    min-height:48px !important;
    box-shadow:0 14px 30px rgba(6,43,37,0.035);
}

/* Decision hero */
.decision-hero-card {
    position:relative;
    overflow:hidden;
    min-height:250px;
    padding:30px 34px;
    border-radius:34px;
    background:
      linear-gradient(135deg, rgba(6,43,37,0.98), rgba(13,67,56,0.94));
    border:1px solid rgba(201,164,93,0.34);
    box-shadow:0 34px 78px rgba(6,43,37,0.19);
}

.decision-hero-card::before {
    content:"";
    position:absolute;
    inset:0;
    background:
      linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.06) 44%, transparent 72%);
    pointer-events:none;
}

.decision-orb {
    position:absolute;
    border-radius:50%;
    filter:blur(2px);
    pointer-events:none;
}

.decision-orb-one {
    width:250px;
    height:250px;
    right:-90px;
    top:-120px;
    background:rgba(201,164,93,0.16);
}

.decision-orb-two {
    width:220px;
    height:220px;
    left:-100px;
    bottom:-130px;
    background:rgba(15,139,95,0.22);
}

.decision-top-row {
    position:relative;
    z-index:2;
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:12px;
}

.decision-eyebrow {
    color:rgba(255,249,236,0.72);
    font-size:12px;
    font-weight:900;
}

.decision-status-chip {
    padding:7px 14px;
    border-radius:999px;
    background:rgba(255,255,255,0.08);
    border:1px solid;
    font-size:12px;
    font-weight:900;
}

.decision-main {
    position:relative;
    z-index:2;
    text-align:center;
    margin-top:26px;
}

.decision-label {
    color:#E5CA8E;
    font-size:13px;
    font-weight:900;
}

.decision-level {
    font-size:72px;
    font-weight:900;
    line-height:1.05;
    letter-spacing:-1.8px;
    margin-top:4px;
    text-shadow:0 14px 36px rgba(0,0,0,0.18);
}

.decision-text {
    color:rgba(255,249,236,0.78);
    font-size:13px;
    font-weight:700;
    max-width:620px;
    line-height:1.8;
    margin:10px auto 0 auto;
}

.decision-meta-grid {
    position:relative;
    z-index:2;
    margin-top:22px;
    display:grid;
    grid-template-columns:repeat(3, 1fr);
    gap:10px;
}

.decision-meta-item {
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.10);
    border-radius:18px;
    padding:11px 12px;
    text-align:center;
}

.decision-meta-item span {
    display:block;
    color:rgba(255,249,236,0.58);
    font-size:10px;
    font-weight:800;
}

.decision-meta-item strong {
    display:block;
    color:#FFF9EC;
    font-size:14px;
    font-weight:900;
    margin-top:4px;
}

/* KPI Cards */
.metric-card {
    position:relative;
    overflow:hidden;
    min-height:120px;
    padding:17px 18px;
    border-radius:26px;
    text-align:right;
    border:1px solid rgba(201,164,93,0.22);
    backdrop-filter:blur(14px);
    transition:all 0.18s ease;
}

.metric-card:hover {
    transform:translateY(-4px);
    box-shadow:0 28px 62px rgba(6,43,37,0.075) !important;
}

.metric-label {
    color:#65736D;
    font-size:12px;
    font-weight:900;
}

.metric-value {
    font-size:30px;
    font-weight:900;
    line-height:1.15;
    margin-top:14px;
    letter-spacing:-0.7px;
}

.metric-subtitle {
    color:#6D7772;
    font-size:11px;
    font-weight:800;
    margin-top:8px;
}

.metric-accent {
    position:absolute;
    right:0;
    bottom:0;
    height:4px;
    width:100%;
    opacity:0.70;
}

/* Recommendation */
.reco-box {
    position:relative;
    overflow:hidden;
    background:rgba(255,255,255,0.76);
    border:1px solid rgba(201,164,93,0.24);
    border-radius:28px;
    padding:18px 28px;
    box-shadow:0 24px 58px rgba(6,43,37,0.065);
    backdrop-filter:blur(16px);
}

.reco-label {
    text-align:center;
    color:#7B6A48;
    font-size:11px;
    font-weight:900;
}

.reco-title {
    text-align:center;
    font-size:27px;
    font-weight:900;
    margin-top:4px;
    letter-spacing:-0.6px;
}

.reco-line {
    width:130px;
    height:2px;
    background:linear-gradient(90deg, transparent, var(--gold), transparent);
    margin:9px auto 10px auto;
}

.reco-text {
    text-align:center;
    color:#33433D;
    font-size:14px;
    font-weight:700;
    line-height:1.9;
    max-width:980px;
    margin:auto;
}

/* Pills */
.small-pill {
    background:rgba(255,255,255,0.78);
    border:1px solid rgba(201,164,93,0.26);
    color:#514833;
    border-radius:999px;
    padding:9px 15px;
    min-height:36px;
    display:flex;
    align-items:center;
    justify-content:center;
    gap:8px;
    font-size:11px;
    font-weight:900;
    box-shadow:0 16px 34px rgba(6,43,37,0.045);
    backdrop-filter:blur(12px);
}

.pill-icon {
    min-width:32px;
    height:24px;
    border-radius:999px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    background:rgba(15,139,95,0.08);
    color:#0F6E52;
    font-size:10px;
    font-weight:900;
    padding:0 8px;
}

/* Chart and suggestion */
.section-box {
    background:rgba(255,255,255,0.72);
    border:1px solid rgba(201,164,93,0.20);
    border-radius:26px;
    padding:16px 20px;
    box-shadow:0 22px 54px rgba(6,43,37,0.055);
    backdrop-filter:blur(14px);
}

.section-title {
    text-align:right;
    color:var(--deep);
    font-size:16px;
    font-weight:900;
}

.section-line {
    width:96px;
    height:2px;
    background:linear-gradient(90deg, var(--gold), transparent);
    margin:9px 0 8px auto;
}

.suggest-box {
    background:linear-gradient(180deg, rgba(6,43,37,0.98), rgba(13,67,56,0.96));
    border:1px solid rgba(201,164,93,0.30);
    border-radius:30px;
    padding:28px 22px;
    text-align:center;
    box-shadow:0 28px 64px rgba(6,43,37,0.18);
}

.side-suggest-box {
    min-height:205px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    margin-top:18px;
}

.suggest-title {
    color:#FFF9EC;
    font-size:25px;
    font-weight:900;
    letter-spacing:-0.6px;
}

.suggest-sub {
    color:#E5CA8E;
    font-size:12px;
    font-weight:800;
    margin-top:12px;
    line-height:1.95;
}

.best-card {
    background:rgba(255,255,255,0.82);
    border:1px solid rgba(15,139,95,0.18);
    border-radius:28px;
    padding:20px 18px;
    margin-top:14px;
    text-align:center;
    box-shadow:0 22px 48px rgba(15,139,95,0.09);
    backdrop-filter:blur(14px);
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
    color:var(--deep);
    font-size:26px;
    font-weight:900;
}

.best-meta {
    margin-top:8px;
    color:#53615B;
    font-size:12px;
    font-weight:800;
    line-height:1.9;
    display:flex;
    flex-direction:column;
    gap:2px;
}

.best-note {
    margin-top:10px;
    color:#0F6E52;
    font-size:11px;
    font-weight:800;
    line-height:1.7;
}

@media (max-width: 900px) {
    .main-header {
        flex-direction:column;
        align-items:flex-start;
    }
    .header-main-title { font-size:22px; }
    .header-subtitle { font-size:12px; }
    .hero-title { font-size:28px; }
    .decision-level { font-size:52px; }
    .decision-meta-grid { grid-template-columns:1fr; }
    .metric-value { font-size:24px; }
}
</style>
""")




def show_header():
    H("""
    <div class="main-header">
        <div class="header-brand">
            <div class="header-mark">م</div>
            <div>
                <div class="header-kicker">Makkah Crowd Intelligence</div>
                <div class="header-main-title">نظام ذكي للتنبؤ بمستويات ازدحام المعتمرين</div>
            </div>
        </div>
        <div class="header-subtitle">تحليل تنبؤي وتوصية بأفضل أوقات أداء العمرة</div>
    </div>
    """)



def render_nav():
    H('<div class="nav-frame">')
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
    H('</div>')
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)


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
        main_prediction_card(format_number(prediction), crowd_level, weekday, hijri_date)

    with side_col:
        top_card("العدد المتوقع للمعتمرين", format_number(prediction), "معتمر", "")
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            top_card("اليوم المختار", weekday, hijri_date, "")
        with s2:
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
