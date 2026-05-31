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


def results_overview(value, level, day_name, hijri_date, temp_text):
    level = normalize_level(level)
    color = LEVEL_COLORS.get(level, "#0F6E52")
    H(f"""
    <div class="results-overview">
        <div class="summary-cards">
            <div class="summary-card compact-card">
                <div class="summary-label">اليوم المختار</div>
                <div class="summary-value day-value">{esc(day_name)}</div>
                <div class="summary-sub">{esc(hijri_date)}</div>
            </div>
            <div class="summary-card compact-card">
                <div class="summary-label">العدد المتوقع</div>
                <div class="summary-value">{esc(value)}</div>
                <div class="summary-sub">معتمر</div>
            </div>
            <div class="summary-card wide-card">
                <div>
                    <div class="summary-label">درجة الحرارة</div>
                    <div class="summary-sub">متوسط اليوم</div>
                </div>
                <div class="summary-value temp-value">{esc(temp_text)}</div>
            </div>
        </div>

        <div class="decision-card" style="--level-color:{color};">
            <div class="decision-surface"></div>
            <div class="decision-kicker">مؤشر الازدحام المتوقع</div>
            <div class="decision-level" style="color:{color};">{esc(level)}</div>
            <div class="decision-label">قرار سريع يساعدك على اختيار وقت الزيارة</div>
            <div class="decision-meta">
                <span><strong>العدد المتوقع:</strong> {esc(value)} معتمر</span>
                <span><strong>اليوم:</strong> {esc(day_name)}</span>
                <span><strong>التاريخ:</strong> {esc(hijri_date)}</span>
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
    --emerald-950:#031f1b;
    --emerald-900:#062b25;
    --emerald-800:#0b3d34;
    --emerald-700:#0f6e52;
    --emerald-600:#159a5c;
    --gold-600:#c7a35a;
    --gold-500:#d8bd78;
    --gold-soft:#f3e6bd;
    --cream:#fbf7ee;
    --paper:rgba(255,255,255,0.78);
    --line:rgba(199,163,90,0.28);
    --ink:#092b25;
    --muted:#68756f;
}

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
}

.stApp {
    background:
      radial-gradient(circle at 10% 8%, rgba(216,189,120,0.20), transparent 24%),
      radial-gradient(circle at 88% 12%, rgba(15,110,82,0.18), transparent 26%),
      radial-gradient(circle at 44% 100%, rgba(6,43,37,0.10), transparent 34%),
      linear-gradient(135deg, #FFFCF5 0%, #F4EDDE 42%, #EDE4D1 100%);
    color: var(--ink);
}

.stApp::before {
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    opacity:0.07;
    background-image:
      linear-gradient(30deg, rgba(6,43,37,0.45) 12%, transparent 12.5%, transparent 87%, rgba(6,43,37,0.45) 87.5%, rgba(6,43,37,0.45)),
      linear-gradient(150deg, rgba(6,43,37,0.45) 12%, transparent 12.5%, transparent 87%, rgba(6,43,37,0.45) 87.5%, rgba(6,43,37,0.45)),
      linear-gradient(30deg, rgba(6,43,37,0.45) 12%, transparent 12.5%, transparent 87%, rgba(6,43,37,0.45) 87.5%, rgba(6,43,37,0.45)),
      linear-gradient(150deg, rgba(6,43,37,0.45) 12%, transparent 12.5%, transparent 87%, rgba(6,43,37,0.45) 87.5%, rgba(6,43,37,0.45));
    background-size: 96px 166px;
    background-position: 0 0, 0 0, 48px 83px, 48px 83px;
    z-index:0;
}

.stApp::after {
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    background:
      linear-gradient(90deg, rgba(255,255,255,0.28), transparent 18%, transparent 82%, rgba(255,255,255,0.24)),
      radial-gradient(circle at 50% 20%, rgba(255,255,255,0.38), transparent 45%);
    z-index:0;
}

.stApp > * { position:relative; z-index:1; }
#MainMenu, footer, header { visibility:hidden; }
section[data-testid="stSidebar"] { display:none !important; }

.block-container {
    max-width: 1380px !important;
    padding-top: 0.85rem !important;
    padding-bottom: 1.5rem !important;
}

div[data-testid="stHorizontalBlock"] { gap: 0.9rem !important; }

/* Buttons / navigation */
.stButton button {
    position:relative;
    overflow:hidden;
    background: linear-gradient(135deg, #0f6e52 0%, #062b25 100%) !important;
    color:#fff8e8 !important;
    border:1px solid rgba(216,189,120,0.34) !important;
    border-radius: 18px !important;
    height: 46px !important;
    font-size: 13px !important;
    font-weight: 900 !important;
    letter-spacing:-0.1px;
    box-shadow: 0 18px 34px rgba(6,43,37,0.14), inset 0 1px 0 rgba(255,255,255,0.10) !important;
    transition: all .22s ease !important;
}

.stButton button::before {
    content:"";
    position:absolute;
    inset:0;
    background: linear-gradient(115deg, transparent 0%, rgba(255,255,255,0.14) 45%, transparent 72%);
    transform: translateX(120%);
    transition: .4s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    filter: brightness(1.06);
    box-shadow: 0 24px 46px rgba(6,43,37,0.18), inset 0 1px 0 rgba(255,255,255,0.12) !important;
}

.stButton button:hover::before { transform: translateX(-120%); }
.stButton button:focus, .stButton button:active { outline:none !important; box-shadow: 0 18px 34px rgba(6,43,37,0.14) !important; }

.nav-caption {
    width: fit-content;
    margin: -2px auto 10px auto;
    padding: 5px 18px;
    border-radius: 999px;
    color: rgba(6,43,37,0.62);
    background: rgba(255,255,255,0.48);
    border: 1px solid rgba(199,163,90,0.20);
    box-shadow: 0 12px 28px rgba(6,43,37,0.045);
    backdrop-filter: blur(14px);
    font-size:10px;
    font-weight:900;
}

/* Header */
.main-header {
    position:relative;
    overflow:hidden;
    min-height: 126px;
    padding: 23px 32px;
    margin-bottom: 14px;
    border-radius: 34px;
    border:1px solid rgba(216,189,120,0.38);
    background:
      radial-gradient(circle at 18% 22%, rgba(216,189,120,0.22), transparent 26%),
      radial-gradient(circle at 88% 78%, rgba(21,154,92,0.18), transparent 28%),
      linear-gradient(135deg, rgba(3,31,27,0.99) 0%, rgba(8,59,50,0.98) 52%, rgba(4,35,31,0.99) 100%);
    box-shadow: 0 30px 70px rgba(6,43,37,0.18), inset 0 1px 0 rgba(255,255,255,0.10);
    display:flex;
    align-items:center;
    justify-content:space-between;
}

.main-header::before {
    content:"";
    position:absolute;
    inset:0;
    opacity:.13;
    background-image:
      linear-gradient(60deg, rgba(255,255,255,0.30) 1px, transparent 1px),
      linear-gradient(120deg, rgba(216,189,120,0.35) 1px, transparent 1px);
    background-size: 38px 38px;
}

.main-header::after {
    content:"";
    position:absolute;
    inset:0;
    background: linear-gradient(110deg, transparent 0%, rgba(255,255,255,0.08) 44%, transparent 75%);
    pointer-events:none;
}

.header-logo-box {
    position:relative;
    z-index:2;
    width: 70px;
    height:70px;
    border-radius: 24px;
    display:grid;
    place-items:center;
    color:#fff8e8;
    font-size:15px;
    font-weight:900;
    background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04));
    border:1px solid rgba(216,189,120,0.46);
    box-shadow: inset 0 0 28px rgba(216,189,120,0.08), 0 18px 34px rgba(0,0,0,0.12);
}

.header-logo-box::before { content:"عمرة"; }
.header-logo-box { font-size:0 !important; }

.header-main-title {
    position:relative;
    z-index:2;
    color:#fff8e8;
    text-align:center;
    font-size: 31px;
    font-weight:900;
    letter-spacing:-0.9px;
    line-height:1.35;
    text-shadow: 0 10px 28px rgba(0,0,0,0.16);
}

.header-subtitle {
    position:relative;
    z-index:2;
    color:#ddc37f;
    text-align:center;
    font-size:14px;
    font-weight:900;
    margin-top:8px;
}

.header-decor {
    position:relative;
    z-index:2;
    width: 320px;
    height:2px;
    margin: 12px auto 0 auto;
    background: linear-gradient(90deg, transparent, rgba(216,189,120,0.96), transparent);
}

/* Home */
.hero-box, .form-shell, .section-box, .metric-card, .reco-box, .best-card {
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

.hero-box {
    position:relative;
    overflow:hidden;
    text-align:center;
    padding: 44px 42px;
    margin-bottom: 20px;
    border-radius: 34px;
    background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,251,242,0.68));
    border: 1px solid rgba(199,163,90,0.26);
    box-shadow: 0 28px 64px rgba(6,43,37,0.075);
}

.hero-box::before {
    content:"";
    position:absolute;
    inset:18px;
    border-radius: 28px;
    border:1px solid rgba(199,163,90,0.12);
    pointer-events:none;
}

.hero-box::after {
    content:"";
    position:absolute;
    width:330px;
    height:330px;
    border-radius:50%;
    background: radial-gradient(circle, rgba(15,110,82,0.10), transparent 68%);
    left:-120px;
    bottom:-150px;
}

.hero-kicker {
    width:fit-content;
    margin:0 auto 14px auto;
    padding:8px 18px;
    border-radius:999px;
    color:#0f6e52;
    background: rgba(15,110,82,0.075);
    border:1px solid rgba(15,110,82,0.13);
    font-size:12px;
    font-weight:900;
}

.hero-title {
    color:#062b25;
    font-size:38px;
    font-weight:900;
    letter-spacing:-1.2px;
}

.hero-sub {
    color:#a87312;
    font-size:16px;
    font-weight:900;
    margin-top:10px;
}

.hero-text {
    color:#465651;
    font-size:14px;
    font-weight:700;
    max-width:790px;
    margin:17px auto 0 auto;
    line-height:2;
}

.feature-card {
    position:relative;
    overflow:hidden;
    min-height:145px;
    padding:26px 22px;
    border-radius: 28px;
    text-align:right;
    background: linear-gradient(145deg, rgba(255,255,255,0.78), rgba(255,251,242,0.62));
    border: 1px solid rgba(199,163,90,0.25);
    box-shadow: 0 24px 50px rgba(6,43,37,0.065);
    backdrop-filter: blur(15px);
    transition: all .22s ease;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 32px 70px rgba(6,43,37,0.10);
}

.feature-card::after {
    content:"";
    position:absolute;
    width:90px;
    height:90px;
    border-radius:50%;
    left:-35px;
    bottom:-35px;
    background:rgba(199,163,90,0.10);
}

.feature-icon {
    width:42px;
    height:42px;
    border-radius:16px;
    display:grid;
    place-items:center;
    color:#a87312;
    background:rgba(216,189,120,0.13);
    border:1px solid rgba(199,163,90,0.30);
    font-size:13px;
    font-weight:900;
    margin-bottom:13px;
}

.feature-title { color:#062b25; font-size:17px; font-weight:900; }
.feature-desc { color:#65726c; font-size:12px; font-weight:700; line-height:1.8; margin-top:8px; }

/* Dashboard cards */
.main-kpi-card {
    position:relative;
    overflow:hidden;
    min-height: 238px;
    padding: 32px 36px;
    border-radius: 36px;
    text-align:center;
    background:
      radial-gradient(circle at 18% 22%, rgba(216,189,120,0.20), transparent 25%),
      radial-gradient(circle at 84% 82%, rgba(21,154,92,0.20), transparent 25%),
      linear-gradient(135deg, rgba(3,31,27,0.99), rgba(15,110,82,0.96));
    border: 1px solid rgba(216,189,120,0.40);
    box-shadow: 0 34px 78px rgba(6,43,37,0.20), inset 0 1px 0 rgba(255,255,255,0.10);
}

.main-kpi-card::before {
    content:"";
    position:absolute;
    inset:0;
    opacity:.10;
    background-image:
      linear-gradient(60deg, rgba(255,255,255,0.35) 1px, transparent 1px),
      linear-gradient(120deg, rgba(216,189,120,0.35) 1px, transparent 1px);
    background-size: 36px 36px;
}

.main-kpi-card::after {
    content:"";
    position:absolute;
    inset:0;
    background: linear-gradient(110deg, transparent 0%, rgba(255,255,255,0.09) 45%, transparent 73%);
    pointer-events:none;
}

.main-kpi-kicker, .main-kpi-number, .main-kpi-label, .main-kpi-meta { position:relative; z-index:2; }

.main-kpi-kicker {
    width:fit-content;
    margin:0 auto 14px auto;
    padding:8px 18px;
    border-radius:999px;
    background:rgba(255,255,255,0.09);
    color:#f5e7c2;
    border:1px solid rgba(245,231,194,0.18);
    font-size:12px;
    font-weight:900;
}

.main-kpi-number {
    color:#ffffff;
    font-size:70px;
    font-weight:900;
    line-height:1;
    letter-spacing:-2px;
    text-shadow: 0 18px 42px rgba(0,0,0,0.18);
}

.main-kpi-label {
    color:#ddc37f;
    font-size:16px;
    font-weight:900;
    margin-top:12px;
}

.main-kpi-meta {
    margin-top:20px;
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    justify-content:center;
    gap:10px;
}

.main-kpi-meta span {
    padding:8px 14px;
    border-radius:999px;
    background:rgba(255,255,255,0.09);
    color:#fff8e8;
    border:1px solid rgba(255,255,255,0.10);
    font-size:11px;
    font-weight:900;
}

.metric-card {
    position:relative;
    overflow:hidden;
    min-height: 126px;
    padding: 17px 18px;
    border-radius: 28px;
    text-align:right;
    border:1px solid rgba(199,163,90,0.28);
    background: linear-gradient(145deg, rgba(255,255,255,0.80), rgba(255,251,242,0.64)) !important;
    box-shadow: 0 24px 48px rgba(6,43,37,0.075) !important;
    transition: all .22s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 32px 66px rgba(6,43,37,0.11) !important;
}

.metric-label { color:#68766f; font-size:12px; font-weight:900; }
.metric-value { font-size:28px; font-weight:900; line-height:1.18; margin-top:13px; letter-spacing:-.5px; }
.metric-subtitle { color:#6d756f; font-size:11px; font-weight:800; margin-top:7px; }
.metric-accent { position:absolute; right:0; bottom:0; width:100%; height:4px; opacity:.75; }

.reco-box {
    position:relative;
    overflow:hidden;
    padding: 18px 26px;
    border-radius: 30px;
    background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(255,251,242,0.68));
    border:1px solid rgba(199,163,90,0.31);
    box-shadow: 0 24px 54px rgba(6,43,37,0.075);
}

.reco-box::before {
    content:"";
    position:absolute;
    right:0;
    top:0;
    height:100%;
    width:5px;
    background:linear-gradient(180deg, #c7a35a, #0f6e52);
}

.reco-label { text-align:center; color:#786a4e; font-size:11px; font-weight:900; }
.reco-title { text-align:center; font-size:25px; font-weight:900; margin-top:4px; letter-spacing:-.5px; }
.reco-line { width:140px; height:2px; margin:9px auto 9px auto; background:linear-gradient(90deg, transparent, #c7a35a, transparent); }
.reco-text { text-align:center; color:#31413b; font-size:13.5px; font-weight:700; line-height:1.85; max-width:1030px; margin:auto; }

.small-pill {
    background: rgba(255,255,255,0.80);
    border:1px solid rgba(199,163,90,0.31);
    color:#5d5138;
    border-radius:999px;
    min-height:36px;
    padding:8px 15px;
    display:flex;
    align-items:center;
    justify-content:center;
    gap:8px;
    font-size:11px;
    font-weight:900;
    box-shadow:0 14px 30px rgba(6,43,37,0.055);
}

.pill-icon {
    min-width:32px;
    height:24px;
    border-radius:999px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    background:rgba(15,110,82,0.09);
    color:#0f6e52;
    font-size:10px;
    font-weight:900;
    padding:0 8px;
}

.section-box {
    border-radius: 28px;
    padding: 16px 20px;
    background: linear-gradient(145deg, rgba(255,255,255,0.78), rgba(255,251,242,0.62));
    border:1px solid rgba(199,163,90,0.24);
    box-shadow: 0 24px 52px rgba(6,43,37,0.065);
}

.section-title { text-align:right; color:#062b25; font-size:16px; font-weight:900; }
.section-line { width:100px; height:2px; margin:9px 0 8px auto; background:linear-gradient(90deg, #c7a35a, transparent); }

.suggest-box {
    position:relative;
    overflow:hidden;
    border-radius: 32px;
    padding: 28px 24px;
    text-align:center;
    background:
      radial-gradient(circle at 18% 14%, rgba(216,189,120,0.18), transparent 28%),
      linear-gradient(180deg, rgba(3,31,27,0.99), rgba(16,75,64,0.97));
    border:1px solid rgba(216,189,120,0.34);
    box-shadow: 0 28px 62px rgba(6,43,37,0.18);
}

.side-suggest-box { min-height: 200px; display:flex; flex-direction:column; justify-content:center; margin-top:16px; }
.suggest-title { color:#fff8e8; font-size:25px; font-weight:900; letter-spacing:-.6px; text-align:center; }
.suggest-sub { color:#ddc37f; font-size:12px; font-weight:800; line-height:1.95; margin-top:12px; }

.best-card {
    border-radius: 28px;
    padding: 20px 17px;
    margin-top:14px;
    text-align:center;
    background: linear-gradient(135deg, rgba(236,247,240,0.98), rgba(255,255,255,0.90));
    border:1px solid rgba(15,139,95,0.22);
    box-shadow: 0 24px 48px rgba(15,139,95,0.10);
}
.best-kicker { width:fit-content; margin:0 auto 8px auto; padding:5px 13px; border-radius:999px; color:#0f8b5f; background:rgba(15,139,95,0.08); font-size:10px; font-weight:900; }
.best-title { color:#062b25; font-size:26px; font-weight:900; line-height:1.5; }
.best-meta { margin-top:8px; color:#53615b; font-size:12px; font-weight:800; line-height:1.9; display:flex; flex-direction:column; gap:2px; }
.best-meta strong { color:#062b25; font-weight:900; }
.best-note { margin-top:10px; color:#0f6e52; font-size:11px; font-weight:800; line-height:1.7; }

/* Form */
.form-shell {
    position:relative;
    overflow:hidden;
    border-radius: 34px !important;
    padding: 34px 38px !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,251,242,0.66)) !important;
    border:1px solid rgba(199,163,90,0.28) !important;
    box-shadow:0 28px 64px rgba(6,43,37,0.075);
}

.form-title { text-align:center; color:#062b25; font-size:30px; font-weight:900; letter-spacing:-.7px; }
.form-sub { text-align:center; color:#a87312; font-size:13px; font-weight:900; margin-top:8px; }
.form-line { width:300px; height:2px; margin:15px auto 18px auto; background:linear-gradient(90deg, transparent, #c7a35a, transparent); }
.form-section { width:fit-content; margin:0 auto 4px auto; padding:8px 20px; border-radius:999px; color:#0f6e52; background:rgba(15,110,82,0.075); border:1px solid rgba(15,110,82,0.12); font-size:13px; font-weight:900; }

.stTextInput label, .stSelectbox label { color:#062b25 !important; font-weight:900 !important; font-size:13px !important; }
.stTextInput input, .stSelectbox > div > div {
    min-height:48px !important;
    border-radius:18px !important;
    background:rgba(255,254,250,0.92) !important;
    border:1px solid rgba(199,163,90,0.36) !important;
    box-shadow:0 14px 30px rgba(6,43,37,0.045);
}

@media (max-width: 900px) {
    .header-main-title { font-size:22px; }
    .header-subtitle { font-size:12px; }
    .header-logo-box { width:58px; height:58px; }
    .hero-title { font-size:27px; }
    .main-kpi-number { font-size:48px; }
    .metric-value { font-size:24px; }
}
</style>
""")



H("""
<style>
/* Global premium overrides */
.stApp {
    background:
      radial-gradient(circle at 14% 12%, rgba(199,163,90,0.18), transparent 24%),
      radial-gradient(circle at 86% 9%, rgba(15,110,82,0.14), transparent 26%),
      radial-gradient(circle at 55% 100%, rgba(6,43,37,0.08), transparent 34%),
      linear-gradient(135deg, #FBFAF5 0%, #F2E9D9 48%, #F8F5EC 100%) !important;
}

.block-container {
    max-width: 1400px !important;
    padding-top: 0.9rem !important;
}

.premium-header {
    min-height: 112px !important;
    padding: 20px 30px !important;
    border-radius: 34px !important;
    background:
      radial-gradient(circle at 10% 25%, rgba(216,189,120,0.22), transparent 26%),
      linear-gradient(135deg, #052A24 0%, #0B453A 52%, #062B25 100%) !important;
    box-shadow: 0 32px 70px rgba(6,43,37,0.18) !important;
}

.brand-mark {
    width: 64px;
    height: 64px;
    border-radius: 24px;
    display:grid;
    place-items:center;
    position:relative;
    color:#F6E7BE;
    background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.025));
    border:1px solid rgba(216,189,120,0.46);
    box-shadow: inset 0 0 30px rgba(216,189,120,0.10), 0 16px 32px rgba(0,0,0,0.10);
}
.brand-crescent { font-size: 34px; transform: translateY(-1px); }
.brand-spark {
    position:absolute;
    top:11px;
    right:13px;
    font-size:11px;
    color:#D8BD78;
}
.header-status {
    min-width: 98px;
    display:flex;
    align-items:center;
    justify-content:center;
    gap:7px;
    padding:8px 12px;
    border-radius:999px;
    color:#FFF8E8;
    font-size:11px;
    font-weight:900;
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(216,189,120,0.22);
}
.status-dot {
    width:8px;height:8px;border-radius:50%;
    background:#20C784;
    box-shadow:0 0 0 6px rgba(32,199,132,0.12);
}
.header-logo-box { display:none !important; }

.nav-caption {
    margin: -2px auto 10px auto !important;
    padding: 6px 16px !important;
    background: rgba(255,255,255,0.66) !important;
    color: rgba(6,43,37,0.70) !important;
    box-shadow: 0 12px 24px rgba(8,46,40,0.04);
}

.premium-landing {
    min-height: 285px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    isolation:isolate;
}
.premium-landing::before {
    content:"";
    position:absolute;
    inset:18px;
    border-radius:26px;
    border:1px solid rgba(199,163,90,0.13);
    pointer-events:none;
}
.hero-visual-ring {
    position:absolute;
    left:42px;
    top:50%;
    width:150px;
    height:150px;
    transform:translateY(-50%);
    border-radius:50%;
    border:1px solid rgba(199,163,90,0.28);
    box-shadow: inset 0 0 0 24px rgba(15,110,82,0.035), 0 0 45px rgba(15,110,82,0.08);
    opacity:0.9;
}
.hero-visual-ring::before,
.hero-visual-ring::after {
    content:"";
    position:absolute;
    border-radius:50%;
    inset:26px;
    border:1px dashed rgba(6,43,37,0.18);
}
.hero-visual-ring::after {
    inset:56px;
    background:rgba(199,163,90,0.16);
    border:none;
}

.feature-card, .metric-card, .reco-box, .section-box, .form-shell, .best-card {
    backdrop-filter: blur(18px) saturate(128%) !important;
}
.feature-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(255,248,232,0.62)) !important;
}
.feature-card::before {
    content:"";
    position:absolute;
    top:0;right:0;left:0;height:4px;
    background:linear-gradient(90deg, #C7A35A, rgba(15,110,82,0.55), transparent);
}

.crowding-hero {
    background:
      radial-gradient(circle at 14% 15%, rgba(216,189,120,0.20), transparent 27%),
      linear-gradient(135deg, #052A24 0%, #0C473D 54%, #062B25 100%) !important;
}
.crowding-level-text {
    font-size: 76px !important;
    text-shadow: 0 0 34px rgba(255,255,255,0.10);
    letter-spacing: -1.3px !important;
}
.hero-orb {
    position:absolute;
    border-radius:50%;
    pointer-events:none;
}
.hero-orb-1 {
    width:170px;height:170px;
    right:-55px;top:-55px;
    background:rgba(216,189,120,0.11);
    border:1px solid rgba(216,189,120,0.12);
}
.hero-orb-2 {
    width:118px;height:118px;
    left:34px;bottom:-55px;
    background:rgba(255,255,255,0.055);
    border:1px solid rgba(255,255,255,0.08);
}

.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,250,238,0.65)) !important;
    box-shadow: 0 22px 44px rgba(8,46,40,0.07) !important;
}
.metric-value { font-size: 27px !important; }

.suggest-box {
    background:
      radial-gradient(circle at 16% 20%, rgba(216,189,120,0.18), transparent 30%),
      linear-gradient(180deg, rgba(6,43,37,0.99), rgba(16,75,64,0.96)) !important;
}

.js-plotly-plot .plotly .modebar { display:none !important; }

@media (max-width: 900px) {
    .header-status { display:none; }
    .brand-mark { width:52px;height:52px;border-radius:18px; }
    .crowding-level-text { font-size: 48px !important; }
    .hero-visual-ring { display:none; }
}
</style>
""")


H("""
<style>
/* Refined results section: balanced dashboard layout */
.results-overview {
    display: grid;
    grid-template-columns: minmax(520px, 1.45fr) minmax(360px, 0.95fr);
    gap: 22px;
    align-items: stretch;
    direction: ltr;
    margin-top: 4px;
}
.results-overview > * { direction: rtl; }

.decision-card {
    position: relative;
    overflow: hidden;
    min-height: 252px;
    border-radius: 34px;
    padding: 34px 38px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background:
      radial-gradient(circle at 12% 20%, rgba(216,189,120,0.16), transparent 30%),
      radial-gradient(circle at 88% 82%, rgba(255,255,255,0.055), transparent 28%),
      linear-gradient(135deg, #062B25 0%, #0E4C40 54%, #031F1B 100%);
    border: 1px solid rgba(216,189,120,0.38);
    box-shadow: 0 28px 64px rgba(6,43,37,0.18), inset 0 1px 0 rgba(255,255,255,0.10);
}
.decision-card::before {
    content: "";
    position: absolute;
    inset: 0;
    opacity: 0.08;
    background-image:
      linear-gradient(60deg, rgba(255,255,255,0.45) 1px, transparent 1px),
      linear-gradient(120deg, rgba(216,189,120,0.45) 1px, transparent 1px);
    background-size: 38px 38px;
}
.decision-card::after {
    content: "";
    position: absolute;
    width: 210px;
    height: 210px;
    border-radius: 50%;
    right: -70px;
    top: -80px;
    background: rgba(216,189,120,0.09);
    border: 1px solid rgba(216,189,120,0.12);
}
.decision-surface {
    position: absolute;
    width: 150px;
    height: 150px;
    border-radius: 50%;
    left: -42px;
    bottom: -56px;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.08);
}
.decision-kicker,
.decision-level,
.decision-label,
.decision-meta { position: relative; z-index: 2; }
.decision-kicker {
    width: fit-content;
    padding: 8px 18px;
    border-radius: 999px;
    color: #F6E7BE;
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(246,231,190,0.16);
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 16px;
}
.decision-level {
    font-size: 74px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -1.8px;
    text-shadow: 0 14px 38px rgba(0,0,0,0.16);
}
.decision-label {
    margin-top: 12px;
    color: #D8BD78;
    font-size: 16px;
    font-weight: 900;
}
.decision-meta {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-top: 24px;
}
.decision-meta span {
    padding: 9px 16px;
    border-radius: 999px;
    color: #FFF8E8;
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.10);
    font-size: 11.5px;
    font-weight: 800;
}

.summary-cards {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    align-content: stretch;
}
.summary-card {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(255,250,238,0.68));
    border: 1px solid rgba(199,163,90,0.28);
    box-shadow: 0 22px 48px rgba(6,43,37,0.065);
    backdrop-filter: blur(18px) saturate(125%);
}
.summary-card::after {
    content: "";
    position: absolute;
    right: 0;
    bottom: 0;
    height: 5px;
    width: 100%;
    background: linear-gradient(90deg, transparent, rgba(199,163,90,0.82), transparent);
}
.compact-card {
    min-height: 118px;
    padding: 20px 22px;
}
.wide-card {
    grid-column: span 2;
    min-height: 118px;
    padding: 22px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.summary-label {
    color: #66736D;
    font-size: 13px;
    font-weight: 900;
}
.summary-value {
    margin-top: 12px;
    color: #062B25;
    font-size: 30px;
    font-weight: 900;
    line-height: 1.12;
    letter-spacing: -0.6px;
}
.day-value { font-size: 32px; }
.temp-value { font-size: 36px; margin-top: 0; }
.summary-sub {
    margin-top: 8px;
    color: #6D756F;
    font-size: 12px;
    font-weight: 800;
}

@media (max-width: 1050px) {
    .results-overview {
        grid-template-columns: 1fr;
        direction: rtl;
    }
    .decision-card { min-height: 230px; }
    .decision-level { font-size: 56px; }
}
</style>
""")

def show_header():
    H("""
    <div class="main-header premium-header clean-header">
        <div class="header-center">
            <div class="header-main-title">المنصة الذكية لتوقع الازدحام</div>
            <div class="header-subtitle">
                اختيار الوقت الأنسب للعمرة اعتمادًا على<br>
                التوقعات المستقبلية وتحليل مستويات الازدحام
            </div>
            <div class="header-decor"></div>
        </div>
    </div>
    """)



H("""
<style>
/* Final clean executive header without logo/icon */
.clean-header {
    justify-content: center !important;
    text-align: center !important;
    min-height: 128px !important;
    padding: 26px 34px !important;
}

.clean-header .header-center {
    position: relative;
    z-index: 3;
    width: 100%;
}

.clean-header .header-main-title {
    font-size: 34px !important;
    letter-spacing: -0.8px !important;
}

.clean-header .header-subtitle {
    line-height: 1.8 !important;
    font-size: 14px !important;
    margin-top: 8px !important;
}

.clean-header::before {
    opacity: 0.09 !important;
}

.clean-header::after {
    opacity: 0.85 !important;
}

/* Refined KPI card: less empty, cleaner, presentation-ready */
.refined-kpi {
    min-height: 232px !important;
    padding: 30px 38px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}

.refined-kpi .main-kpi-kicker {
    margin-bottom: 12px !important;
}

.refined-kpi .crowding-level-text {
    font-size: 78px !important;
    line-height: 0.95 !important;
    margin: 2px 0 8px 0 !important;
}

.refined-kpi .main-kpi-label {
    margin-top: 6px !important;
    color: #E1C985 !important;
}

.refined-meta {
    margin-top: 18px !important;
    gap: 12px !important;
}

.refined-meta span {
    min-width: 138px !important;
    text-align: center !important;
    padding: 9px 15px !important;
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
}

/* Remove old icon/status visually if any previous CSS remains */
.brand-mark,
.header-status,
.header-logo-box {
    display: none !important;
}

@media (max-width: 900px) {
    .clean-header .header-main-title {
        font-size: 24px !important;
    }
    .clean-header .header-subtitle {
        font-size: 12px !important;
    }
    .refined-kpi .crowding-level-text {
        font-size: 50px !important;
    }
}
</style>
""")


def render_nav():
    H('<div class="nav-caption">توقعات ذكية للأيام القادمة</div>')
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
    <div class="hero-box premium-landing">
        <div class="hero-visual-ring"></div>
        <div class="hero-kicker">نظام دعم القرار</div>
        <div class="hero-title">اختر الوقت الأنسب للعمرة</div>
        <div class="hero-sub">توقع مستوى الازدحام والحصول على توصية للوقت الأنسب للزيارة</div>
        <div class="hero-text">
            يعرض النظام العدد المتوقع للمعتمرين ومستوى الازدحام والتوصية المناسبة لمساعدتك في اختيار وقت الزيارة.
        </div>
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    cards = [
        ("01", "مؤشر ازدحام", "عرض مستوى اليوم المختار بشكل واضح وفوري."),
        ("02", "توقع عددي", "تقدير العدد المتوقع للمعتمرين في اليوم المحدد."),
        ("03", "توصية ذكية", "اقتراح يوم بديل أقل ازدحامًا عند الحاجة."),
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
        <div class="form-title">تحديد موعد الزيارة</div>
        <div class="form-sub">اختر تاريخ الزيارة لمعرفة مستوى الازدحام المتوقع</div>
        <div class="form-line"></div>
        <div class="form-section">معلومات الزيارة</div>
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

    results_overview(format_number(prediction), crowd_level, weekday, hijri_date, temp_text)
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

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
