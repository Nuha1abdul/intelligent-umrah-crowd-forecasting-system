from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="التنبؤ بمستويات الازدحام وتوصية أوقات الزيارة",
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
    "منخفض": "#0F8B5F",
    "متوسط": "#B88318",
    "مرتفع": "#B94B42"
}

LEVEL_BG = {
    "منخفض": "#ECF7F0",
    "متوسط": "#FFF6E2",
    "مرتفع": "#FFF0EE"
}

LEVEL_BORDER = {
    "منخفض": "#BDE3CB",
    "متوسط": "#E8C77E",
    "مرتفع": "#E4A7A1"
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



def build_chart(df7, selected_month=None, selected_day=None, best_day=None):
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

    best_mask = pd.Series(False, index=cdf.index)
    if best_day is not None:
        try:
            best_month = normalize_month_name(best_day.get(MONTH_COL, ""))
            best_day_num = int(best_day.get("Hijri_Day_Num"))
            best_mask = (
                (cdf[MONTH_COL].astype(str).str.strip() == str(best_month).strip()) &
                (cdf["Hijri_Day_Num"].astype(int) == best_day_num)
            )
        except Exception:
            pass

    y_min = cdf["Prediction"].min()
    y_max = cdf["Prediction"].max()
    pad = max((y_max - y_min) * 0.22, 500)
    base_y = max(0, y_min - pad)

    fig = go.Figure()

    # soft premium area
    fig.add_trace(go.Scatter(
        x=cdf["x_label"],
        y=cdf["Prediction"],
        mode="lines",
        line=dict(color="rgba(15,110,82,0.12)", width=0, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(15,110,82,0.08)",
        hoverinfo="skip",
        showlegend=False,
    ))

    # main muted line
    fig.add_trace(go.Scatter(
        x=cdf["x_label"],
        y=cdf["Prediction"],
        mode="lines",
        line=dict(color="rgba(8,46,40,0.45)", width=4, shape="spline"),
        hoverinfo="skip",
        showlegend=False,
    ))

    normal_df = cdf[~selected_mask & ~best_mask].copy()
    fig.add_trace(go.Scatter(
        x=normal_df["x_label"],
        y=normal_df["Prediction"],
        mode="markers+text",
        marker=dict(size=15, color="rgba(8,46,40,0.46)", line=dict(color="white", width=2)),
        text=normal_df["Prediction"].apply(lambda x: f"{int(round(float(x))):,}" if not pd.isna(x) else ""),
        textposition="top center",
        textfont=dict(size=12, color="#47534f"),
        hovertemplate="<b>%{x}</b><br>العدد المتوقع: %{y:,.0f}<extra></extra>",
        name="الأيام القريبة",
    ))

    if best_mask.any():
        best_df = cdf[best_mask].iloc[[0]]
        fig.add_trace(go.Scatter(
            x=best_df["x_label"],
            y=best_df["Prediction"],
            mode="markers+text",
            marker=dict(size=25, color="#159a5c", line=dict(color="white", width=4)),
            text=["أفضل بديل"],
            textposition="bottom center",
            textfont=dict(size=12, color="#0b5c43"),
            hovertemplate="<b>أفضل يوم بديل</b><br>%{x}<br>العدد المتوقع: %{y:,.0f}<extra></extra>",
            name="أفضل بديل",
        ))

    if selected_mask.any():
        selected_df = cdf[selected_mask].iloc[[0]]
        fig.add_trace(go.Scatter(
            x=selected_df["x_label"],
            y=selected_df["Prediction"],
            mode="markers",
            marker=dict(size=44, color="rgba(199,163,90,0.20)", line=dict(color="rgba(199,163,90,0.24)", width=2)),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=selected_df["x_label"],
            y=selected_df["Prediction"],
            mode="markers+text",
            marker=dict(size=24, color="#fff8e8", line=dict(color="#c7a35a", width=5)),
            text=["اليوم المختار"],
            textposition="bottom center",
            textfont=dict(size=12, color="#062b25"),
            hovertemplate="<b>اليوم المختار</b><br>%{x}<br>العدد المتوقع: %{y:,.0f}<extra></extra>",
            name="اليوم المختار",
        ))

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=34, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.64)",
        font=dict(family="Cairo", size=12, color="#082E28"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0.01,
            title="",
            font=dict(size=11)
        ),
        xaxis=dict(showgrid=False, title="", zeroline=False),
        yaxis=dict(
            range=[base_y, y_max + pad],
            showgrid=True,
            gridcolor="rgba(120,100,60,0.08)",
            tickformat=",",
            title=dict(text="العدد المتوقع للمعتمرين", font=dict(size=11)),
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor="#062B25",
            bordercolor="#C7A35A",
            font=dict(color="#FFF8E8", family="Cairo", size=12)
        )
    )

    return fig


def level_hero_card(level, prediction, weekday, hijri_date):
    level = normalize_level(level)
    color = LEVEL_COLORS.get(level, "#0F6E52")
    bg = LEVEL_BG.get(level, "#eef8f1")
    H(f"""
    <div class="level-hero" style="--level-color:{color}; --level-bg:{bg};">
        <div class="level-kicker">مستوى الازدحام المتوقع</div>
        <div class="level-value" style="color:{color};">{esc(level)}</div>
        <div class="level-meta">
            <span>العدد المتوقع: <strong>{esc(format_number(prediction))}</strong></span>
            <span>اليوم المختار: <strong>{esc(weekday)}</strong></span>
            <span>التاريخ الهجري: <strong>{esc(hijri_date)}</strong></span>
        </div>
    </div>
    """)


def top_card(title, value, subtitle, level=None):
    if level is None:
        color = "#123F35"
        bg = "rgba(255,255,255,0.82)"
        border = "rgba(201,169,95,0.28)"
        accent = "#C7A35A"
    else:
        level = normalize_level(level)
        color = LEVEL_COLORS.get(level, "#123F35")
        bg = LEVEL_BG.get(level, "rgba(255,255,255,0.82)")
        border = LEVEL_BORDER.get(level, "rgba(201,169,95,0.28)")
        accent = color

    H(f"""
    <div class="metric-card" style="background:{bg}; border-color:{border};">
        <div class="metric-label">{esc(title)}</div>
        <div class="metric-value" style="color:{color};">{esc(value)}</div>
        <div class="metric-subtitle">{esc(subtitle)}</div>
        <div class="metric-accent" style="background:{accent};"></div>
    </div>
    """)


def info_pill(text, icon=""):
    H(f"""
    <div class="small-pill">
        <span class="pill-icon">{esc(icon)}</span>
        <span>{esc(text)}</span>
    </div>
    """)


def best_day_card(row, selected_prediction=None):
    if row is None:
        H('<div class="best-card"><div class="best-title">لا يوجد يوم بديل ضمن الأيام السبعة القريبة</div></div>')
        return

    day_name = esc(row.get("Weekday_AR", "غير متاح"))
    hijri_date = esc(row.get("Hijri_Date", ""))
    visitors_val = float(row.get("Prediction", 0))
    visitors = format_number(visitors_val)

    diff_text = ""
    if selected_prediction is not None and visitors_val > 0:
        diff = ((float(selected_prediction) - visitors_val) / visitors_val) * 100
        if diff > 0:
            diff_text = f"أقل بحوالي {diff:.0f}% من اليوم المختار"
        else:
            diff_text = "الأقرب من حيث انخفاض الازدحام"

    H(f"""
    <div class="best-card">
        <div class="best-kicker">أفضل يوم بديل</div>
        <div class="best-title">{day_name}</div>
        <div class="best-meta">
            <span>التاريخ الهجري: <strong>{hijri_date}</strong></span>
            <span>العدد المتوقع: <strong>{visitors}</strong></span>
        </div>
        <div class="best-note">{esc(diff_text or 'الأقل ازدحامًا ضمن الأيام السبعة القريبة.')}</div>
    </div>
    """)


def quick_summary(level, prediction, best_day):
    best_label = "غير متاح"
    if best_day is not None:
        best_label = str(best_day.get("Weekday_AR", "غير متاح"))
    H(f"""
    <div class="summary-card">
        <div class="summary-title">ملخص سريع</div>
        <div class="summary-grid">
            <div><span>مستوى الازدحام</span><strong>{esc(level)}</strong></div>
            <div><span>العدد المتوقع</span><strong>{esc(format_number(prediction))}</strong></div>
            <div><span>أفضل يوم بديل</span><strong>{esc(best_label)}</strong></div>
        </div>
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
    --gold-700:#A87312;
    --gold-500:#C7A35A;
    --cream-50:#FCFAF4;
    --cream-100:#F5EFE3;
    --ink:#10241F;
    --muted:#66736D;
    --line:rgba(199,163,90,0.24);
}

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
}

.stApp {
    background:
      radial-gradient(circle at 15% 8%, rgba(199,163,90,0.09), transparent 24%),
      radial-gradient(circle at 90% 9%, rgba(15,110,82,0.08), transparent 24%),
      linear-gradient(180deg, #fbfaf5 0%, #f4efe6 100%);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.025;
    background-image:
      linear-gradient(30deg, rgba(8,46,40,0.45) 12%, transparent 12.5%, transparent 87%, rgba(8,46,40,0.45) 87.5%, rgba(8,46,40,0.45)),
      linear-gradient(150deg, rgba(8,46,40,0.45) 12%, transparent 12.5%, transparent 87%, rgba(8,46,40,0.45) 87.5%, rgba(8,46,40,0.45));
    background-size: 92px 160px;
    z-index: 0;
}

.stApp > * { position: relative; z-index: 1; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 1320px !important;
    padding-top: 0.9rem !important;
    padding-bottom: 1.5rem !important;
}

section[data-testid="stSidebar"] {
    width: 170px !important;
    min-width: 170px !important;
    max-width: 170px !important;
    background: linear-gradient(180deg, #082E28 0%, #0B3C34 100%) !important;
    border-left: 1px solid rgba(199,163,90,0.30) !important;
    box-shadow: -10px 0 28px rgba(8,46,40,0.10);
}

section[data-testid="stSidebar"] > div {
    width: 170px !important;
    min-width: 170px !important;
    max-width: 170px !important;
    padding-top: 1.2rem !important;
}

.sidebar-wrap { text-align:center; padding: 18px 8px 14px 8px; }
.sidebar-title { color:#FFF8E8; font-size: 15px; font-weight: 900; }
.sidebar-line { height:1px; background: linear-gradient(90deg, transparent, rgba(216,189,120,0.7), transparent); margin: 18px 20px 22px 20px; }

section[data-testid="stSidebar"] .stButton { display:flex !important; justify-content:center !important; }
section[data-testid="stSidebar"] .stButton button {
    width: 132px !important;
    background: rgba(255,255,255,0.055) !important;
    color: #F7F1E3 !important;
    border: 1px solid rgba(246,231,190,0.12) !important;
    font-weight: 800 !important;
    font-size: 12px !important;
    border-radius: 14px !important;
    height: 38px !important;
    margin-bottom: 12px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton button:hover { background: rgba(199,163,90,0.18) !important; }

.stButton button {
    background: linear-gradient(135deg, #0F6E52, #062B25) !important;
    color: white !important;
    border: 1px solid rgba(199,163,90,0.22) !important;
    box-shadow: 0 14px 26px rgba(8,46,40,0.13) !important;
    border-radius: 16px !important;
    height: 44px !important;
    font-weight: 900 !important;
    font-size: 13px !important;
    transition: all 0.18s ease !important;
}
.stButton button:hover { filter: brightness(1.05); transform: translateY(-1px); }

.main-header {
    position:relative;
    overflow:hidden;
    background: linear-gradient(135deg, rgba(6,43,37,0.98), rgba(16,75,64,0.96));
    border: 1px solid rgba(199,163,90,0.34);
    border-radius: 28px;
    padding: 20px 28px;
    min-height: 98px;
    box-shadow: 0 24px 54px rgba(8,46,40,0.14);
    margin-bottom: 16px;
}
.main-header::before {
    content:"";
    position:absolute;
    inset:0;
    background: radial-gradient(circle at 18% 30%, rgba(199,163,90,0.16), transparent 24%), linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.055) 44%, transparent 72%);
    pointer-events:none;
}
.header-main-title { color:#FFF8E8; font-size:28px; font-weight:900; text-align:center; letter-spacing:-0.7px; position:relative; }
.header-subtitle { color:#D8BD78; font-size:14px; font-weight:800; text-align:center; margin-top:8px; position:relative; }
.header-decor { width:280px; height:2px; background:linear-gradient(90deg, transparent, #D8BD78, transparent); margin:12px auto 0 auto; position:relative; }

.form-shell, .hero-box, .section-box, .metric-card, .reco-box, .best-card, .suggest-box, .summary-card, .level-hero {
    backdrop-filter: blur(14px);
}

.form-shell, .hero-box {
    background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,251,242,0.78)) !important;
    border:1px solid rgba(201,169,95,0.28) !important;
    border-radius:30px !important;
    box-shadow: 0 24px 54px rgba(8,46,40,0.07);
}
.form-shell { padding:30px 34px !important; }
.form-title { text-align:center; color:#062B25; font-size:28px; font-weight:900; letter-spacing:-0.6px; }
.form-sub { text-align:center; color:#A87312; font-size:13px; font-weight:900; margin-top:8px; }
.form-line { width:280px; height:2px; background:linear-gradient(90deg, transparent, #C7A35A, transparent); margin:14px auto 18px auto; }
.form-section { width:fit-content; margin:0 auto 5px auto; color:#0F6E52; background:rgba(15,110,82,0.075); border:1px solid rgba(15,110,82,0.12); border-radius:999px; padding:7px 18px; font-size:13px; font-weight:900; }

.stTextInput input, .stSelectbox > div > div {
    background:#FFFEFA !important;
    border:1px solid rgba(201,169,95,0.36) !important;
    border-radius:16px !important;
    min-height:46px !important;
    box-shadow:0 12px 28px rgba(8,46,40,0.035);
}

div[data-testid="stHorizontalBlock"] { gap:0.85rem !important; }

.level-hero {
    position:relative;
    overflow:hidden;
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,251,242,0.82));
    border:1px solid rgba(201,169,95,0.30);
    border-radius:32px;
    padding:30px 32px;
    min-height:220px;
    box-shadow:0 30px 64px rgba(8,46,40,0.12);
    text-align:center;
}
.level-hero::before {
    content:"";
    position:absolute;
    inset:0;
    background: radial-gradient(circle at 18% 20%, var(--level-bg), transparent 38%);
    opacity:0.75;
}
.level-kicker { position:relative; width:fit-content; margin:0 auto 10px auto; padding:7px 16px; border-radius:999px; background:rgba(8,46,40,0.055); color:#66736D; font-size:12px; font-weight:900; }
.level-value { position:relative; font-size:72px; font-weight:900; line-height:1.05; letter-spacing:-2px; }
.level-meta { position:relative; margin-top:18px; display:flex; flex-wrap:wrap; justify-content:center; gap:10px; }
.level-meta span { padding:7px 12px; border-radius:999px; background:rgba(255,255,255,0.72); color:#33433d; border:1px solid rgba(201,169,95,0.20); font-size:11px; font-weight:800; }

.metric-card {
    position:relative;
    overflow:hidden;
    border:1px solid rgba(201,169,95,0.28);
    border-radius:26px;
    padding:16px 17px;
    min-height:118px;
    box-shadow:0 18px 36px rgba(8,46,40,0.055);
    transition:0.18s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow:0 24px 46px rgba(8,46,40,0.08) !important; }
.metric-label { color:#68766F; font-size:12px; font-weight:900; }
.metric-value { font-size:27px; font-weight:900; line-height:1.18; margin-top:12px; letter-spacing:-0.5px; }
.metric-subtitle { color:#6D756F; font-size:11px; font-weight:800; margin-top:7px; }
.metric-accent { position:absolute; right:0; bottom:0; height:4px; width:100%; opacity:0.65; }

.reco-box {
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,251,242,0.82));
    border:1px solid rgba(201,169,95,0.30);
    border-radius:26px;
    padding:14px 22px;
    box-shadow:0 18px 38px rgba(8,46,40,0.06);
}
.reco-label { text-align:center; color:#786A4E; font-size:11px; font-weight:900; }
.reco-title { text-align:center; font-size:24px; font-weight:900; margin-top:3px; letter-spacing:-0.5px; }
.reco-line { width:132px; height:2px; background:linear-gradient(90deg, transparent, #C7A35A, transparent); margin:7px auto 8px auto; }
.reco-text { text-align:center; color:#31413B; font-size:13px; font-weight:700; line-height:1.75; max-width:1000px; margin:auto; }

.summary-card {
    margin-top:10px;
    background: rgba(255,255,255,0.76);
    border:1px solid rgba(201,169,95,0.24);
    border-radius:24px;
    padding:16px 18px;
    box-shadow:0 18px 38px rgba(8,46,40,0.05);
}
.summary-title { color:#062B25; font-size:14px; font-weight:900; margin-bottom:10px; text-align:right; }
.summary-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; }
.summary-grid div { background:rgba(255,255,255,0.68); border:1px solid rgba(201,169,95,0.18); border-radius:18px; padding:10px 12px; }
.summary-grid span { display:block; color:#68766F; font-size:11px; font-weight:800; }
.summary-grid strong { display:block; color:#062B25; font-size:16px; font-weight:900; margin-top:4px; }

.section-box {
    background:rgba(255,255,255,0.74);
    border:1px solid rgba(201,169,95,0.22);
    border-radius:26px;
    padding:14px 18px;
    box-shadow:0 18px 40px rgba(8,46,40,0.055);
}
.section-title { text-align:right; color:#062B25; font-size:16px; font-weight:900; }
.section-line { width:92px; height:2px; background:linear-gradient(90deg, #C7A35A, transparent); margin:8px 0 8px auto; }

.suggest-box {
    background:linear-gradient(180deg, rgba(6,43,37,0.98), rgba(16,75,64,0.96));
    border:1px solid rgba(199,163,90,0.34);
    border-radius:28px;
    padding:22px 20px;
    text-align:center;
    box-shadow:0 24px 52px rgba(8,46,40,0.16);
}
.side-suggest-box { min-height:145px; display:flex; flex-direction:column; justify-content:center; margin-top:18px; }
.suggest-title { color:#FFF8E8; font-size:21px; font-weight:900; text-align:center; }
.suggest-sub { color:#D8BD78; font-size:12px; font-weight:800; margin-top:10px; line-height:1.8; }

.best-card {
    background:linear-gradient(135deg, rgba(236,247,240,0.98), rgba(255,255,255,0.92));
    border:1px solid rgba(15,139,95,0.22);
    border-radius:26px;
    padding:18px 16px;
    margin-top:14px;
    text-align:center;
    box-shadow:0 20px 42px rgba(15,139,95,0.10);
}
.best-kicker { width:fit-content; margin:0 auto 8px auto; padding:5px 12px; border-radius:999px; background:rgba(15,139,95,0.08); color:#0F8B5F; font-size:10px; font-weight:900; }
.best-title { color:#062B25; font-size:25px; font-weight:900; line-height:1.5; }
.best-meta { margin-top:8px; color:#53615B; font-size:12px; font-weight:800; line-height:1.9; display:flex; flex-direction:column; gap:2px; }
.best-meta strong { color:#062B25; font-weight:900; }
.best-note { margin-top:10px; color:#0F6E52; font-size:11px; font-weight:800; line-height:1.7; }

.small-pill { background:rgba(255,255,255,0.84); border:1px solid rgba(201,169,95,0.32); color:#5D5138; border-radius:999px; padding:8px 14px; min-height:34px; display:flex; align-items:center; justify-content:center; gap:8px; font-size:11px; font-weight:900; box-shadow:0 12px 28px rgba(8,46,40,0.05); }
.pill-icon { min-width:30px; height:24px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center; background:rgba(15,110,82,0.08); color:#0F6E52; font-size:10px; font-weight:900; padding:0 8px; }

@media (max-width:900px) {
    section[data-testid="stSidebar"] { width:145px !important; min-width:145px !important; max-width:145px !important; }
    .header-main-title { font-size:21px; }
    .level-value { font-size:48px; }
    .summary-grid { grid-template-columns:1fr; }
}
</style>
""")


def show_header():
    H("""
    <div class="main-header">
        <div class="header-main-title">التنبؤ بمستويات الازدحام وتوصية أوقات الزيارة</div>
        <div class="header-subtitle">تحليل تنبؤي وتوصية بأفضل أوقات أداء العمرة</div>
        <div class="header-decor"></div>
    </div>
    """)


if "page" not in st.session_state:
    st.session_state.page = "input"

if "entered" not in st.session_state:
    st.session_state.entered = False

if "show_best_day" not in st.session_state:
    st.session_state.show_best_day = True


with st.sidebar:
    H("""
    <div class="sidebar-wrap">
        <div class="sidebar-title">نظام التنبؤ</div>
        <div class="sidebar-line"></div>
    </div>
    """)

    if st.button("إدخال البيانات"):
        st.session_state.page = "input"
        st.rerun()

    if st.button("لوحة النتائج"):
        st.session_state.page = "dashboard"
        st.rerun()


def input_page():
    show_header()

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
        st.session_state.show_best_day = True
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

    main_col, side_col = st.columns([1.45, 1], gap="large")
    with main_col:
        level_hero_card(crowd_level, prediction, weekday, hijri_date)
    with side_col:
        top_card("العدد المتوقع", format_number(prediction), "معتمر")
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        cday, ctemp = st.columns(2)
        with cday:
            top_card("اليوم المختار", weekday, hijri_date)
        with ctemp:
            top_card("درجة الحرارة", temp_text, "متوسط اليوم")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    H(f"""
    <div class="reco-box">
        <div class="reco-label">التوصية النهائية</div>
        <div class="reco-title" style="color:{LEVEL_COLORS.get(crowd_level, '#064b3b')};">{esc(decision)}</div>
        <div class="reco-line"></div>
        <div class="reco-text">{esc(reason)}</div>
    </div>
    """)

    quick_summary(crowd_level, prediction, best_day)

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

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    suggest_col, chart_col = st.columns([0.9, 2.4], gap="large")

    with suggest_col:
        H("""
        <div class="suggest-box side-suggest-box">
            <div class="suggest-title">اليوم البديل</div>
            <div class="suggest-sub">أقل يوم ازدحامًا ضمن الأيام السبعة القريبة.</div>
        </div>
        """)
        best_day_card(best_day, prediction)

    with chart_col:
        H("""
        <div class="section-box">
            <div class="section-title">الاتجاه المتوقع خلال الأيام السبعة القريبة</div>
            <div class="section-line"></div>
        </div>
        """)
        fig = build_chart(df7, month, day, best_day)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        if st.button("إدخال تاريخ جديد", use_container_width=True):
            st.session_state.page = "input"
            st.session_state.show_best_day = True
            st.rerun()


if st.session_state.page == "dashboard":
    dashboard_page()
else:
    input_page()
