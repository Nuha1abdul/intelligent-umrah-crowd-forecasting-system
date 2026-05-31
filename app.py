from pathlib import Path
import html
import re

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="بصيرة | توقع ازدحام المعتمرين",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    APP_DIR / "models" / "growth_rate_xgboost_model.pkl",
    APP_DIR / "growth_rate_xgboost_model.pkl",
    Path("models/growth_rate_xgboost_model.pkl"),
    Path("growth_rate_xgboost_model.pkl"),
]
DATE_COL = "Gregorian_Date"
MONTH_COL = "Hijri_Month"
TARGET_COL = "المعتمرين"
HIJRI_MONTHS = [
    "محرم", "صفر", "ربيع الأول", "ربيع الآخر", "جماد الأول", "جماد الآخر",
    "رجب", "شعبان", "رمضان", "شوال", "ذو القعدة", "ذو الحجة",
]
NATIONALITY_OPTIONS = ["سعودي", "غير سعودي"]
LEVEL_COLORS = {"منخفض": "#159A6B", "متوسط": "#C68B2C", "مرتفع": "#D45B52"}
LEVEL_TINTS = {"منخفض": "#EAF8F0", "متوسط": "#FFF7E5", "مرتفع": "#FFF0EE"}


def H(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def esc(value):
    if value is None or pd.isna(value):
        return ""
    return html.escape(str(value))


def normalize_month_name(month_name):
    text = str(month_name).strip()
    aliases = {
        "ربيع الاول": "ربيع الأول", "ربيع اول": "ربيع الأول",
        "ربيع الاخر": "ربيع الآخر", "ربيع الثاني": "ربيع الآخر",
        "جمادى الأولى": "جماد الأول", "جمادى الاولى": "جماد الأول",
        "جماد الأولى": "جماد الأول", "جماد الاولى": "جماد الأول",
        "جماد الاول": "جماد الأول", "جمادى الآخرة": "جماد الآخر",
        "جمادى الاخرة": "جماد الآخر", "جمادى الثانية": "جماد الآخر",
        "جماد الآخرة": "جماد الآخر", "جماد الاخرة": "جماد الآخر",
        "جماد الاخر": "جماد الآخر", "جماد الثاني": "جماد الآخر",
        "ذو القعده": "ذو القعدة", "ذو الحجه": "ذو الحجة",
    }
    return aliases.get(text, text)


def normalize_level(level):
    text = str(level).strip()
    return {
        "Low": "منخفض", "low": "منخفض", "منخفض": "منخفض",
        "Medium": "متوسط", "medium": "متوسط", "متوسط": "متوسط",
        "High": "مرتفع", "high": "مرتفع", "Critical": "مرتفع",
        "critical": "مرتفع", "حرج": "مرتفع", "عالي": "مرتفع",
        "مرتفع جدًا": "مرتفع",
    }.get(text, text)


def format_number(value, empty="غير متاح"):
    try:
        if pd.isna(value):
            return empty
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return empty


def extract_hijri_day(value):
    if pd.isna(value):
        return np.nan
    nums = [int(number) for number in re.findall(r"\d+", str(value))]
    if not nums:
        return np.nan
    if len(nums) >= 3 and nums[0] >= 1400:
        return nums[2]
    if len(nums) >= 3 and nums[-1] >= 1400:
        return nums[0]
    return nums[0]


def classify_by_quantiles(series):
    values = pd.to_numeric(series, errors="coerce")
    q1, q2 = values.quantile([0.33, 0.66])

    def classify(value):
        if pd.isna(value):
            return "متوسط"
        if value <= q1:
            return "منخفض"
        if value <= q2:
            return "متوسط"
        return "مرتفع"

    return values.apply(classify)


def demo_data():
    dates = pd.date_range("2026-05-18", periods=60, freq="D")
    days = np.tile(np.arange(1, 31), 2)
    months = np.repeat(["ذو القعدة", "ذو الحجة"], 30)
    base = np.array([
        18420, 19240, 20780, 21950, 23200, 24800, 25765, 24100, 22600, 21400,
        19800, 18700, 20150, 22300, 24600, 26800, 25100, 23800, 21600, 20500,
        19400, 20800, 23100, 25900, 27100, 26400, 24500, 22900, 21100, 19600,
    ])
    predictions = np.concatenate([base, base + np.linspace(4200, 9500, 30).round()])
    weekday_map = {
        0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس",
        4: "الجمعة", 5: "السبت", 6: "الأحد",
    }
    return pd.DataFrame({
        DATE_COL: dates,
        MONTH_COL: months,
        "Hijri_Day": days,
        "Hijri_Date": [
            f"{day:02d}-{'11' if month == 'ذو القعدة' else '12'}-1447"
            for day, month in zip(days, months)
        ],
        "Weekday_AR": [weekday_map[date.dayofweek] for date in dates],
        "AvgTemp_C": np.linspace(38.5, 44.0, 60).round(1),
        "Prediction": predictions,
        "Hajj": np.concatenate([np.linspace(9200, 16800, 30), np.linspace(17400, 39000, 30)]).round(),
        "Tawaf_Ifadah": np.concatenate([np.zeros(39), np.linspace(18200, 32600, 21)]).round(),
    })


def normalize_columns(df):
    rename_map = {
        "Hajj_Feature": "Hajj", "Tawaf_Ifadah_Feature": "Tawaf_Ifadah",
        "Tawaf_Ifadah_Featureh": "Tawaf_Ifadah", "Umrah_Count": TARGET_COL,
        "Visitors": TARGET_COL, "Predicted": "Prediction",
        "Prediction_Count": "Prediction", "predicted_visitors": "Prediction",
        "Gregorian Date": DATE_COL, "Hijri Month": MONTH_COL, "Hijri Day": "Hijri_Day",
    }
    return df.rename(columns={column: rename_map.get(column, column) for column in df.columns}).copy()


def add_display_cols(df):
    df = normalize_columns(df)
    if DATE_COL not in df:
        raise ValueError(f"الجدول لا يحتوي على العمود المطلوب: {DATE_COL}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if "Hijri_Date" not in df:
        df["Hijri_Date"] = ""
    if "Hijri_Day" not in df:
        df["Hijri_Day"] = df["Hijri_Date"].apply(extract_hijri_day)
    df["Hijri_Day_Num"] = pd.to_numeric(df["Hijri_Day"], errors="coerce")
    if df["Hijri_Day_Num"].isna().all():
        df["Hijri_Day_Num"] = df["Hijri_Date"].apply(extract_hijri_day)
    if MONTH_COL not in df:
        raise ValueError(f"الجدول لا يحتوي على العمود المطلوب: {MONTH_COL}")
    df[MONTH_COL] = df[MONTH_COL].apply(normalize_month_name)
    if "Prediction" not in df:
        if TARGET_COL not in df:
            raise ValueError("الجدول لا يحتوي على Prediction أو عمود المعتمرين.")
        df["Prediction"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df["Prediction"] = pd.to_numeric(df["Prediction"], errors="coerce")
    if "Crowding_Level" not in df:
        df["Crowding_Level"] = classify_by_quantiles(df["Prediction"])
    else:
        df["Crowding_Level"] = df["Crowding_Level"].apply(normalize_level)
        invalid = ~df["Crowding_Level"].isin(LEVEL_COLORS)
        fallback = classify_by_quantiles(df["Prediction"])
        df.loc[invalid, "Crowding_Level"] = fallback.loc[invalid]
    if "Weekday_AR" not in df:
        names = {
            0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس",
            4: "الجمعة", 5: "السبت", 6: "الأحد",
        }
        df["Weekday_AR"] = df[DATE_COL].dt.dayofweek.map(names)
    for column in ["AvgTemp_C", "Hajj", "Tawaf_Ifadah"]:
        if column not in df:
            df[column] = np.nan if column == "AvgTemp_C" else 0
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return (
        df.dropna(subset=[DATE_COL, "Hijri_Day_Num", "Prediction"])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


@st.cache_data
def load_data():
    load_error = ""
    for model_path in MODEL_PATHS:
        if not model_path.exists():
            continue
        try:
            package = joblib.load(model_path)
            if isinstance(package, pd.DataFrame):
                return add_display_cols(package), False, ""
            if isinstance(package, dict):
                for key in ["test_df", "results_df", "df", "data", "predictions"]:
                    if isinstance(package.get(key), pd.DataFrame):
                        return add_display_cols(package[key]), False, ""
            load_error = "تم العثور على ملف المودل، لكنه لا يحتوي على جدول نتائج مناسب."
        except Exception as error:
            load_error = f"تعذر قراءة ملف المودل: {error}"
    return add_display_cols(demo_data()), True, load_error


def get_available_months(df):
    available = set(df[MONTH_COL].dropna().astype(str))
    ordered = [month for month in HIJRI_MONTHS if month in available]
    return ordered or sorted(available)


def get_available_days(df, month):
    return (
        df.loc[df[MONTH_COL] == month, "Hijri_Day_Num"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )


def get_7_days(df, month, day):
    selected = df[(df[MONTH_COL] == month) & (df["Hijri_Day_Num"] == int(day))]
    if selected.empty:
        return pd.DataFrame()
    start = selected.iloc[0][DATE_COL]
    out = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= start + pd.Timedelta(days=6))].copy()
    out["Local_Crowding_Level"] = classify_by_quantiles(out["Prediction"]).apply(normalize_level)
    return out.reset_index(drop=True)


def get_best_alternative(df7, month, day):
    alternatives = df7[~((df7[MONTH_COL] == month) & (df7["Hijri_Day_Num"] == int(day)))]
    if alternatives.empty:
        return None
    return alternatives.sort_values("Prediction").iloc[0]


def decision_for(level):
    level = normalize_level(level)
    if level == "منخفض":
        return "مناسب للزيارة", "العدد المتوقع منخفض مقارنة بالأيام القريبة، وهذا اليوم خيار مريح لأداء العمرة."
    if level == "مرتفع":
        return "يفضل اختيار يوم آخر", "الازدحام المتوقع مرتفع مقارنة بالأيام القريبة. راجع اليوم البديل المقترح قبل تثبيت موعد الزيارة."
    return "مناسب مع الحذر", "الازدحام المتوقع ضمن المستوى المتوسط. يمكن أداء العمرة مع اختيار الوقت الأنسب خلال اليوم."


def seasonal_metric(row, month, day):
    if month == "ذو القعدة" or (month == "ذو الحجة" and day <= 9):
        return "عدد الحجاج المتوقع", format_number(row.get("Hajj"), "لا يوجد"), "حج"
    if month == "ذو الحجة" and day >= 10:
        return "طواف الإفاضة المتوقع", format_number(row.get("Tawaf_Ifadah"), "لا يوجد"), "طواف"
    return None


def build_chart(df7, month, day):
    chart = df7.copy()
    chart["label"] = (
        chart["Weekday_AR"].astype(str) + "<br>" +
        chart["Hijri_Day_Num"].astype(int).astype(str) + " " +
        chart[MONTH_COL].astype(str)
    )
    colors = chart["Local_Crowding_Level"].map(LEVEL_COLORS)
    low = max(0, chart["Prediction"].min() * 0.86)
    high = chart["Prediction"].max() * 1.13
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart["label"], y=chart["Prediction"], mode="lines",
        line=dict(color="#17664F", width=4, shape="spline"),
        fill="tozeroy", fillcolor="rgba(27,147,104,0.08)",
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=chart["label"], y=chart["Prediction"], mode="markers+text",
        marker=dict(size=15, color=colors, line=dict(color="white", width=3)),
        text=chart["Prediction"].map(format_number), textposition="top center",
        textfont=dict(size=11, color="#24433B"),
        hovertemplate="<b>%{x}</b><br>العدد المتوقع: %{y:,.0f}<extra></extra>",
        showlegend=False,
    ))
    selected = chart[(chart[MONTH_COL] == month) & (chart["Hijri_Day_Num"] == int(day))]
    fig.add_trace(go.Scatter(
        x=selected["label"], y=selected["Prediction"], mode="markers",
        marker=dict(size=28, color="rgba(255,255,255,0)", line=dict(color="#B88830", width=4)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        height=345, margin=dict(l=8, r=8, t=42, b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Cairo", color="#3F554E"),
        xaxis=dict(showgrid=False, title="", tickfont=dict(size=10)),
        yaxis=dict(range=[low, high], showgrid=True, gridcolor="rgba(16,75,64,0.08)", tickformat=",", title=""),
        hoverlabel=dict(bgcolor="#123F35", font_color="white", font_family="Cairo"),
    )
    return fig


H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&display=swap');
:root{--ink:#123F35;--green:#17664F;--deep:#0C382F;--gold:#B88830;--cream:#F8F5ED;--muted:#71817B}
html,body,[class*="css"]{font-family:'Cairo',sans-serif!important;direction:rtl}
.stApp{background:radial-gradient(circle at 10% 7%,rgba(184,136,48,.10),transparent 24%),radial-gradient(circle at 90% 9%,rgba(23,102,79,.10),transparent 27%),linear-gradient(145deg,#FAF8F2,#F1EEE4);color:var(--ink)}
#MainMenu,footer,header,section[data-testid="stSidebar"]{visibility:hidden;display:none!important}
.block-container{max-width:1380px!important;padding:1rem 2rem 2rem!important}
div[data-testid="stHorizontalBlock"]{gap:.85rem!important}
.topbar{position:relative;overflow:hidden;display:flex;align-items:center;justify-content:space-between;padding:18px 23px;border-radius:22px;background:linear-gradient(135deg,#103C33,#1D5A4B);color:#fff;box-shadow:0 16px 38px rgba(18,63,53,.14)}
.topbar:after{content:"";position:absolute;left:-45px;top:-60px;width:180px;height:180px;border:1px solid rgba(225,196,130,.22);border-radius:50%;box-shadow:0 0 0 22px rgba(225,196,130,.04),0 0 0 48px rgba(225,196,130,.03)}
.brand-wrap{display:flex;align-items:center;gap:14px;position:relative;z-index:1}.brand{width:47px;height:47px;border-radius:15px;display:grid;place-items:center;border:1px solid rgba(225,196,130,.42);color:#E1C482;font-size:21px;background:rgba(255,255,255,.05)}
.topbar h1{margin:0;font-size:21px;font-weight:900}.topbar p{margin:2px 0 0;color:#E1C482;font-size:11px;font-weight:700}.status{position:relative;z-index:1;padding:7px 11px;border:1px solid rgba(255,255,255,.12);border-radius:999px;color:#F4E4BC;font-size:10px;font-weight:800;background:rgba(255,255,255,.06)}
.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#43C58C;margin-left:7px;box-shadow:0 0 0 4px rgba(67,197,140,.12)}
.nav-title{text-align:center;color:#78857F;font-size:10px;font-weight:800;margin:10px 0 4px}
.stButton button{height:40px;border-radius:13px!important;background:rgba(255,255,255,.76)!important;color:#31544B!important;border:1px solid rgba(184,136,48,.18)!important;font-weight:800!important;box-shadow:none!important;transition:.18s ease!important}
.stButton button:hover{color:#fff!important;background:#17664F!important;border-color:#17664F!important;transform:translateY(-1px)}
.welcome{display:flex;justify-content:space-between;align-items:center;margin:15px 0 12px;padding:0 4px}.eyebrow{color:#B88830;font-size:11px;font-weight:900}.welcome h2{margin:3px 0 0;color:#123F35;font-size:25px;font-weight:900}.welcome p{margin:4px 0 0;color:#7A8882;font-size:12px;font-weight:700}.date-chip{background:#FFFDFC;border:1px solid rgba(184,136,48,.2);border-radius:999px;padding:8px 14px;color:#63746E;font-size:11px;font-weight:800}
.hero{position:relative;overflow:hidden;margin-top:14px;padding:74px 28px 68px;text-align:center;border:1px solid rgba(184,136,48,.16);border-radius:26px;background:rgba(255,255,255,.52)}.hero:before{content:"";position:absolute;width:250px;height:250px;border-radius:50%;right:-110px;top:-125px;background:rgba(23,102,79,.07)}.hero h2{margin:8px 0 0;color:#123F35;font-size:42px;font-weight:900;letter-spacing:-1px}.hero p{max-width:690px;margin:10px auto 0;color:#71817B;font-size:14px;font-weight:700;line-height:1.9}
.features{margin-top:14px}.feature{min-height:112px;padding:17px;border-radius:19px;background:rgba(255,255,255,.70);border:1px solid rgba(184,136,48,.16)}.feature-num{color:#B88830;font-size:11px;font-weight:900}.feature-title{margin-top:7px;color:#123F35;font-size:15px;font-weight:900}.feature-text{margin-top:4px;color:#7A8882;font-size:11px;font-weight:700;line-height:1.7}
.form-shell{max-width:910px;margin:18px auto 0;padding:23px 25px 17px;border-radius:23px;background:rgba(255,255,255,.70);border:1px solid rgba(184,136,48,.17);box-shadow:0 14px 36px rgba(18,63,53,.05)}.form-head{text-align:center;margin-bottom:14px}.form-head h2{margin:4px 0 0;color:#123F35;font-size:25px;font-weight:900}.form-head p{margin:4px 0 0;color:#7A8882;font-size:11px;font-weight:700}
.stTextInput label,.stSelectbox label{color:#31544B!important;font-weight:800!important;font-size:12px!important}.stTextInput input,div[data-baseweb="select"]>div{min-height:44px!important;border-radius:13px!important;background:#FFFDFC!important;border-color:rgba(184,136,48,.20)!important}
.demo{margin:9px 0;padding:7px 12px;text-align:center;border-radius:999px;color:#9A742B;background:rgba(255,248,229,.72);border:1px solid rgba(184,136,48,.13);font-size:10px;font-weight:700}
.kpi{position:relative;overflow:hidden;min-height:259px;padding:29px;border-radius:25px;background:linear-gradient(135deg,#123F35,#237158);color:#fff;box-shadow:0 18px 42px rgba(18,63,53,.17)}.kpi:after{content:"";position:absolute;width:240px;height:240px;border-radius:50%;left:-78px;bottom:-118px;background:rgba(255,255,255,.05);box-shadow:0 0 0 28px rgba(255,255,255,.025)}
.kpi-tag{display:inline-block;border-radius:999px;padding:6px 11px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.13);font-size:10px;font-weight:800}.kpi h3{margin:20px 0 0;font-size:63px;line-height:1;font-weight:900}.kpi h4{margin:8px 0 0;color:#E1C482;font-size:14px}.kpi-details{display:flex;gap:8px;flex-wrap:wrap;margin-top:23px}.kpi-details span{padding:7px 10px;background:rgba(255,255,255,.08);border-radius:999px;font-size:10px;font-weight:700}
.mini-card{min-height:119px;padding:16px 17px;border-radius:19px;background:rgba(255,255,255,.80);border:1px solid rgba(184,136,48,.18)}.mini-label{color:#7A8882;font-size:10px;font-weight:800}.mini-value{color:#123F35;font-size:24px;font-weight:900;margin-top:10px}.mini-sub{color:#A08451;font-size:10px;font-weight:700;margin-top:2px}
.recommendation{margin:12px 0;padding:16px 20px;border-radius:19px;display:flex;gap:16px;align-items:center;background:#FFFDFC;border:1px solid rgba(184,136,48,.18)}.reco-icon{width:42px;height:42px;display:grid;place-items:center;flex:none;border-radius:14px;font-size:20px;font-weight:900}.reco-label{color:#8C7957;font-size:10px;font-weight:900}.reco-title{font-size:20px;font-weight:900}.reco-text{color:#66766F;font-size:11px;font-weight:700;line-height:1.8;margin-top:2px}
.season{margin:0 0 12px;padding:11px 15px;border-radius:16px;background:rgba(255,255,255,.62);border:1px solid rgba(184,136,48,.16);color:#6C603E;font-size:11px;font-weight:800}.season b{color:#123F35;font-size:14px;margin-right:5px}
.panel{background:rgba(255,255,255,.70);border:1px solid rgba(184,136,48,.16);border-radius:21px;padding:17px 18px;min-height:391px}.panel-title{font-size:15px;font-weight:900;color:#123F35}.panel-sub{font-size:10px;color:#81908A;font-weight:700;margin-top:2px}
.best{margin-top:15px;padding:15px;border-radius:16px;background:#EBF7F0;border:1px solid rgba(27,147,104,.16);text-align:center}.best-label{font-size:10px;color:#1B9368;font-weight:900}.best-day{font-size:22px;color:#123F35;font-weight:900;margin-top:3px}.best-note{font-size:10px;color:#60776F;font-weight:700;margin-top:3px}.best-copy{margin-top:17px;color:#74847E;font-size:11px;font-weight:700;line-height:2}
@media(max-width:800px){.block-container{padding:.8rem!important}.status,.date-chip{display:none}.hero{padding:52px 16px}.hero h2{font-size:30px}.kpi h3{font-size:48px}.welcome{display:block}}
</style>
""")


def render_header():
    H("""<div class="topbar"><div class="brand-wrap"><div class="brand">◇</div><div><h1>بصيرة</h1>
    <p>نظام ذكي للتنبؤ بازدحام المعتمرين واختيار الوقت الأنسب</p></div></div>
    <div class="status"><span class="status-dot"></span>النظام جاهز للتوقع</div></div>""")
    H('<div class="nav-title">التنقل بين أقسام النظام</div>')
    columns = st.columns(3)
    for column, label, page in zip(
        columns,
        ["الرئيسية", "إدخال البيانات", "لوحة النتائج"],
        ["home", "input", "dashboard"],
    ):
        with column:
            if st.button(label, key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()


def render_demo_notice(demo, load_error=""):
    if demo:
        detail = f" {esc(load_error)}" if load_error else ""
        H(f'<div class="demo">وضع العرض التجريبي: أضف ملف المودل داخل مجلد models لتظهر بيانات المشروع الفعلية.{detail}</div>')


def home_page():
    render_header()
    H("""<div class="hero"><div class="eyebrow">تخطيط أفضل قبل الوصول</div>
    <h2>اختر توقيت عمرتك بثقة</h2>
    <p>قراءة ذكية تساعد المعتمر على فهم مستوى الازدحام المتوقع، مقارنة الأيام القريبة،
    والوصول إلى توصية عملية واضحة قبل اختيار موعد الزيارة.</p></div>""")
    columns = st.columns(3)
    cards = [
        ("01", "مؤشر ازدحام واضح", "تصنيف مباشر لليوم المختار: منخفض أو متوسط أو مرتفع."),
        ("02", "توقع عددي يومي", "عرض العدد المتوقع ومقارنته بالأيام السبعة القريبة."),
        ("03", "اقتراح بديل ذكي", "ترشيح يوم أقل ازدحامًا عندما يكون تغيير الموعد أفضل."),
    ]
    for column, (number, title, text) in zip(columns, cards):
        with column:
            H(f"""<div class="feature"><div class="feature-num">{number}</div>
            <div class="feature-title">{title}</div><div class="feature-text">{text}</div></div>""")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1, 1])
    with middle:
        if st.button("ابدأ التوقع الآن", key="start_prediction", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()


def input_page():
    render_header()
    df, demo, load_error = load_data()
    H("""<div class="form-shell"><div class="form-head"><div class="eyebrow">خطوة واحدة فقط</div>
    <h2>بيانات الزيارة</h2><p>اختر التاريخ الهجري لعرض قراءة الازدحام والتوصية المناسبة.</p></div>""")
    render_demo_notice(demo, load_error)
    with st.form("visit_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("الاسم الكامل", placeholder="أدخل اسم المعتمر")
        with c2:
            nationality = st.selectbox("الجنسية", NATIONALITY_OPTIONS)
        c3, c4 = st.columns(2)
        months = get_available_months(df)
        with c3:
            month = st.selectbox("الشهر الهجري", months)
        days = get_available_days(df, month)
        with c4:
            day = st.selectbox("اليوم الهجري", days)
        submitted = st.form_submit_button("عرض التوقع", use_container_width=True)
    H("</div>")
    blocked = month == "ذو القعدة" and nationality == "غير سعودي" and 1 <= int(day) <= 15
    if blocked:
        st.error("لا يمكن عرض النتائج لغير السعودي خلال الفترة من 1 إلى 15 ذو القعدة.")
    if submitted:
        if not name.strip():
            st.warning("الرجاء إدخال الاسم الكامل.")
            return
        if blocked:
            return
        st.session_state.update(
            entered=True,
            selected_name=name.strip(),
            selected_nationality=nationality,
            selected_month=month,
            selected_day=int(day),
            page="dashboard",
        )
        st.rerun()


def dashboard_page():
    render_header()
    if not st.session_state.get("entered"):
        st.warning("ابدأ بإدخال بيانات الزيارة لعرض لوحة النتائج.")
        _, middle, _ = st.columns([1, 1, 1])
        with middle:
            if st.button("الذهاب إلى إدخال البيانات", key="go_input", use_container_width=True):
                st.session_state.page = "input"
                st.rerun()
        return
    df, demo, load_error = load_data()
    month, day = st.session_state.selected_month, int(st.session_state.selected_day)
    df7 = get_7_days(df, month, day)
    if df7.empty:
        st.error("لا توجد بيانات مطابقة للتاريخ المختار داخل ملف المودل.")
        return
    selected = df7[(df7[MONTH_COL] == month) & (df7["Hijri_Day_Num"] == day)]
    if selected.empty:
        st.error("لا توجد بيانات لليوم المختار داخل ملف المودل.")
        return
    row = selected.iloc[0]
    prediction = float(row["Prediction"])
    level = normalize_level(row["Local_Crowding_Level"])
    color = LEVEL_COLORS.get(level, LEVEL_COLORS["متوسط"])
    tint = LEVEL_TINTS.get(level, LEVEL_TINTS["متوسط"])
    weekday = row.get("Weekday_AR", "غير محدد")
    hijri_date = row.get("Hijri_Date") or f"{day} {month}"
    temp = "غير متاحة" if pd.isna(row.get("AvgTemp_C")) else f'{float(row["AvgTemp_C"]):.1f}°C'
    best = get_best_alternative(df7, month, day)
    decision, reason = decision_for(level)
    H(f"""<div class="welcome"><div><div class="eyebrow">لوحة النتائج</div>
    <h2>مرحبًا، {esc(st.session_state.get("selected_name", "ضيف بصيرة"))}</h2>
    <p>هذه قراءة مبسطة للتاريخ المختار مع مقارنة الأيام السبعة القريبة.</p></div>
    <div class="date-chip">التاريخ الهجري المختار: {esc(hijri_date)}</div></div>""")
    render_demo_notice(demo, load_error)
    main, side = st.columns([1.45, 1])
    with main:
        H(f"""<div class="kpi"><span class="kpi-tag">التوقع الرئيسي</span>
        <h3 style="color:{color}">{esc(level)}</h3><h4>مستوى الازدحام المتوقع</h4>
        <div class="kpi-details"><span>العدد المتوقع: <b>{format_number(prediction)}</b></span>
        <span>اليوم: <b>{esc(weekday)}</b></span><span>التاريخ: <b>{esc(hijri_date)}</b></span></div></div>""")
    with side:
        c1, c2 = st.columns(2)
        with c1:
            H(f"""<div class="mini-card"><div class="mini-label">عدد المعتمرين المتوقع</div>
            <div class="mini-value">{format_number(prediction)}</div><div class="mini-sub">معتمر</div></div>""")
        with c2:
            H(f"""<div class="mini-card"><div class="mini-label">اليوم المختار</div>
            <div class="mini-value">{esc(weekday)}</div><div class="mini-sub">{esc(hijri_date)}</div></div>""")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        H(f"""<div class="mini-card"><div class="mini-label">درجة الحرارة المتوقعة</div>
        <div class="mini-value">{esc(temp)}</div><div class="mini-sub">متوسط اليوم</div></div>""")
    H(f"""<div class="recommendation"><div class="reco-icon" style="color:{color};background:{tint}">◇</div>
    <div><div class="reco-label">التوصية النهائية</div><div class="reco-title" style="color:{color}">{esc(decision)}</div>
    <div class="reco-text">{esc(reason)}</div></div></div>""")
    metric = seasonal_metric(row, month, day)
    if metric:
        label, value, badge = metric
        H(f'<div class="season">{badge} · {esc(label)} <b>{esc(value)}</b></div>')
    chart_col, best_col = st.columns([2.2, 1])
    with chart_col:
        H('<div class="panel-title">اتجاه الازدحام خلال 7 أيام</div><div class="panel-sub">مقارنة يومية للعدد المتوقع من المعتمرين</div>')
        st.plotly_chart(build_chart(df7, month, day), use_container_width=True, config={"displayModeBar": False})
    with best_col:
        if best is None:
            H("""<div class="panel"><div class="panel-title">أفضل وقت قريب</div>
            <div class="panel-sub">لا يوجد يوم بديل ضمن البيانات المتاحة</div></div>""")
        else:
            H(f"""<div class="panel"><div class="panel-title">أفضل وقت قريب</div>
            <div class="panel-sub">الأقل ازدحامًا ضمن الأيام السبعة القادمة</div>
            <div class="best"><div class="best-label">اليوم المقترح</div>
            <div class="best-day">{esc(best["Weekday_AR"])}</div>
            <div class="best-note">{esc(best["Hijri_Date"])} · {format_number(best["Prediction"])} معتمر</div></div>
            <div class="best-copy">تم اختيار هذا اليوم آليًا لأنه يسجل أقل عدد متوقع في الفترة القريبة،
            ويمكن استخدامه كبديل عملي عند ارتفاع الازدحام.</div></div>""")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1, 1])
    with middle:
        if st.button("اختيار تاريخ جديد", key="new_date", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()


st.session_state.setdefault("page", "home")
st.session_state.setdefault("entered", False)
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "input":
    input_page()
else:
    dashboard_page()
