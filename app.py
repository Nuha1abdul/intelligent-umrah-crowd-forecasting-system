
from pathlib import Path
from textwrap import dedent
import glob
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
 
 
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
 
DATA_FILE_NAME = "Intelligent System for Predicting Visitor Crowd Levels and Optimal Visiting Times at Al-Masjid Al-Haram.xlsx"
 
MODEL_PATHS = [
    Path("models/growth_rate_xgboost_model.pkl"),
    Path("growth_rate_xgboost_model.pkl")
]
 
DATE_COL = "Gregorian_Date"
MONTH_COL = "Hijri_Month"
TARGET_COL = "المعتمرين"
 
HIJRI_MONTHS = [
    "محرم", "صفر", "ربيع الأول", "ربيع الآخر",
    "جماد الأول", "جماد الآخر", "رجب", "شعبان",
    "رمضان", "شوال", "ذو القعدة", "ذو الحجة"
]
 
NATIONALITY_OPTIONS = ["سعودي", "غير سعودي"]
 
 
# =====================================================
# Safe HTML
# =====================================================
 
def H(markup: str):
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)
 
 
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
 
 
def format_influence_value(value):
    try:
        if pd.isna(value):
            return "غير متاح"
 
        value = float(value)
 
        if value <= 0:
            return "لا يوجد"
 
        if value == 1:
            return "موجود"
 
        return f"{int(round(value)):,}"
    except Exception:
        value = str(value).strip()
        return value if value else "غير متاح"
 
 
# =====================================================
# Data Helpers
# =====================================================
 
def display_flag(value):
    try:
        if pd.isna(value):
            return "لا"
        return "نعم" if float(value) > 0 else "لا"
    except Exception:
        return "نعم" if str(value).strip() in ["1", "نعم", "yes", "Yes", "True", "true"] else "لا"
 
 
def is_hajj_related_month(month_name):
    return str(month_name).strip() in ["ذو القعدة", "ذو الحجة"]
 
 
def find_data_file():
    exact_path = Path(DATA_FILE_NAME)
 
    if exact_path.exists():
        return exact_path
 
    matches = glob.glob(
        "Intelligent System for Predicting Visitor Crowd Levels and Optimal Visiting Times at Al-Masjid Al-Haram*.xlsx"
    )
 
    if matches:
        return Path(matches[0])
 
    return None
 
 
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
 
 
def normalize_columns(df):
    df = df.copy()
 
    rename_map = {
        "Hajj_Feature": "Hajj",
        "Tawaf_Ifadah_Feature": "Tawaf_Ifadah",
        "Tawaf_Ifadah_Featureh": "Tawaf_Ifadah",
        "Umrah_Count": "المعتمرين",
        "Visitors": "المعتمرين",
        "Gregorian Date": "Gregorian_Date",
        "Hijri Month": "Hijri_Month",
        "Hijri Day": "Hijri_Day",
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
 
    df["Hijri_Day_Num"] = df["Hijri_Date"].apply(extract_hijri_day)
 
    if df["Hijri_Day_Num"].isna().all() and "Hijri_Day" in df.columns:
        df["Hijri_Day_Num"] = pd.to_numeric(df["Hijri_Day"], errors="coerce")
 
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
 
    if "Weather_Category" not in df.columns:
        df["Weather_Category"] = "غير محدد"
 
    if "Hajj_Count_For_Recommendation" not in df.columns:
        if "Hajj" in df.columns:
            df["Hajj_Count_For_Recommendation"] = pd.to_numeric(
                df["Hajj"], errors="coerce"
            ).fillna(0)
        else:
            df["Hajj_Count_For_Recommendation"] = 0
 
    if "Tawaf_Ifadah_Count_For_Recommendation" not in df.columns:
        if "Tawaf_Ifadah" in df.columns:
            df["Tawaf_Ifadah_Count_For_Recommendation"] = pd.to_numeric(
                df["Tawaf_Ifadah"], errors="coerce"
            ).fillna(0)
        else:
            df["Tawaf_Ifadah_Count_For_Recommendation"] = 0
 
    df["Hajj_Display"] = df["Hajj_Count_For_Recommendation"].apply(display_flag)
    df["Tawaf_Display"] = df["Tawaf_Ifadah_Count_For_Recommendation"].apply(display_flag)
 
    if "Prediction" not in df.columns:
        if TARGET_COL in df.columns:
            df["Prediction"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        elif "predicted_visitors" in df.columns:
            df["Prediction"] = pd.to_numeric(df["predicted_visitors"], errors="coerce")
        else:
            st.error("لا يوجد عمود Prediction أو عمود المعتمرين في ملف البيانات.")
            st.stop()
 
    df["Prediction"] = pd.to_numeric(df["Prediction"], errors="coerce")
 
    if "Crowding_Level" not in df.columns:
        q1 = df["Prediction"].quantile(0.33)
        q2 = df["Prediction"].quantile(0.66)
 
        df["Crowding_Level"] = np.where(
            df["Prediction"] <= q1,
            "منخفض",
            np.where(df["Prediction"] <= q2, "متوسط", "مرتفع")
        )
 
    return df
 
 
def load_package():
    model_path = get_existing_model_path()
 
    if model_path is not None:
        package = joblib.load(model_path)
        if isinstance(package, dict):
            return package
        return {"model": package}
 
    data_file = find_data_file()
 
    if data_file is not None:
        return {"test_df": pd.read_excel(data_file)}
 
    st.error("لم يتم العثور على ملف المودل أو ملف البيانات.")
    st.stop()
 
 
# -----------------------------------------------------
# Deployment loader (cached) — loads the trained model
# package ONCE per server session. The model is NOT
# retrained on startup or on widget interactions. It is
# only trained a single time as a fallback if the .pkl
# artifact is missing (e.g. a fresh clone that did not
# commit it), then persisted so it is never retrained.
# -----------------------------------------------------
 
@st.cache_resource(show_spinner="جارٍ تحميل النموذج ...")
def get_model_package():
    model_path = get_existing_model_path()
 
    # Normal path: a pre-trained .pkl exists -> just load it.
    if model_path is not None:
        package = joblib.load(model_path)
        return package if isinstance(package, dict) else {"model": package}
 
    # Fallback (only if absolutely necessary): no .pkl committed.
    # Train exactly once and save it; the cache keeps it for the
    # rest of the session so this never runs again.
    data_file = find_data_file()
    if data_file is not None:
        try:
            import train_model as _train_model
            return _train_model.train_model()
        except Exception:
            return load_package()
 
    return load_package()
 
 
def get_test_df_from_package(package):
    if "test_df" in package:
        return add_display_cols(package["test_df"])
 
    if "results_df" in package:
        return add_display_cols(package["results_df"])
 
    if "df" in package:
        return add_display_cols(package["df"])
 
    data_file = find_data_file()
 
    if data_file is not None:
        return add_display_cols(pd.read_excel(data_file))
 
    st.error("لم يتم العثور على بيانات الاختبار.")
    st.stop()
 
 
def get_7_day_comparison(test_df, selected_month, selected_day):
    start_day = int(selected_day)
    end_day = min(start_day + 6, 30)
 
    comparison_df = test_df[
        (test_df[MONTH_COL].astype(str).str.strip() == str(selected_month).strip()) &
        (test_df["Hijri_Day_Num"] >= start_day) &
        (test_df["Hijri_Day_Num"] <= end_day)
    ].copy()
 
    return comparison_df, start_day, end_day
 
 
def get_best_alternative(comparison_df, selected_day):
    alternatives = comparison_df[comparison_df["Hijri_Day_Num"] != int(selected_day)].copy()
 
    if alternatives.empty:
        return None
 
    return alternatives.sort_values("Prediction").iloc[0]
 
 
def get_weather_reason(row):
    try:
        temp = float(row.get("AvgTemp_C", np.nan))
    except Exception:
        temp = np.nan
 
    if pd.isna(temp):
        return "لا توجد معلومات كافية عن درجة الحرارة، لذلك تم بناء التوصية بشكل أساسي على مستوى الازدحام المتوقع."
 
    if temp >= 40:
        return f"درجة الحرارة مرتفعة جدًا تقريبًا {temp:.1f}°C، لذلك يفضل تجنب وقت الظهيرة واختيار وقت مبكر جدًا أو بعد العشاء."
    elif temp >= 35:
        return f"درجة الحرارة مرتفعة تقريبًا {temp:.1f}°C، لذلك يفضل اختيار وقت مبكر لتقليل الإجهاد الحراري."
    elif temp >= 28:
        return f"درجة الحرارة مناسبة نسبيًا تقريبًا {temp:.1f}°C، لكن الازدحام المتوقع يظل العامل الأهم في القرار."
    else:
        return f"درجة الحرارة مناسبة تقريبًا {temp:.1f}°C، وهذا يدعم إمكانية الزيارة إذا كان الازدحام مناسبًا."
 
 
def get_decision(row, comparison_df, selected_month):
    avg_prediction = comparison_df["Prediction"].mean()
    selected_prediction = row["Prediction"]
 
    reasons = []
 
    if selected_prediction <= avg_prediction * 0.90:
        decision = "مناسب للزيارة"
        level = "منخفض"
        reasons.append("الازدحام المتوقع أقل من متوسط الأيام السبعة القريبة، لذلك يعتبر اليوم مناسبًا نسبيًا للزيارة.")
    elif selected_prediction >= avg_prediction * 1.15:
        decision = "يفضل اختيار يوم آخر"
        level = "مرتفع"
        reasons.append("الازدحام المتوقع أعلى من متوسط الأيام السبعة القريبة، لذلك يفضل اختيار يوم أقل ازدحامًا.")
    else:
        decision = "مناسب مع الحذر"
        level = "متوسط"
        reasons.append("الازدحام المتوقع متوسط مقارنة بالأيام السبعة القريبة، لذلك يمكن الزيارة مع اختيار وقت مناسب.")
 
    reasons.append(f"عدد المعتمرين المتوقعين في هذا اليوم هو تقريبًا {format_number(selected_prediction)} معتمر.")
 
    if is_hajj_related_month(selected_month):
        hajj_value = row.get("Hajj_Count_For_Recommendation", 0)
        tawaf_value = row.get("Tawaf_Ifadah_Count_For_Recommendation", 0)
 
        if display_flag(hajj_value) == "نعم":
            reasons.append(
                f"كما تظهر البيانات وجود تأثير مرتبط بالحج خلال هذا التاريخ، وقيمة/عدد مؤشر الحج في البيانات: {format_influence_value(hajj_value)}."
            )
 
        if display_flag(tawaf_value) == "نعم":
            reasons.append(
                f"كما تظهر البيانات وجود تأثير لطواف الإفاضة خلال هذا التاريخ، وقيمة/عدد مؤشر طواف الإفاضة في البيانات: {format_influence_value(tawaf_value)}."
            )
 
    reasons.append(get_weather_reason(row))
 
    return {
        "decision": decision,
        "level": level,
        "reason": " ".join(reasons),
    }
 
 
def get_row_recommendation(row, avg_prediction):
    prediction = row["Prediction"]
 
    if prediction <= avg_prediction * 0.90:
        return "أفضل للزيارة"
 
    if prediction >= avg_prediction * 1.15:
        return "يفضل تجنب الزيارة"
 
    return "يمكن الذهاب مع الحذر"
 
 
def prepare_display_table(df, show_hajj_info):
    display_df = df.copy()
    avg_prediction = display_df["Prediction"].mean()
 
    display_df["Recommendation_Text"] = display_df.apply(
        lambda r: get_row_recommendation(r, avg_prediction),
        axis=1
    )
 
    cols = [
        "Hijri_Date",
        "Weekday_AR",
        "Prediction",
        "Crowding_Level",
    ]
 
    rename_map = {
        "Hijri_Date": "التاريخ الهجري",
        "Weekday_AR": "اليوم",
        "Prediction": "المعتمرون المتوقعون",
        "Crowding_Level": "الازدحام",
    }
 
    display_df = display_df[cols].rename(columns=rename_map)
 
    display_df["المعتمرون المتوقعون"] = display_df["المعتمرون المتوقعون"].apply(
        lambda x: f"{int(float(x)):,}" if not pd.isna(x) else "غير متاح"
    )
 
    return display_df
 
 
def df_to_html_table(df):
    header_html = "".join([f"<th>{esc(col)}</th>" for col in df.columns])
    rows = []
 
    for _, row in df.iterrows():
        tds = "".join([f"<td>{esc(val)}</td>" for val in row.values])
        rows.append(f"<tr>{tds}</tr>")
 
    return f"""
    <table class="clean-table">
        <thead>
            <tr>{header_html}</tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """
 
 
# =====================================================
# CSS
# =====================================================
 
H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800;900&display=swap');
 
/* ============================================================
   Design tokens
   ============================================================ */
:root{
    --green-900:#064b3b;
    --green-700:#0b6b53;
    --gold-600:#a97908;
    --gold-500:#c99a2e;
    --gold-400:#d4a53a;
    --ink-700:#2f2b1d;
    --ink-500:#4a3d28;
    --cream-card:#fbf6ec;
    --line-gold:#d6bd7f;
    --shadow-sm:0 8px 22px rgba(80,58,20,0.05);
    --shadow-md:0 10px 24px rgba(88,66,32,0.055);
    --shadow-lg:0 16px 34px rgba(88,66,32,0.07);
    --radius-lg:24px;
    --radius-xl:28px;
}
 
/* ============================================================
   Base + Arabic typography
   ============================================================ */
html, body, [class*="css"] {
    font-family: 'Cairo', Tahoma, Arial, sans-serif !important;
    direction: rtl;
}
 
html, body {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}
 
/* Arabic reads best with normal tracking */
* { letter-spacing: 0 !important; }
 
.stApp {
    background: linear-gradient(180deg, #fbf8ef 0%, #f5efdf 100%) !important;
}
 
#MainMenu, header, footer {
    visibility: hidden;
}
 
/* Long Arabic strings never push a card sideways */
.hero-text, .reason-text, .best-note, .feature-text,
.mini-value, .question-text, .header-title, .hero-title {
    overflow-wrap: break-word;
    word-wrap: break-word;
}
 
.block-container {
    max-width: 1450px !important;
    padding-top: 1rem !important;
    padding-left: clamp(0.75rem, 2vw, 1.4rem) !important;
    padding-right: clamp(0.75rem, 2vw, 1.4rem) !important;
    padding-bottom: 2.4rem !important;
}
 
/* ============================================================
   Sidebar
   ============================================================ */
section[data-testid="stSidebar"] {
    width: 184px !important;
    min-width: 184px !important;
    max-width: 184px !important;
    background: rgba(238, 228, 207, 0.78) !important;
    border-left: 1px solid rgba(196, 164, 99, 0.45) !important;
}
 
section[data-testid="stSidebar"] > div {
    background: rgba(238, 228, 207, 0.78) !important;
    padding-top: 18px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
}
 
.sidebar-card {
    background: rgba(255,255,255,0.28);
    border: 1px solid rgba(201,173,112,0.50);
    border-radius: 22px;
    padding: 16px 8px;
    text-align: center;
    margin-bottom: 22px;
}
 
.sidebar-logo {
    width: 64px;
    height: 64px;
    border-radius: 20px;
    background: rgba(248,240,220,0.85);
    border: 1px solid #d8bd7e;
    margin: auto;
    display: grid;
    place-items: center;
    font-size: 30px;
}
 
.sidebar-title {
    margin-top: 14px;
    color: var(--green-900);
    font-size: 21px;
    font-weight: 900;
}
 
.sidebar-subtitle {
    color: #7a5a24;
    font-size: 11px;
    font-weight: 700;
    margin-top: 3px;
}
 
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #b58a2a, transparent);
    margin: 10px 6px 18px 6px;
}
 
/* Sidebar buttons without visible boxes */
section[data-testid="stSidebar"] .stButton button {
    height: 44px !important;
    width: 100% !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #5a3e1a !important;
    font-size: 13.5px !important;
    font-weight: 800 !important;
    text-align: right !important;
    padding-right: 12px !important;
    transition: background .18s ease, color .18s ease;
}
 
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.40) !important;
    color: var(--green-900) !important;
    border-radius: 12px !important;
}
 
/* ============================================================
   Main action buttons
   ============================================================ */
.stButton button {
    background: linear-gradient(135deg, #064b3b, #0b6b53) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    height: 48px !important;
    box-shadow: 0 8px 18px rgba(6,75,59,0.18) !important;
    transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
}
 
.stButton button:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
    box-shadow: 0 12px 24px rgba(6,75,59,0.24) !important;
}
 
.stDownloadButton button {
    border-radius: 14px !important;
    font-weight: 800 !important;
    height: 48px !important;
}
 
/* ============================================================
   Header
   ============================================================ */
.main-header {
    background: linear-gradient(90deg, #ffffff 0%, #fffdf8 52%, #f6eddb 100%);
    border: 1px solid #dcc58e;
    border-radius: 26px;
    min-height: 132px;
    padding: clamp(18px, 2.4vw, 24px) clamp(20px, 3vw, 34px);
    box-shadow: var(--shadow-lg);
    margin-bottom: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
}
 
.header-logo {
    width: 78px;
    height: 78px;
    border-radius: 22px;
    background: #f5ead0;
    border: 1px solid #dfc98f;
    display: grid;
    place-items: center;
    font-size: 38px;
    flex-shrink: 0;
}
 
.header-title {
    color: var(--green-900);
    font-size: clamp(20px, 3.4vw, 30px);
    font-weight: 900;
    text-align: center;
    line-height: 1.3;
}
 
.header-subtitle {
    color: var(--gold-600);
    font-size: clamp(12px, 1.6vw, 14px);
    font-weight: 800;
    margin-top: 8px;
    text-align: center;
}
 
.header-line {
    width: min(360px, 80%);
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold-500), transparent);
    margin: 14px auto 0 auto;
}
 
/* ============================================================
   Hero
   ============================================================ */
.hero-box {
    background: var(--cream-card);
    border: 1px solid var(--line-gold);
    border-radius: var(--radius-xl);
    padding: clamp(28px, 5vw, 48px) clamp(20px, 4vw, 36px);
    text-align: center;
    box-shadow: var(--shadow-lg);
    margin-bottom: 24px;
}
 
.hero-icon {
    width: 82px;
    height: 82px;
    margin: 0 auto 16px auto;
    border-radius: 50%;
    background: rgba(212,165,58,0.11);
    border: 1px solid rgba(212,165,58,0.42);
    display: grid;
    place-items:center;
    font-size: 38px;
}
 
.hero-title {
    color: var(--green-900);
    font-size: clamp(22px, 4vw, 32px);
    font-weight: 900;
    line-height: 1.3;
}
 
.hero-subtitle {
    color: var(--gold-600);
    font-size: clamp(14px, 2vw, 17px);
    font-weight: 800;
    margin-top: 10px;
}
 
.hero-text {
    color: var(--ink-500);
    font-size: clamp(13.5px, 1.7vw, 15px);
    line-height: 2;
    font-weight: 600;
    margin-top: 14px;
    max-width: 760px;
    margin-left: auto;
    margin-right: auto;
}
 
/* ============================================================
   Feature cards (equal height + aligned)
   ============================================================ */
.feature-card {
    background: var(--cream-card);
    border: 1px solid var(--line-gold);
    border-radius: 22px;
    padding: 24px 18px;
    min-height: 150px;
    height: 100%;
    text-align: center;
    box-shadow: var(--shadow-md);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: transform .18s ease, box-shadow .18s ease;
}
 
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 30px rgba(88,66,32,0.10);
}
 
.feature-icon {
    font-size: 30px;
    margin-bottom: 10px;
}
 
.feature-title {
    color: var(--green-900);
    font-size: 16px;
    font-weight: 900;
}
 
.feature-text {
    color: #7b6a40;
    font-size: 12.5px;
    font-weight: 600;
    margin-top: 8px;
    line-height: 1.7;
}
 
/* Make Streamlit column children stretch so cards align */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex;
    flex-direction: column;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
    height: 100%;
}
 
/* ============================================================
   Form + inputs
   ============================================================ */
div[data-testid="stForm"] {
    background: var(--cream-card) !important;
    border: 1px solid var(--line-gold) !important;
    border-radius: 30px !important;
    padding: clamp(24px, 4vw, 38px) clamp(20px, 4vw, 44px) clamp(22px, 3vw, 32px) clamp(20px, 4vw, 44px) !important;
    box-shadow: 0 16px 38px rgba(88,66,32,0.09) !important;
    position: relative !important;
    overflow: hidden !important;
}
 
div[data-testid="stForm"]::before {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 100%;
    height: 6px;
    background: linear-gradient(90deg, #064b3b, #d4a53a, #064b3b);
}
 
.input-title {
    text-align: center;
    color: var(--green-900);
    font-size: clamp(22px, 3.4vw, 29px);
    font-weight: 900;
    margin-top: 10px;
    line-height: 1.3;
}
 
.input-subtitle {
    text-align: center;
    color: var(--gold-600);
    font-size: clamp(12px, 1.6vw, 14px);
    font-weight: 800;
    margin-top: 8px;
}
 
.input-line {
    width: min(360px, 80%);
    height: 2px;
    margin: 14px auto 26px auto;
    background: linear-gradient(90deg, transparent, #b58a2a, transparent);
}
 
.input-section-title {
    text-align: center;
    color: var(--green-900);
    font-size: clamp(19px, 2.8vw, 25px);
    font-weight: 900;
    margin-bottom: 24px;
}
 
.stTextInput input {
    background: #fffefa !important;
    border: 1px solid #d5bd82 !important;
    border-radius: 13px !important;
    min-height: 50px !important;
    font-family: 'Cairo', Tahoma, Arial, sans-serif !important;
    font-size: 14px !important;
    text-align: right !important;
}
 
.stSelectbox > div > div {
    background: #fffefa !important;
    border: 1px solid #d5bd82 !important;
    border-radius: 13px !important;
    min-height: 50px !important;
    font-family: 'Cairo', Tahoma, Arial, sans-serif !important;
    font-size: 14px !important;
}
 
label {
    color: #4d3517 !important;
    font-weight: 800 !important;
    font-size: 14px !important;
}
 
/* ============================================================
   Dashboard cards (chart + table) — unified, aligned
   ============================================================ */
.best-card, .better-question-card, .table-card {
    background: rgba(255,255,255,0.60);
    border: 1px solid var(--line-gold);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    padding: 20px;
}
 
/* Chart title is a clean header above the chart (no floating box),
   and the plotly element itself becomes the single chart card. This
   avoids a visible seam between two separate Streamlit elements. */
.chart-card {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 2px 6px 4px 6px;
    margin-bottom: 6px;
}
 
div[data-testid="stPlotlyChart"]{
    background: rgba(255,255,255,0.60);
    border: 1px solid var(--line-gold);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    padding: 10px 14px 14px 14px;
}
 
.table-card {
    height: 100%;
}
 
.section-title {
    color: var(--green-900);
    font-size: clamp(17px, 2.2vw, 20px);
    font-weight: 900;
    text-align: center;
    margin-bottom: 10px;
}
 
.title-line {
    width: 86px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold-500), transparent);
    margin: 0 auto 16px auto;
}
 
/* ============================================================
   Clean table
   ============================================================ */
.table-scroll { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }
 
.clean-table {
    width: 100%;
    border-collapse: collapse;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #c9ad70;
    background: #fffefa;
}
 
.clean-table th {
    background: #f2e3bd;
    color: #6b5730;
    font-weight: 800;
    font-size: 12px;
    padding: 10px 9px;
    white-space: nowrap;
}
 
.clean-table td {
    padding: 10px 9px;
    border-bottom: 1px solid #eee1bd;
    color: #234138;
    font-weight: 600;
    font-size: 12px;
    text-align: center;
}
 
.clean-table tbody tr:nth-child(even) td { background: #fffdf7; }
.clean-table tbody tr:last-child td { border-bottom: none; }
 
/* ============================================================
   Prediction summary box (migrated from iframe -> responsive DOM)
   ============================================================ */
.summary-box {
    background: rgba(255, 253, 247, 0.96);
    border: 1px solid #d8c79d;
    border-radius: var(--radius-xl);
    padding: clamp(18px, 3vw, 26px) clamp(18px, 3vw, 30px);
    box-shadow: 0 10px 28px rgba(80, 58, 20, 0.055);
}
 
.cards-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
 
.mini-card {
    border: 1px solid #e2d2a6;
    background: rgba(255,255,255,0.62);
    border-radius: 20px;
    padding: 14px 18px;
    min-height: 78px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
 
.mini-icon {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    background: #f5ead0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
 
.mini-label {
    color: #6d6042;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 4px;
}
 
.mini-value {
    color: var(--green-900);
    font-size: clamp(17px, 2.2vw, 21px);
    font-weight: 900;
    line-height: 1.25;
}
 
.reason-area {
    border-top: 1px solid rgba(216,199,157,0.65);
    padding-top: 22px;
    text-align: center;
}
 
.reason-title {
    color: var(--green-900);
    font-size: clamp(19px, 2.6vw, 24px);
    font-weight: 900;
    margin-bottom: 12px;
}
 
.reason-line {
    width: min(120px, 60%);
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold-500), transparent);
    margin: 0 auto 22px auto;
}
 
.reason-text {
    color: var(--ink-700);
    font-size: clamp(13px, 1.6vw, 14px);
    font-weight: 700;
    line-height: 2.05;
    max-width: 1080px;
    margin: auto;
}
 
.chips {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 22px;
}
 
.chip {
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    border: 1px solid #e4c87c;
    background: #fff6df;
    color: #8a6415;
}
 
.chip.green { background: #edf8ef; border-color: #b7d8b9; color: #1f7a4d; }
.chip.blue  { background: #eef7f7; border-color: #b9d8df; color: #0d6b69; }
.chip.gold  { background: #fff6df; border-color: #e4c87c; color: #8a6415; }
 
.summary-foot {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid rgba(216,199,157,0.45);
    color: #8a8168;
    font-size: 11.5px;
    font-weight: 700;
    text-align: center;
}
 
/* ============================================================
   Better-day question card (migrated from iframe)
   ============================================================ */
.question-card {
    background: rgba(255, 253, 247, 0.96);
    border: 1px solid #d8c79d;
    border-radius: var(--radius-lg);
    padding: 22px 20px;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
 
.question-title {
    color: var(--green-900);
    font-size: clamp(18px, 2.4vw, 22px);
    font-weight: 900;
    margin-bottom: 8px;
}
 
.question-text {
    color: #7a5a24;
    font-size: clamp(12px, 1.5vw, 13px);
    font-weight: 700;
    line-height: 1.8;
}
 
/* ============================================================
   Best-day answer card (migrated from iframe)
   ============================================================ */
.best-card {
    background: rgba(255, 253, 247, 0.96);
    border: 1px solid #d8c79d;
    border-radius: var(--radius-lg);
    padding: clamp(22px, 3vw, 28px) 20px;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
 
.best-icon {
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: #f3ead6;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 14px auto;
    font-size: 28px;
}
 
.best-label {
    color: var(--green-900);
    font-size: clamp(16px, 2.2vw, 18px);
    font-weight: 900;
    line-height: 1.5;
}
 
.best-subtitle {
    color: #7a5a24;
    font-size: 13px;
    font-weight: 800;
    margin-top: 6px;
}
 
.best-day {
    color: var(--green-900);
    font-size: clamp(26px, 4vw, 34px);
    font-weight: 900;
    margin-bottom: 6px;
}
 
.best-date {
    color: #b18a2e;
    font-size: clamp(15px, 2vw, 18px);
    font-weight: 900;
    margin-bottom: 18px;
}
 
.best-note {
    color: #43574f;
    font-size: 13px;
    font-weight: 700;
    line-height: 1.9;
    border-top: 1px solid #e4d4a8;
    padding-top: 16px;
    max-width: 700px;
    margin: auto;
}
 
/* ============================================================
   Responsive breakpoints
   ============================================================ */
@media (max-width: 1100px) {
    .main-header { min-height: auto; }
}
 
@media (max-width: 900px) {
    .cards-grid { grid-template-columns: 1fr 1fr; }
    .hero-box { padding: 30px 22px; }
    .summary-box { padding: 20px 18px; }
}
 
@media (max-width: 600px) {
    .block-container { padding-bottom: 1.6rem !important; }
    .main-header {
        flex-direction: column;
        text-align: center;
        gap: 12px;
        padding: 18px 16px;
    }
    .header-logo { width: 64px; height: 64px; font-size: 32px; }
    .cards-grid { grid-template-columns: 1fr; gap: 12px; }
    .mini-card { min-height: 66px; padding: 12px 14px; }
    .chips { gap: 8px; }
    .chip { font-size: 11px; padding: 6px 11px; }
    .reason-text { line-height: 1.95; }
    .feature-card { min-height: 0; padding: 20px 16px; }
    div[data-testid="stForm"] { border-radius: 22px !important; }
}
 
/* On phones Streamlit auto-stacks columns; keep the sidebar usable */
@media (max-width: 640px) {
    section[data-testid="stSidebar"] {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }
}
</style>
""")
 
 
# =====================================================
# UI Components
# =====================================================
 
def show_header():
    H("""
    <div class="main-header">
        <div class="header-logo">🕋</div>
        <div style="flex:1;">
            <div class="header-title">نظام التنبؤ بمستويات ازدحام المعتمرين</div>
            <div class="header-subtitle">المسجد الحرام الشريف</div>
            <div class="header-line"></div>
        </div>
    </div>
    """)
 
 
def show_prediction_summary_box(
    predicted_umrah,
    crowding_level,
    selected_day_label,
    reason_text,
    temp_text,
    start_day,
    end_day,
    hajj_value,
    tawaf_value,
    show_hajj_info
):
    hajj_chips_html = ""
 
    if show_hajj_info:
        hajj_text = format_influence_value(hajj_value)
        tawaf_text = format_influence_value(tawaf_value)
 
        hajj_chips_html = f"""
            <div class="chip gold">تأثير الحج: {esc(hajj_text)}</div>
            <div class="chip gold">تأثير طواف الإفاضة: {esc(tawaf_text)}</div>
        """
 
    summary_html = f"""
    <div class="summary-box">
 
        <div class="cards-grid">
            <div class="mini-card">
                <div>
                    <div class="mini-label">المعتمرون المتوقعون</div>
                    <div class="mini-value">{predicted_umrah:,}</div>
                </div>
                <div class="mini-icon">👥</div>
            </div>
 
            <div class="mini-card">
                <div>
                    <div class="mini-label">مستوى الازدحام</div>
                    <div class="mini-value">{esc(crowding_level)}</div>
                </div>
                <div class="mini-icon">📊</div>
            </div>
 
            <div class="mini-card">
                <div>
                    <div class="mini-label">اليوم المختار</div>
                    <div class="mini-value">{esc(selected_day_label)}</div>
                </div>
                <div class="mini-icon">📅</div>
            </div>
        </div>
 
        <div class="reason-area">
            <div class="reason-title">⭐ سبب التوصية</div>
            <div class="reason-line"></div>
 
            <div class="reason-text">
                {esc(reason_text)}
            </div>
 
            <div class="chips">
                <div class="chip green">مستوى الازدحام: {esc(crowding_level)}</div>
                <div class="chip">درجة الحرارة: {esc(temp_text)}</div>
                <div class="chip blue">المعتمرون المتوقعون: {esc(f"{predicted_umrah:,}")}</div>
                {hajj_chips_html}
            </div>
 
            <div class="summary-foot">
                المقارنة المعروضة: 7 أيام من يوم {esc(start_day)} إلى يوم {esc(end_day)}
            </div>
        </div>
 
    </div>
    """
 
    # Rendered directly in the responsive page DOM (no fixed-height iframe),
    # so long Arabic text and stacked cards on small screens are never cropped.
    H(summary_html)
 
 
def show_better_day_question():
    question_html = """
    <div class="question-card">
        <div class="question-title">هل تريد يومًا أفضل؟</div>
        <div class="question-text">
            يمكن عرض أقل يوم ازدحامًا ضمن الأيام السبعة القريبة عند الحاجة.
        </div>
    </div>
    """
 
    H(question_html)
 
 
def show_best_day_answer(best_day_name, best_day_date, best_prediction):
    best_html = f"""
    <div class="best-card">
        <div class="best-icon">✅</div>
        <div class="best-label">أفضل يوم موصى به خلال الأيام السبعة القادمة</div>
        <div class="best-subtitle">بناءً على أقل ازدحام متوقع ضمن فترة المقارنة</div>
 
        <div class="title-line"></div>
 
        <div class="best-day">{esc(best_day_name)}</div>
        <div class="best-date">{esc(best_day_date)}</div>
 
        <div class="best-note">
            هذا اليوم هو الأقل ازدحامًا ضمن الأيام السبعة القريبة.
            {"عدد المعتمرين المتوقعين تقريبًا " + f"{best_prediction:,}" + " معتمر." if best_prediction is not None else ""}
        </div>
    </div>
    """
 
    H(best_html)
 
 
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
    <div class="sidebar-card">
        <div class="sidebar-logo">🕋</div>
        <div class="sidebar-title">التنبؤ</div>
        <div class="sidebar-subtitle">بازدحام المعتمرين</div>
    </div>
    <div class="sidebar-divider"></div>
    """)
 
    if st.button("الرئيسية  🏠"):
        st.session_state.page = "home"
        st.rerun()
 
    if st.button("إدخال البيانات  ✎"):
        st.session_state.page = "input"
        st.rerun()
 
    if st.button("لوحة النتائج  ▮"):
        st.session_state.page = "dashboard"
        st.rerun()
 
 
# =====================================================
# Home Page
# =====================================================
 
def landing_page():
    show_header()
 
    H("""
    <div class="hero-box">
        <div class="hero-icon">🕋</div>
        <div class="hero-title">مرحبًا بك في نظام التنبؤ بازدحام المعتمرين</div>
        <div class="hero-subtitle">Umrah Visitors Smart Forecasting System</div>
        <div class="hero-text">
            يساعد هذا النظام على عرض توقعات الازدحام بصورة واضحة وهادئة،
            مع اقتراح يوم مناسب للزيارة بناءً على البيانات المتاحة.
        </div>
    </div>
    """)
 
    c1, c2, c3 = st.columns(3)
 
    cards = [
        ("📈", "تنبؤ ذكي", "عرض مبسط لاتجاه الازدحام المتوقع."),
        ("📅", "أفضل يوم", "اقتراح اليوم الأقل ازدحامًا ضمن الفترة القريبة."),
        ("✅", "قرار أوضح", "توصية مباشرة وسهلة القراءة."),
    ]
 
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            H(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-text">{desc}</div>
            </div>
            """)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        if st.button("ابدأ إدخال البيانات", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
 
 
# =====================================================
# Input Page
# =====================================================
 
def input_page():
    with st.form("prediction_form"):
        H("""
        <div class="input-title">نظام التنبؤ بمستويات ازدحام المعتمرين</div>
        <div class="input-subtitle">المسجد الحرام الشريف</div>
        <div class="input-line"></div>
        <div class="input-section-title">✎ إدخال بيانات</div>
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
 
        blocked = month == "ذو القعدة" and nationality == "غير سعودي" and 1 <= int(day) <= 15
 
        if blocked:
            st.error("لا يمكن عرض النتائج لهذا الاختيار؛ لأن الجنسية غير سعودي خلال الفترة من 1 إلى 15 ذو القعدة.")
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        _, btn_col, _ = st.columns([1, 1.15, 1])
 
        with btn_col:
            submitted = st.form_submit_button("عرض النتائج", use_container_width=True)
 
        if submitted:
            if not name.strip():
                st.warning("الرجاء إدخال الاسم الكامل.")
                return
 
            if blocked:
                return
 
            st.session_state.entered = True
            st.session_state.show_best_day = False
            st.session_state.page = "dashboard"
            st.session_state.selected_name = name.strip()
            st.session_state.selected_nationality = nationality
            st.session_state.selected_month = month
            st.session_state.selected_day = int(day)
            st.rerun()
 
 
# =====================================================
# Dashboard Page
# =====================================================
 
def dashboard_page():
    show_header()
 
    if not st.session_state.entered:
        st.warning("الرجاء إدخال البيانات أولًا.")
        if st.button("الذهاب إلى صفحة إدخال البيانات"):
            st.session_state.page = "input"
            st.rerun()
        return
 
    month = st.session_state.get("selected_month")
    day = st.session_state.get("selected_day")
    show_hajj_info = is_hajj_related_month(month)
 
    with st.spinner("جاري تحميل البيانات..."):
        package = get_model_package()
        test_df = get_test_df_from_package(package)
 
    filtered_df, start_day, end_day = get_7_day_comparison(test_df, month, day)
 
    if filtered_df.empty:
        st.error("لا توجد بيانات مطابقة للشهر والتاريخ المختار.")
        return
 
    selected_day_df = filtered_df[filtered_df["Hijri_Day_Num"] == int(day)].copy()
 
    if selected_day_df.empty:
        st.error("لا توجد بيانات للتاريخ المختار.")
        return
 
    selected_row = selected_day_df.iloc[0]
    weekday = selected_row.get("Weekday_AR", "غير محدد")
    decision_result = get_decision(selected_row, filtered_df, month)
 
    predicted_umrah = int(selected_row["Prediction"])
    crowding_level = decision_result["level"]
 
    temp_text = "غير متاحة"
    if "AvgTemp_C" in selected_row.index and not pd.isna(selected_row.get("AvgTemp_C")):
        temp_text = f"{float(selected_row.get('AvgTemp_C')):.1f}°C"
 
    hajj_value = selected_row.get("Hajj_Count_For_Recommendation", 0)
    tawaf_value = selected_row.get("Tawaf_Ifadah_Count_For_Recommendation", 0)
 
    best_alt = get_best_alternative(filtered_df, day)
 
    if best_alt is not None:
        best_day_name = str(best_alt["Weekday_AR"])
        best_day_date = str(best_alt["Hijri_Date"])
        best_prediction = int(best_alt["Prediction"])
    else:
        best_day_name = "غير متاح"
        best_day_date = ""
        best_prediction = None
 
    selected_day_label = f"{weekday} - {day}"
 
    show_prediction_summary_box(
        predicted_umrah=predicted_umrah,
        crowding_level=crowding_level,
        selected_day_label=selected_day_label,
        reason_text=decision_result["reason"],
        temp_text=temp_text,
        start_day=start_day,
        end_day=end_day,
        hajj_value=hajj_value,
        tawaf_value=tawaf_value,
        show_hajj_info=show_hajj_info
    )
 
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
 
    # =====================================================
    # Chart + Small Table Side by Side
    # =====================================================
 
    chart_df = filtered_df.sort_values(["Hijri_Day_Num", DATE_COL]).copy()
    chart_df["Day_Label"] = (
        chart_df["Weekday_AR"].astype(str) +
        "<br>" +
        chart_df["Hijri_Day_Num"].astype(int).astype(str)
    )
 
    best_day_hijri = chart_df.loc[chart_df["Prediction"].idxmin(), "Hijri_Date"]
    worst_day_hijri = chart_df.loc[chart_df["Prediction"].idxmax(), "Hijri_Date"]
 
    point_colors = []
 
    for _, r in chart_df.iterrows():
        if r["Hijri_Date"] == best_day_hijri:
            point_colors.append("#1E7A47")
        elif r["Hijri_Date"] == worst_day_hijri:
            point_colors.append("#C44334")
        else:
            point_colors.append("#C49A3A")
 
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=chart_df["Day_Label"],
        y=chart_df["Prediction"],
        mode="lines+markers+text",
        marker=dict(size=10, color=point_colors),
        line=dict(color="#C49A3A", width=3),
        text=chart_df["Prediction"].apply(lambda x: f"{int(x):,}"),
        textposition="top center",
        hovertemplate="اليوم %{x}<br>المعتمرون المتوقعون %{y:,.0f}<extra></extra>"
    ))
 
    fig.update_layout(
        height=350,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.18)",
        margin=dict(l=16, r=16, t=34, b=16),
        font=dict(family="Cairo", size=12, color="#064b3b"),
        xaxis=dict(
            showgrid=False,
            title="",
            tickfont=dict(size=12, color="#6d6042")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(150,120,60,.16)",
            tickformat=",",
            title="",
            tickfont=dict(size=11, color="#6d6042")
        )
    )
 
    left_col, right_col = st.columns([1.45, 0.85])
 
    with left_col:
        H("""
        <div class="chart-card">
            <div class="section-title">اتجاه الازدحام المتوقع</div>
            <div class="title-line"></div>
        </div>
        """)
 
        st.plotly_chart(fig, use_container_width=True)
 
    with right_col:
        display_table = prepare_display_table(chart_df, show_hajj_info)
 
        # Render the whole card as one markdown block so the .table-card div
        # actually wraps the table (separate H() calls get auto-closed by the
        # browser, leaving the card as an empty title bar). A scroll wrapper
        # lets the table scroll horizontally on narrow screens instead of
        # overflowing the card.
        H(f"""
        <div class="table-card">
            <div class="section-title">ملخص الأيام السبعة</div>
            <div class="title-line"></div>
            <div class="table-scroll">
                {df_to_html_table(display_table)}
            </div>
        </div>
        """)
 
    # =====================================================
    # Better Day Question
    # =====================================================
 
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
 
    show_better_day_question()
 
    _, btn_col, _ = st.columns([1, 1.1, 1])
 
    with btn_col:
        if st.button("عرض أفضل يوم مقترح", use_container_width=True):
            st.session_state.show_best_day = True
            st.rerun()
 
    if st.session_state.show_best_day:
        show_best_day_answer(
            best_day_name=best_day_name,
            best_day_date=best_day_date,
            best_prediction=best_prediction
        )
 
    # =====================================================
    # Bottom Buttons Only
    # =====================================================
 
    b1, b2 = st.columns(2)
 
    with b1:
        if st.button("رجوع لإدخال بيانات جديدة", use_container_width=True):
            st.session_state.page = "input"
            st.session_state.show_best_day = False
            st.rerun()
 
    with b2:
        report_df = pd.DataFrame([{
            "Hijri_Month": month,
            "Hijri_Date": day,
            "Weekday": weekday,
            "Predicted_Umrah_Visitors": predicted_umrah,
            "Crowding_Level": crowding_level,
            "Decision": decision_result["decision"],
            "Best_Day": best_day_name,
            "Best_Date": best_day_date,
        }])
 
        csv = report_df.to_csv(index=False).encode("utf-8-sig")
 
        st.download_button(
            label="تحميل التقرير CSV",
            data=csv,
            file_name="umrah_prediction_result.csv",
            mime="text/csv",
            use_container_width=True
        )
 
 
# =====================================================
# Router
# =====================================================
 
if st.session_state.page == "home":
    landing_page()
elif st.session_state.page == "input":
    input_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
