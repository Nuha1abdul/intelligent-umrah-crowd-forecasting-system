# Intelligent Umrah Crowd Forecasting System

A Data Science graduation project for predicting Umrah visitor crowd levels and recommending suitable visiting days at Al-Masjid Al-Haram, built using a Growth-Rate XGBoost forecasting model and an interactive Streamlit dashboard.

---

## Live Dashboard

🔗 [Open Dashboard](https://intelligent-umrah-crowd-forecasting-system-bbm9vprvwhhezu7dr5s.streamlit.app)

---

## Project Overview

This system enables users to explore expected crowd levels for Umrah visits by selecting a Hijri month and day. The model was trained on historical visitor data for Hijri year 1445 and evaluated on Hijri year 1446.

The system outputs:

- Predicted number of Umrah visitors
- Crowd level classification
- Visiting day recommendation
- Best alternative lower-crowd day within a 7-day window
- Interactive forecast charts and detailed result tables
- Supporting context including Hijri date, weekday, temperature, and Hajj-related indicators

---

## Project Layers

### 1. Modeling Layer

The modeling pipeline covers:

- Data collection, preprocessing, and cleaning
- Hijri calendar alignment
- Weather feature integration
- Feature engineering including lag-based and rolling-window features
- Growth-rate target construction (`Predicted_Growth_Rate_7`)
- XGBoost model training on Hijri year 1445
- Model evaluation and testing on Hijri year 1446
- Forecast result generation

**Final notebook:** `notebooks/final_model_notebook.ipynb`  
**Training script:** `train_model.py`

### 2. Dashboard / Deployment Layer

The Streamlit dashboard provides the full user-facing interface for the system.

**Dashboard file:** `app.py`  
**Loaded model artifact:** `growth_rate_xgboost_model.pkl`

---

## Repository Structure

```
intelligent-umrah-crowd-forecasting-system/
├── app.py
├── train_model.py
├── README.md
├── requirements.txt
├── runtime.txt
├── growth_rate_xgboost_model.pkl
├── growth_rate_xgboost_results_1446.xlsx
├── Intelligent System for Predicting Visitor Crowd Levels and Optimal Visiting Times at Al-Masjid Al-Haram.xlsx
│
├── notebooks/
│   └── final_model_notebook.ipynb
│
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
│
├── reports/
├── poster/
└── presentation/
```

---

## Data Organization

| Folder | Contents |
|---|---|
| `data/raw/` | Original source datasets used during preprocessing and feature engineering |
| `data/external/` | Supporting external files including Hijri calendar data for 1445H and 1446H |
| `data/processed/` | Final processed dataset used for model training and dashboard deployment |

> **Note:** The processed dataset is also stored in the repository root because the Streamlit app and training script currently read it from that location.

---

## Model

**Model type:** Growth-Rate XGBoost

The model predicts a 7-day growth rate target:

```
Predicted_Growth_Rate_7
```

The final visitor count is derived using the previous 7-day lag value:

```
Prediction = lag_7 × (1 + Predicted_Growth_Rate_7)
```

| File | Description |
|---|---|
| `growth_rate_xgboost_model.pkl` | Trained model artifact loaded by the dashboard |
| `growth_rate_xgboost_results_1446.xlsx` | Forecast results for Hijri year 1446 |

---

## Dashboard Features

The Streamlit dashboard provides:

- Arabic user interface
- Hijri month and day selection inputs
- Predicted visitor count display
- Crowd level indicator (low / moderate / high)
- Visiting day recommendation
- Best alternative day suggestion within a 7-day window
- 7-day forecast visualization
- Detailed forecast result table
- Seasonal Hajj-related context where applicable

---

## How to Run Locally

**1. Clone the repository:**

```bash
git clone https://github.com/Nuha1abdul/intelligent-umrah-crowd-forecasting-system.git
cd intelligent-umrah-crowd-forecasting-system
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Launch the dashboard:**

```bash
streamlit run app.py
```

---

## Re-training the Model

To regenerate the trained model artifact, run:

```bash
python train_model.py
```

The script reads the processed dataset from the repository root:

```
Intelligent System for Predicting Visitor Crowd Levels and Optimal Visiting Times at Al-Masjid Al-Haram.xlsx
```

It trains on Hijri year 1445 and evaluates on Hijri year 1446.

---

## Requirements

All required packages are listed in `requirements.txt`. Key libraries include:

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `xgboost`
- `scikit-learn`
- `openpyxl`
- `joblib`

The Streamlit deployment targets **Python 3.11**, as specified in `runtime.txt`.

---

## Deployment

The dashboard is deployed via **Streamlit Community Cloud** directly from this repository.

Files used during deployment:

- `app.py` — dashboard application
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python version specification
- `growth_rate_xgboost_model.pkl` — trained model artifact
- `growth_rate_xgboost_results_1446.xlsx` — precomputed forecast results
- Processed dataset file (repository root)

---

## GP2 Submission Status

### Completed

- [x] Dashboard source code (`app.py`)
- [x] Model training script (`train_model.py`)
- [x] Final modeling notebook (`notebooks/final_model_notebook.ipynb`)
- [x] Raw datasets (`data/raw/`)
- [x] External Hijri calendar files (`data/external/`)
- [x] Processed dataset (`data/processed/` and repository root)
- [x] Trained model artifact (`growth_rate_xgboost_model.pkl`)
- [x] Forecast results (`growth_rate_xgboost_results_1446.xlsx`)
- [x] Requirements file (`requirements.txt`)
- [x] Streamlit dashboard deployed and live

### Pending

- [ ] Final project report → `reports/`
- [ ] Project poster → `poster/`
- [ ] Presentation slides → `presentation/`

> The `reports/`, `poster/`, and `presentation/` folders are placeholder directories. They will be populated once the final GP2 deliverables are completed.

---

## Authors

Data Science Graduation Project 2 (GP2)  
Umm Al-Qura University  
College of Computing — Data Science Department
