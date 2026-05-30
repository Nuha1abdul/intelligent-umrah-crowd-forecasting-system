Intelligent Umrah Crowd Forecasting System
An interactive Data Science graduation project for predicting Umrah visitor crowd levels and recommending suitable visiting days at Al-Masjid Al-Haram.

The project combines a forecasting model, prepared Hijri-year datasets, model results, and a deployed Streamlit dashboard.

Live Dashboard
https://intelligent-umrah-crowd-forecasting-system-bbm9vprvwhhezu7dr5s.streamlit.app

Project Overview
This system helps users explore expected crowd levels for Umrah visits by selecting a Hijri month and day. The dashboard displays:

Predicted number of Umrah visitors
Crowd level classification
Recommendation on whether the selected day is suitable
Alternative lower-crowd day within the nearby 7-day window
Interactive charts and result tables
Supporting information such as Hijri date, weekday, temperature, and Hajj-related indicators where available
The deployed dashboard is built on top of a trained Growth-Rate XGBoost forecasting model.

Project Layers
1. Modeling Layer
The modeling layer includes:

Data collection and organization
Data preprocessing and cleaning
Hijri calendar alignment
Weather feature integration
Feature engineering
Lag-based and rolling-window features
Growth-rate target construction
XGBoost model training
Testing on Hijri year 1446
Forecast result generation
Final modeling notebook:

notebooks/final_model_notebook.ipynb

Model training script:

train_model.py

2. Dashboard / Deployment Layer
The dashboard layer is implemented using Streamlit and provides the user-facing interface for the project.

Dashboard file:

app.py

The app loads the trained model artifact:

growth_rate_xgboost_model.pkl

Repository Structure
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
│ └── final_model_notebook.ipynb
│
├── data/
│ ├── raw/
│ ├── external/
│ └── processed/
│
├── reports/
├── poster/
└── presentation/

Data Organization
Folder	Purpose
data/raw/	Original source datasets used during preprocessing and feature engineering.
data/external/	Supporting external files, including Hijri calendar files for 1445H and 1446H.
data/processed/	Final processed dataset used for modeling and dashboard deployment.
The processed dataset is also kept in the repository root because the deployed Streamlit app and training script currently read it from that location.

Model
The final deployed model is a Growth-Rate XGBoost model.

The model predicts a 7-day growth rate:

Predicted_Growth_Rate_7

The final visitor prediction is calculated using the previous 7-day lag value:

Prediction = lag_7 * (1 + Predicted_Growth_Rate_7)

Trained model artifact:

growth_rate_xgboost_model.pkl

Model results file:

growth_rate_xgboost_results_1446.xlsx

Dashboard Features
The Streamlit dashboard provides:

Arabic user interface
Hijri month and day selection
Visitor crowd prediction display
Crowd level indicator
Visiting recommendation
Best alternative day suggestion
7-day forecast visualization
Detailed forecast table
Seasonal Hajj-related information where applicable
How to Run Locally
Clone the repository:
git clone https://github.com/Nuha1abdul/intelligent-umrah-crowd-forecasting-system.git

cd intelligent-umrah-crowd-forecasting-system

Install dependencies:
pip install -r requirements.txt

Run the Streamlit dashboard:
streamlit run app.py

Re-training the Model
The model can be regenerated using:

python train_model.py

The training script uses the processed Excel dataset in the repository root:

Intelligent System for Predicting Visitor Crowd Levels and Optimal Visiting Times at Al-Masjid Al-Haram.xlsx

It trains on Hijri year 1445 and tests on Hijri year 1446.

Requirements
The required Python packages are listed in:

requirements.txt

Main libraries:

Streamlit
pandas
NumPy
Plotly
XGBoost
scikit-learn
openpyxl
joblib
The Streamlit deployment uses:

runtime.txt

with Python 3.11.

Deployment
The dashboard is deployed through Streamlit Community Cloud using this GitHub repository.

Deployment files used by Streamlit:

app.py
requirements.txt
runtime.txt
growth_rate_xgboost_model.pkl
growth_rate_xgboost_results_1446.xlsx
Processed dataset file
GP2 Submission Notes
This repository is prepared to satisfy the GP2 code repository requirement:

Final source code files
Final notebook
Dataset files
Trained model artifact
Forecasting results
Streamlit dashboard
Requirements file
Deployment link
The following folders are included for final deliverables once completed:

reports/
poster/
presentation/

These folders will contain the final report, poster, and presentation slides when they are finalized.

Current Repository Status
Completed:

Dashboard source code uploaded
Training script uploaded
Final notebook uploaded
Raw datasets uploaded
External Hijri calendar files uploaded
Processed dataset uploaded
Trained model uploaded
Forecast results uploaded
Streamlit dashboard deployed
Pending final GP2 materials:

Final report PDF
Project poster
Presentation slides
Authors
Data Science Graduation Project 2
Umm Al-Qura University
College of Computing
Data Science Department


انزلي تحت.
في commit message اكتبي:
Update README for final GP2 submission
اضغطي:
Commit changes
