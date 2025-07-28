import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Pavement Performance Prediction Tool")

uploaded_file = st.file_uploader("Upload Excel File (optional)", type=["xlsx"])

if uploaded_file:
    data_source = uploaded_file
    st.success("Using uploaded file.")
else:
    data_source = "Result Summary.xlsx"
    st.info("No file uploaded. Using default file: Result Summary.xlsx")

@st.cache_resource
def train_ct_model(file):
    df = pd.read_excel(file, sheet_name='CT')

    features = [
        'Additives', 'Design Gyration', 'RAP', 'BSG (field)', 'Air Voids (field)',
        'VMA (field)', 'Dust/Binder (Field)', 'Ignition Oven AC (%) (Field)', 'Slip AC Content (%) (Field)'
    ]
    target = 'Avg. CTindex'

    existing_features = [f for f in features if f in df.columns]
    df_model = df[[target] + existing_features].dropna()
    X = df_model[existing_features]
    y = df_model[target]

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    return model, existing_features, df_model, X, y

@st.cache_resource
def train_rut_model(file):
    df = pd.read_excel(file, sheet_name='Rut depth')

    features = [
        'Additives',
        'Design Gyration',
        'RAP',
        'BSG (field)',
        'Air Voids (field)',
        "VMA (field)",
        "Dust/Binder (Field)",
        'Ignition Oven AC (%) (Field)',
        'Slip AC Content (%) (Field)',
        'Avg. Displacement @ 75% peak load (mm)',
        'Avg. Post-Peak Slope @75% peak load (kN/mm)',
        'Avg. Failure Energy (J/m^2)',
        'Avg. CTindex',
        'CTindex COV(%)'
    ]
    target = 'Avg. Rut Depth'

    existing_features = [f for f in features if f in df.columns]
    df_model = df[[target] + existing_features].dropna()
    X = df_model[existing_features]
    y = df_model[target]

    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    return model, existing_features, df_model, X, y

try:
    # --- CT-Index
    ct_model, ct_features, ct_df, X_ct, y_ct = train_ct_model(data_source)
    y_ct_pred = ct_model.predict(X_ct)
    ct_result_df = ct_df.copy()
    ct_result_df["Predicted CT-Index"] = y_ct_pred

    st.subheader("CT-Index Prediction")
    st.dataframe(ct_result_df[[*ct_features, "Avg. CTindex", "Predicted CT-Index"]])

    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_ct, y=y_ct_pred, ax=ax1)
    ax1.plot([y_ct.min(), y_ct.max()], [y_ct.min(), y_ct.max()], 'r--')
    ax1.set_xlabel("True Avg. CTindex")
    ax1.set_ylabel("Predicted CT-Index")
    ax1.set_title("CT-Index: Predicted vs Actual")
    st.pyplot(fig1)

    # --- Rut Depth
    rut_model, rut_features, rut_df, X_rut, y_rut = train_rut_model(data_source)
    y_rut_pred = rut_model.predict(X_rut)
    rut_result_df = rut_df.copy()
    rut_result_df["Predicted Rut Depth"] = y_rut_pred

    st.subheader("Rut Depth Prediction")
    st.dataframe(rut_result_df[[*rut_features, "Avg. Rut Depth", "Predicted Rut Depth"]])

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=y_rut, y=y_rut_pred, ax=ax2)
    ax2.plot([y_rut.min(), y_rut.max()], [y_rut.min(), y_rut.max()], 'r--')
    ax2.set_xlabel("True Avg. Rut Depth")
    ax2.set_ylabel("Predicted Rut Depth")
    ax2.set_title("Rut Depth: Predicted vs Actual")
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Error: {e}")
