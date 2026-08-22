import pickle
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

DATA_PATH = "student_data.csv"
MODEL_PATH = "student_model.pkl"

if not os.path.exists(DATA_PATH):
    st.warning("student_data.csv not found. Run `python create_dataset.py` first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# ---- Top-level metrics --------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total students", len(df))
c2.metric("Avg. final score", f"{df['final_score'].mean():.1f}")
c3.metric("Avg. study hours", f"{df['study_hours'].mean():.1f} hrs")
c4.metric("Avg. attendance", f"{df['attendance'].mean():.1f}%")

st.divider()

# ---- Model performance, if trained -------------------------------------
st.subheader("Model performance (Bagging Regressor)")
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    metrics = payload["metrics"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['mae']:.2f}")
    m2.metric("RMSE", f"{metrics['rmse']:.2f}")
    m3.metric("R² score", f"{metrics['r2']:.3f}")
    m4.metric("Estimators (trees)", metrics["n_estimators"])

    st.subheader("Feature importance (averaged across bagged trees)")
    fi = pd.Series(payload["feature_importance"]).sort_values(ascending=True)
    st.bar_chart(fi)
else:
    st.info("No trained model found yet. Run `python train_model.py` to train one.")

st.divider()
st.subheader("Score distribution")
st.bar_chart(df["final_score"].value_counts(bins=10).sort_index())

st.subheader("Raw data preview")
st.dataframe(df.head(20), use_container_width=True)
