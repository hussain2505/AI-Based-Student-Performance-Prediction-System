import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Analysis", page_icon="🔍", layout="wide")
st.title("🔍 Data Analysis")

DATA_PATH = "student_data.csv"

if not os.path.exists(DATA_PATH):
    st.warning("student_data.csv not found. Run `python create_dataset.py` first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

numeric_cols = [
    "study_hours", "attendance", "previous_score",
    "sleep_hours", "final_score",
]

st.subheader("Correlation with final score")
corr = df[numeric_cols].corr()["final_score"].drop("final_score").sort_values()
st.bar_chart(corr)

st.divider()

st.subheader("Explore a feature vs. final score")
feature = st.selectbox(
    "Choose a numeric feature",
    [c for c in numeric_cols if c != "final_score"],
)
st.scatter_chart(df, x=feature, y="final_score")

st.divider()

st.subheader("Score by category")
cat_col = st.selectbox(
    "Choose a categorical feature",
    ["study_group", "parental_support", "extracurricular", "internet_access"],
)
grouped = df.groupby(cat_col)["final_score"].mean().sort_values()
st.bar_chart(grouped)

st.divider()
st.subheader("Summary statistics")
st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
