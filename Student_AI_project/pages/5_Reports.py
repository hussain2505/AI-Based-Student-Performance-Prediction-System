import streamlit as st

from database.database import fetch_all_predictions

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")
st.title("📄 Reports")

df = fetch_all_predictions()

if df.empty:
    st.info("No predictions yet. Make one on the Prediction page first.")
    st.stop()

st.write("Summary of all predictions made so far:")

c1, c2, c3 = st.columns(3)
c1.metric("Total predictions", len(df))
c2.metric("Average predicted score", f"{df['predicted_score'].mean():.1f}")
c3.metric("Pass rate", f"{(df['result'] == 'Pass').mean() * 100:.1f}%")

st.subheader("Grade distribution")
st.bar_chart(df["grade"].value_counts())

st.divider()
st.subheader("Full prediction log")
st.dataframe(df, use_container_width=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download report as CSV",
    data=csv_bytes,
    file_name="student_predictions_report.csv",
    mime="text/csv",
)
