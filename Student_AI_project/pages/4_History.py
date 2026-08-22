import streamlit as st

from database.database import fetch_all_predictions, clear_history

st.set_page_config(page_title="History", page_icon="🕘", layout="wide")
st.title("🕘 Prediction History")

df = fetch_all_predictions()

if df.empty:
    st.info("No predictions yet. Make one on the Prediction page.")
else:
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Total predictions", len(df))
    col2.metric("Pass rate", f"{(df['result'] == 'Pass').mean() * 100:.1f}%")

    if st.button("Clear history", type="secondary"):
        clear_history()
        st.rerun()
