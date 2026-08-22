import streamlit as st
from database.database import init_db

st.set_page_config(
    page_title="Student AI System",
    page_icon="🎓",
    layout="wide",
)

# Make sure the predictions table exists before any page needs it
init_db()

st.title("🎓 AI-Based Student Performance Prediction System")
st.caption("Powered by a Bagging ensemble (BaggingRegressor + Decision Trees)")

st.success("Streamlit is working successfully!")

st.write(
    "Welcome to the Student AI Project. Use the sidebar to navigate between "
    "**Dashboard**, **Analysis**, **Prediction**, **History**, and **Reports**."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model type", "Bagging Regressor")
with col2:
    st.metric("Base estimator", "Decision Tree")
with col3:
    st.metric("Pages", "5")

st.divider()
st.markdown(
    """
    ### Getting started
    1. **Generate data** — run `python create_dataset.py` once to create `student_data.csv`.
    2. **Train the model** — run `python train_model.py` to train the Bagging ensemble
       and save `student_model.pkl`.
    3. **Launch the app** — run `streamlit run app.py` and use the sidebar pages:
       - **Dashboard** — quick overview of the dataset and model performance
       - **Analysis** — explore feature relationships and distributions
       - **Prediction** — enter a student's details and get a predicted score
       - **History** — see every prediction made in this session/database
       - **Reports** — download a summary report of predictions
    """
)
