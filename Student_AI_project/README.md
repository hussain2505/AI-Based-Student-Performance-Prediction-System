# Student AI Project — Bagging + Streamlit

A normal (single-model) AI project that predicts student performance
using a **Bagging ensemble** (`BaggingRegressor` wrapping `DecisionTreeRegressor`
from scikit-learn), wired up to a multi-page Streamlit app.

## Project structure

```
Student_AI_Project/
├── app.py                    # Main entry page
├── create_dataset.py         # Generates synthetic student_data.csv
├── train_model.py            # Trains the Bagging model -> student_model.pkl
├── requirements.txt
├── utils/
│   └── preprocessing.py      # Shared encoding/scaling logic
├── database/
│   └── database.py           # SQLite storage for prediction history
└── pages/
    ├── 1_Dashboard.py         # Dataset + model performance overview
    ├── 2_Analysis.py          # Feature exploration / correlations
    ├── 3_Prediction.py        # Enter a student, get a prediction
    ├── 4_History.py           # View all past predictions
    └── 5_Reports.py           # Download a CSV report
```

## Setup

```bash
pip install -r requirements.txt

# 1. Generate the dataset
python create_dataset.py

# 2. Train the Bagging model
python train_model.py

# 3. Launch the app
streamlit run app.py
```

Streamlit will automatically pick up everything in `pages/` and list
them in the sidebar in numeric order (Dashboard, Analysis, Prediction,
History, Reports).

## Why Bagging?

Bagging (Bootstrap Aggregating) trains many Decision Trees, each on a
random bootstrap sample of the data (and a random subset of features),
then averages their predictions. This reduces variance and overfitting
compared to a single Decision Tree, without needing heavy tuning —
a good fit for a small tabular dataset like this one.

## Notes

- The dataset in `create_dataset.py` is **synthetic** — swap in your
  real student data (same column names) to use real records.
- Every prediction made on the Prediction page is saved to
  `database/predictions.db` (SQLite) and shown on the History page.
- The pass/fail threshold (default 50) can be changed in
  `utils/preprocessing.py` → `score_to_pass_fail()`.
