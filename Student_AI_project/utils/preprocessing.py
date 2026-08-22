"""
utils/preprocessing.py
-----------------------
Shared preprocessing helpers used by both train_model.py and the
Streamlit app, so training and inference always match.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "study_hours",
    "attendance",
    "previous_score",
    "sleep_hours",
    "parental_support",
    "extracurricular",
    "internet_access",
    "study_group_A",
    "study_group_B",
    "study_group_C",
]

TARGET_COLUMN = "final_score"


def load_data(path="student_data.csv"):
    """Load the raw dataset from CSV."""
    return pd.read_csv(path)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the study_group column and ensure all expected
    feature columns exist (fills missing dummy columns with 0).
    Works on a single row or the full dataset.
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=["study_group"], prefix="study_group")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df


def split_features_target(df: pd.DataFrame):
    """Return (X, y) ready for model training."""
    df_encoded = encode_features(df)
    X = df_encoded[FEATURE_COLUMNS]
    y = df_encoded[TARGET_COLUMN]
    return X, y


def prepare_single_input(input_dict: dict) -> pd.DataFrame:
    """
    Turn a single form submission (dict) from the Streamlit UI into a
    one-row DataFrame with the exact same columns/order the model
    was trained on.
    """
    df = pd.DataFrame([input_dict])
    df_encoded = encode_features(df)
    return df_encoded[FEATURE_COLUMNS]


def score_to_grade(score: float) -> str:
    """Convert a numeric score into a letter grade for display."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def score_to_pass_fail(score: float, threshold: float = 50.0) -> str:
    return "Pass" if score >= threshold else "Fail"
