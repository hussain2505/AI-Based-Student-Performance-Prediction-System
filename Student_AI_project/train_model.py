"""
train_model.py
----------------
Trains a Bagging ensemble model (BaggingRegressor wrapping Decision
Trees) to predict a student's final_score from the features in
student_data.csv, then saves the trained model + evaluation metrics
to student_model.pkl.

Run:
    python create_dataset.py   # first, if student_data.csv doesn't exist
    python train_model.py
"""

import pickle

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.preprocessing import load_data, split_features_target, FEATURE_COLUMNS

MODEL_PATH = "student_model.pkl"


def train():
    df = load_data("student_data.csv")
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Bagging ensemble: many Decision Trees trained on bootstrap
    #      samples, predictions averaged to reduce variance/overfitting.
    base_estimator = DecisionTreeRegressor(max_depth=6, random_state=42)
    model = BaggingRegressor(
        estimator=base_estimator,
        n_estimators=100,
        max_samples=0.8,
        max_features=0.9,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # ---- Evaluate ----------------------------------------------------
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "n_estimators": model.n_estimators}

    print("Bagging Regressor trained.")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2  : {r2:.3f}")

    # ---- Feature importance (approximated via averaging each base
    #      estimator's feature_importances_) -----------------------
    importances = np.mean(
        [tree.feature_importances_ for tree in model.estimators_], axis=0
    )
    feature_importance = dict(zip(FEATURE_COLUMNS, importances))

    payload = {
        "model": model,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "feature_columns": FEATURE_COLUMNS,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved trained model + metrics to {MODEL_PATH}")


if __name__ == "__main__":
    train()
