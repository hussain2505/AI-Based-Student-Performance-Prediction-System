"""
create_dataset.py
------------------
Generates a synthetic student performance dataset and saves it as
student_data.csv. Run this once before training the model:

    python create_dataset.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 600  # number of students

# ---- Feature generation -----------------------------------------------
study_hours = np.round(np.random.normal(5, 2, N).clip(0, 12), 1)
attendance = np.round(np.random.normal(80, 12, N).clip(40, 100), 1)
previous_score = np.round(np.random.normal(65, 15, N).clip(0, 100), 1)
sleep_hours = np.round(np.random.normal(7, 1.3, N).clip(3, 10), 1)
parental_support = np.random.choice([0, 1], N, p=[0.35, 0.65])   # 0=Low,1=High
extracurricular = np.random.choice([0, 1], N, p=[0.5, 0.5])       # 0=No,1=Yes
internet_access = np.random.choice([0, 1], N, p=[0.2, 0.8])       # 0=No,1=Yes
study_group = np.random.choice(["A", "B", "C"], N)

# ---- Target: final_score, built from a noisy weighted formula ---------
noise = np.random.normal(0, 6, N)
final_score = (
    0.35 * previous_score
    + 4.2 * study_hours
    + 0.25 * attendance
    + 3.0 * parental_support
    + 1.5 * extracurricular
    + 1.0 * internet_access
    + 1.2 * sleep_hours
    + noise
)
final_score = np.round(final_score.clip(0, 100), 1)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "parental_support": parental_support,
    "extracurricular": extracurricular,
    "internet_access": internet_access,
    "study_group": study_group,
    "final_score": final_score,
})

df.to_csv("student_data.csv", index=False)
print(f"student_data.csv created with {len(df)} rows.")
print(df.head())
