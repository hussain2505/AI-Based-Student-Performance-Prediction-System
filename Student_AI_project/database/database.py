"""
database/database.py
----------------------
Tiny SQLite wrapper that stores every prediction made through the
Streamlit UI so the History and Reports pages have something to show.
"""

import sqlite3
from datetime import datetime

DB_PATH = "database/predictions.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            student_name TEXT,
            study_hours REAL,
            attendance REAL,
            previous_score REAL,
            sleep_hours REAL,
            parental_support INTEGER,
            extracurricular INTEGER,
            internet_access INTEGER,
            study_group TEXT,
            predicted_score REAL,
            grade TEXT,
            result TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(record: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (
            timestamp, student_name, study_hours, attendance, previous_score,
            sleep_hours, parental_support, extracurricular, internet_access,
            study_group, predicted_score, grade, result
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            record.get("student_name", "Unnamed"),
            record["study_hours"],
            record["attendance"],
            record["previous_score"],
            record["sleep_hours"],
            record["parental_support"],
            record["extracurricular"],
            record["internet_access"],
            record["study_group"],
            record["predicted_score"],
            record["grade"],
            record["result"],
        ),
    )
    conn.commit()
    conn.close()


def fetch_all_predictions():
    import pandas as pd

    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df


def clear_history():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
