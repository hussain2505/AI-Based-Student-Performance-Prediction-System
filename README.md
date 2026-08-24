# AI-Based-Student-Performance-Prediction-System
Why Bagging?
Bagging (Bootstrap Aggregating) trains many Decision Trees, each on a random bootstrap sample of the data (and a random subset of features), then averages their predictions. This reduces variance and overfitting compared to a single Decision Tree, without needing heavy tuning — a good fit for a small tabular dataset like this one.

Notes
The dataset in create_dataset.py is synthetic — swap in your real student data (same column names) to use real records.
Every prediction made on the Prediction page is saved to database/predictions.db (SQLite) and shown on the History page.
The pass/fail threshold (default 50) can be changed in utils/preprocessing.py → score_to_pass_fail().
