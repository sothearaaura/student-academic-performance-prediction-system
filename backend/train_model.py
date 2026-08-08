"""
train_model.py
==============
Reproduces the data cleaning, feature engineering, model training and
model-selection logic of `notebooks/Student_Prediction_v3_corrected.ipynb`
and exports the artifacts the Flask website loads at runtime:

    trained_models/regression_pipeline.pkl
    trained_models/classification_pipeline.pkl
    trained_models/metrics.json
    trained_models/feature_names.json

Run this ONCE (or whenever the dataset changes) -- the Flask app never
retrains, it only loads these files. See app/services/ml_service.py.

Usage:
    python train_model.py
    python train_model.py --pass-threshold 65        # override the Pass/Fail cut-off
    python train_model.py --dataset /path/to/new.xlsx  # train on a different file

The admin "Retrain Model" button (app/routes/admin.py) calls this script as
a subprocess with --pass-threshold taken from the AppSetting row, so the
value set on the Settings page actually changes what the classifier learns.
"""
import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from app.ml.pipeline import (  # noqa: E402
    FEATURE_COLS,
    GENDER_MAP,
    PASS_THRESHOLD as DEFAULT_PASS_THRESHOLD,
    SUPPORT_MAP,
    StudentPerformancePipeline,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Student_Performance_Cleaned.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def log(msg):
    print(f"[train_model] {msg}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors notebook Step 3 (Data Cleaning) exactly."""
    df_clean = df.copy()

    df_clean = df_clean[df_clean["FinalGrade"].notna()]

    if "StudentID" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["StudentID"], keep="first")
    df_clean = df_clean.drop(columns=["StudentID", "Name"], errors="ignore")

    if "Attendance (%)" in df_clean.columns:
        impossible = (df_clean.get("Attendance (%)", pd.Series(dtype=float)) > 100) | (
            df_clean.get("Study Hours", pd.Series(dtype=float)) < 0
        )
        df_clean = df_clean[~impossible].reset_index(drop=True)

    df_clean["Gender"] = df_clean["Gender"].astype(str).str.strip().str.title()
    df_clean["ParentalSupport"] = df_clean["ParentalSupport"].astype(str).str.strip().str.title()

    # Unify duplicate columns (AttendanceRate/Attendance(%), StudyHoursPerWeek/Study Hours)
    if "Attendance_Final" not in df_clean.columns:
        df_clean["Attendance_Final"] = df_clean["AttendanceRate"].where(
            df_clean["AttendanceRate"].notna(), df_clean.get("Attendance (%)")
        )
    if "StudyHours_Final" not in df_clean.columns:
        df_clean["StudyHours_Final"] = df_clean["StudyHoursPerWeek"].where(
            df_clean["StudyHoursPerWeek"].notna(), df_clean.get("Study Hours")
        )
    df_clean = df_clean.drop(
        columns=["AttendanceRate", "Attendance (%)", "StudyHoursPerWeek", "Study Hours"],
        errors="ignore",
    )

    if "Online_Classes" not in df_clean.columns:
        df_clean["Online_Classes"] = (
            df_clean["Online Classes Taken"].map({True: 1, False: 0}).fillna(0).astype(int)
        )
    df_clean = df_clean.drop(columns=["Online Classes Taken"], errors="ignore")

    return df_clean


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_feat = df_clean.copy()
    df_feat["Attendance_Study_Interaction"] = df_feat["Attendance_Final"] * df_feat["StudyHours_Final"]
    df_feat["Grade_Study_Ratio"] = df_feat["PreviousGrade"] / (df_feat["StudyHours_Final"] + 1)
    return df_feat


def main(dataset_path=None, pass_threshold=None):
    dataset_path = dataset_path or DATASET_PATH
    pass_threshold = DEFAULT_PASS_THRESHOLD if pass_threshold is None else float(pass_threshold)
    log(f"Using pass threshold: {pass_threshold}")
    log(f"Loading dataset from {dataset_path}")
    df = pd.read_excel(dataset_path)
    log(f"Raw shape: {df.shape}")

    df_clean = clean_data(df)
    log(f"Clean shape: {df_clean.shape}")

    df_feat = engineer_features(df_clean)

    X = df_feat[FEATURE_COLS]
    y = df_feat["FinalGrade"]

    grade_bins = pd.qcut(y, q=4, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=grade_bins
    )
    X_train, X_test = X_train.copy(), X_test.copy()

    gender_mode = X_train["Gender"].mode()[0]
    support_mode = X_train["ParentalSupport"].mode()[0]
    for col, mode_val in (("Gender", gender_mode), ("ParentalSupport", support_mode)):
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    X_train["Gender"] = X_train["Gender"].map(GENDER_MAP)
    X_test["Gender"] = X_test["Gender"].map(GENDER_MAP)
    X_train["ParentalSupport"] = X_train["ParentalSupport"].map(SUPPORT_MAP)
    X_test["ParentalSupport"] = X_test["ParentalSupport"].map(SUPPORT_MAP)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    log("Preprocessing complete. Training regression models...")

    # ---------------- Regression ----------------
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge (a=1)": Ridge(alpha=1.0),
        "Lasso (a=0.1)": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=10, max_features="sqrt", random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
        ),
    }
    try:
        from xgboost import XGBRegressor

        reg_models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, random_state=42, verbosity=0,
        )
    except ImportError:
        log("xgboost not installed - skipping XGBoost")

    reg_results, trained_reg = [], {}
    for name, model in reg_models.items():
        model.fit(X_train_scaled, y_train)
        y_tr = model.predict(X_train_scaled)
        y_te = model.predict(X_test_scaled)
        trained_reg[name] = model
        train_r2, test_r2 = r2_score(y_train, y_tr), r2_score(y_test, y_te)
        reg_results.append(
            {
                "Model": name,
                "Train R2": round(train_r2, 4),
                "Test R2": round(test_r2, 4),
                "Test RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_te))), 4),
                "Test MAE": round(mean_absolute_error(y_test, y_te), 4),
                "Overfit Gap": round(train_r2 - test_r2, 4),
            }
        )
        log(f"  {name:20s} Test R2={test_r2:.4f}")

    reg_df = pd.DataFrame(reg_results).sort_values("Test R2", ascending=False).reset_index(drop=True)
    best_reg_name = reg_df.iloc[0]["Model"]
    best_reg = trained_reg[best_reg_name]
    log(f"Best regression model: {best_reg_name} (Test R2={reg_df.iloc[0]['Test R2']})")

    # ---------------- Classification ----------------
    log("Training classification models...")
    y_class_train = (y_train >= pass_threshold).astype(int)
    y_class_test = (y_test >= pass_threshold).astype(int)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_class_train)

    clf_models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=10, class_weight="balanced",
            max_features="sqrt", random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
        ),
    }
    try:
        from xgboost import XGBClassifier

        clf_models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
    except ImportError:
        log("xgboost not installed - skipping XGBoost classifier")

    clf_results, trained_clf = [], {}
    for name, clf in clf_models.items():
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        trained_clf[name] = clf
        clf_results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_class_test, y_pred), 4),
                "Precision": round(precision_score(y_class_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_class_test, y_pred, zero_division=0), 4),
                "F1 Score": round(f1_score(y_class_test, y_pred, zero_division=0), 4),
                "AUC": round(roc_auc_score(y_class_test, y_prob), 4),
            }
        )
        log(f"  {name:22s} F1={clf_results[-1]['F1 Score']:.4f} AUC={clf_results[-1]['AUC']:.4f}")

    clf_df = pd.DataFrame(clf_results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    best_clf_name = clf_df.iloc[0]["Model"]
    best_clf = trained_clf[best_clf_name]
    log(f"Best classifier: {best_clf_name} (F1={clf_df.iloc[0]['F1 Score']})")

    # ---------------- Wrap & save production pipelines ----------------
    reg_pipeline = StudentPerformancePipeline(
        model=best_reg, imputer=imputer, scaler=scaler, task="regression",
        gender_mode=gender_mode, support_mode=support_mode, model_name=best_reg_name,
    )
    clf_pipeline = StudentPerformancePipeline(
        model=best_clf, imputer=imputer, scaler=scaler, task="classification",
        gender_mode=gender_mode, support_mode=support_mode, model_name=best_clf_name,
    )

    joblib.dump(reg_pipeline, os.path.join(MODELS_DIR, "regression_pipeline.pkl"))
    joblib.dump(clf_pipeline, os.path.join(MODELS_DIR, "classification_pipeline.pkl"))
    log("Saved regression_pipeline.pkl and classification_pipeline.pkl")

    metrics = {
        "pass_threshold": pass_threshold,
        "rows_original": int(len(df)),
        "rows_after_cleaning": int(len(df_clean)),
        "best_regression_model": best_reg_name,
        "best_classifier_model": best_clf_name,
        "regression_results": reg_df.to_dict(orient="records"),
        "classification_results": clf_df.to_dict(orient="records"),
        "feature_importance_regression": reg_pipeline.feature_importance(),
        "feature_importance_classification": clf_pipeline.feature_importance(),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLS,
                "gender_map": GENDER_MAP,
                "support_map": SUPPORT_MAP,
                "gender_mode": gender_mode,
                "support_mode": support_mode,
            },
            f,
            indent=2,
        )

    log("Saved metrics.json and feature_names.json")
    log("Training complete. The Flask app will now load these pipelines directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export the student performance ML pipelines.")
    parser.add_argument("--dataset", default=None, help="Path to the training dataset .xlsx file")
    parser.add_argument(
        "--pass-threshold", type=float, default=None,
        help=f"Grade cut-off for Pass/Fail classification (default: {DEFAULT_PASS_THRESHOLD})",
    )
    args = parser.parse_args()
    main(dataset_path=args.dataset, pass_threshold=args.pass_threshold)
