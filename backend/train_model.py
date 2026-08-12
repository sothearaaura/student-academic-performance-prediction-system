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
from imblearn.pipeline import Pipeline as ImbPipeline
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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
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

# Monotonic direction constraints, in the exact order of FEATURE_COLS.
# +1 = increasing this feature must never predict a WORSE outcome
#  0 = no domain-sensible direction to enforce
#
# NOTE: Grade_Study_Ratio = PreviousGrade / (StudyHours_Final + 1) is
# EXCLUDED from the monotonic feature set entirely (see MONOTONIC_FEATURE_COLS
# below), not just left unconstrained. It structurally DECREASES as
# StudyHours_Final increases (for a fixed previous grade), which directly
# fights the StudyHours_Final: +1 constraint -- verified empirically: with
# it included (even unconstrained), the sensitivity sweep on study_hours
# still swung backwards, because the model routed around the direct
# constraint through this derived feature.
MONOTONIC_FEATURE_COLS = [c for c in FEATURE_COLS if c != "Grade_Study_Ratio"]
MONOTONE_CONSTRAINTS = tuple(1 if c in ("Attendance_Final", "StudyHours_Final", "PreviousGrade",
                                          "ParentalSupport", "Attendance_Study_Interaction") else 0
                              for c in MONOTONIC_FEATURE_COLS)
assert len(MONOTONE_CONSTRAINTS) == len(MONOTONIC_FEATURE_COLS)

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

    # Parallel preprocessing for the monotonic candidates, which use a
    # reduced feature set (see MONOTONIC_FEATURE_COLS above) -- needs its
    # own imputer/scaler fit on just those columns.
    X_train_mono = X_train[MONOTONIC_FEATURE_COLS]
    X_test_mono = X_test[MONOTONIC_FEATURE_COLS]
    imputer_mono = SimpleImputer(strategy="median")
    X_train_mono_imp = imputer_mono.fit_transform(X_train_mono)
    X_test_mono_imp = imputer_mono.transform(X_test_mono)
    scaler_mono = StandardScaler()
    X_train_mono_scaled = scaler_mono.fit_transform(X_train_mono_imp)
    X_test_mono_scaled = scaler_mono.transform(X_test_mono_imp)

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
    reg_models_mono = {}
    try:
        from xgboost import XGBRegressor

        reg_models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, random_state=42, verbosity=0,
        )
        reg_models_mono["XGBoost (Monotonic)"] = XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, random_state=42, verbosity=0,
            monotone_constraints=MONOTONE_CONSTRAINTS,
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

    for name, model in reg_models_mono.items():
        model.fit(X_train_mono_scaled, y_train)
        y_tr = model.predict(X_train_mono_scaled)
        y_te = model.predict(X_test_mono_scaled)
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

    # On a dataset this weak (Test R2 near zero for every candidate), letting
    # "highest Test R2 wins" pick the production model is how we ended up
    # with a model that predicts WORSE grades for MORE attendance/study
    # hours -- a fraction-of-a-point R2 edge isn't worth deploying a model
    # that behaves backwards on the inputs students see and act on. If a
    # monotonic-constrained candidate was trained, it is always deployed
    # instead, regardless of whether it happens to win on raw Test R2.
    if "XGBoost (Monotonic)" in trained_reg:
        best_reg_name = "XGBoost (Monotonic)"
        best_reg = trained_reg[best_reg_name]
        best_row = reg_df[reg_df["Model"] == best_reg_name].iloc[0]
        log(
            f"Deploying {best_reg_name} (Test R2={best_row['Test R2']}) over the raw "
            f"leaderboard winner '{reg_df.iloc[0]['Model']}' (Test R2={reg_df.iloc[0]['Test R2']}) "
            "-- monotonic sanity is prioritized over a marginal, likely-noise accuracy edge."
        )
    else:
        best_reg_name = reg_df.iloc[0]["Model"]
        best_reg = trained_reg[best_reg_name]
        log(f"Best regression model: {best_reg_name} (Test R2={reg_df.iloc[0]['Test R2']})")

    # ---------------- Classification ----------------
    log("Training classification models...")
    y_class_train = (y_train >= pass_threshold).astype(int)
    y_class_test = (y_test >= pass_threshold).astype(int)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_class_train)
    X_train_mono_res, y_train_mono_res = sm.fit_resample(X_train_mono_scaled, y_class_train)

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
    clf_models_mono = {}
    try:
        from xgboost import XGBClassifier

        clf_models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        clf_models_mono["XGBoost (Monotonic)"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, eval_metric="logloss",
            random_state=42, verbosity=0,
            monotone_constraints=MONOTONE_CONSTRAINTS,
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

    for name, clf in clf_models_mono.items():
        clf.fit(X_train_mono_res, y_train_mono_res)
        y_pred = clf.predict(X_test_mono_scaled)
        y_prob = clf.predict_proba(X_test_mono_scaled)[:, 1]
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

    if "XGBoost (Monotonic)" in trained_clf:
        best_clf_name = "XGBoost (Monotonic)"
        best_clf = trained_clf[best_clf_name]
        best_row = clf_df[clf_df["Model"] == best_clf_name].iloc[0]
        log(
            f"Deploying {best_clf_name} (F1={best_row['F1 Score']}) over the raw "
            f"leaderboard winner '{clf_df.iloc[0]['Model']}' (F1={clf_df.iloc[0]['F1 Score']}) "
            "-- same monotonic-sanity priority as the regressor."
        )
    else:
        best_clf_name = clf_df.iloc[0]["Model"]
        best_clf = trained_clf[best_clf_name]
        log(f"Best classifier: {best_clf_name} (F1={clf_df.iloc[0]['F1 Score']})")

    # ---------------- Wrap & save production pipelines ----------------
    # A model trained on the reduced monotonic feature set needs the
    # matching imputer/scaler/feature_cols -- feeding it the full 9-column
    # preprocessing would silently misalign the columns.
    reg_feature_cols = MONOTONIC_FEATURE_COLS if best_reg_name == "XGBoost (Monotonic)" else FEATURE_COLS
    reg_imputer = imputer_mono if best_reg_name == "XGBoost (Monotonic)" else imputer
    reg_scaler = scaler_mono if best_reg_name == "XGBoost (Monotonic)" else scaler

    clf_feature_cols = MONOTONIC_FEATURE_COLS if best_clf_name == "XGBoost (Monotonic)" else FEATURE_COLS
    clf_imputer = imputer_mono if best_clf_name == "XGBoost (Monotonic)" else imputer
    clf_scaler = scaler_mono if best_clf_name == "XGBoost (Monotonic)" else scaler

    # ---------------- Cross-validation of the DEPLOYED models ----------------
    # A single train/test split can be noisy, especially on a dataset this
    # size -- 5-fold CV on the full training set gives a more robust,
    # honest estimate of how the actually-deployed model generalizes,
    # rather than relying on one lucky/unlucky split.
    #
    # IMPORTANT: this must refit the imputer AND scaler independently within
    # EACH fold, using only that fold's training portion. Passing
    # already-scaled data (scaled using statistics fit on the FULL X_train
    # before the folds are even formed) would leak each fold's held-out rows
    # into the preprocessing statistics used for its own training portion --
    # a real, if usually small-impact, form of cross-validation leakage.
    # Wrapping imputer+scaler+model in a Pipeline and handing cross_val_score
    # the RAW (unscaled, unimputed) fold data makes it refit both from
    # scratch inside each fold.
    log("Cross-validating the deployed models (5-fold, leakage-free preprocessing per fold)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    X_reg_raw = X_train_mono if best_reg_name == "XGBoost (Monotonic)" else X_train
    reg_cv_pipeline = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", type(best_reg)(**best_reg.get_params())),
    ])
    reg_cv_scores = cross_val_score(reg_cv_pipeline, X_reg_raw, y_train, cv=kf, scoring="r2")
    log(f"  Regression 5-fold CV R2: mean={reg_cv_scores.mean():.4f} std={reg_cv_scores.std():.4f}")

    X_clf_raw = X_train_mono if best_clf_name == "XGBoost (Monotonic)" else X_train
    clf_cv_pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", type(best_clf)(**best_clf.get_params())),
    ])
    # SMOTE belongs INSIDE each CV fold. That makes the CV estimate match the
    # actual training recipe while preventing synthetic samples derived from
    # a validation fold from leaking into that fold's evaluation.
    clf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_cv_scores = cross_val_score(clf_cv_pipeline, X_clf_raw, y_class_train, cv=clf_cv, scoring="f1")
    log(f"  Classification 5-fold CV F1: mean={clf_cv_scores.mean():.4f} std={clf_cv_scores.std():.4f}")

    reg_pipeline = StudentPerformancePipeline(
        model=best_reg, imputer=reg_imputer, scaler=reg_scaler, task="regression",
        feature_cols=reg_feature_cols,
        gender_mode=gender_mode, support_mode=support_mode, model_name=best_reg_name,
    )
    clf_pipeline = StudentPerformancePipeline(
        model=best_clf, imputer=clf_imputer, scaler=clf_scaler, task="classification",
        feature_cols=clf_feature_cols,
        gender_mode=gender_mode, support_mode=support_mode, model_name=best_clf_name,
    )

    joblib.dump(reg_pipeline, os.path.join(MODELS_DIR, "regression_pipeline.pkl"))
    joblib.dump(clf_pipeline, os.path.join(MODELS_DIR, "classification_pipeline.pkl"))
    log("Saved regression_pipeline.pkl and classification_pipeline.pkl")

    deployed_reg_row = reg_df[reg_df["Model"] == best_reg_name].iloc[0].to_dict()
    deployed_clf_row = clf_df[clf_df["Model"] == best_clf_name].iloc[0].to_dict()

    # Confusion matrix for the DEPLOYED classifier specifically, using its
    # correct feature set (full or reduced-monotonic).
    X_clf_test_for_cm = X_test_mono_scaled if best_clf_name == "XGBoost (Monotonic)" else X_test_scaled
    y_pred_deployed = trained_clf[best_clf_name].predict(X_clf_test_for_cm)
    cm = confusion_matrix(y_class_test, y_pred_deployed)
    confusion = {
        "true_negative_fail_correctly_predicted_fail": int(cm[0][0]),
        "false_positive_fail_predicted_as_pass": int(cm[0][1]),
        "false_negative_pass_predicted_as_fail": int(cm[1][0]),
        "true_positive_pass_correctly_predicted_pass": int(cm[1][1]),
    }
    log(f"  Deployed classifier confusion matrix: {confusion}")

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
        "deployed_regression_monotonic": best_reg_name == "XGBoost (Monotonic)",
        "deployed_classification_monotonic": best_clf_name == "XGBoost (Monotonic)",
        "deployed_regression_test_r2": deployed_reg_row["Test R2"],
        "deployed_classification_f1": deployed_clf_row["F1 Score"],
        "deployed_regression_cv_r2_mean": round(float(reg_cv_scores.mean()), 4),
        "deployed_regression_cv_r2_std": round(float(reg_cv_scores.std()), 4),
        "deployed_classification_cv_f1_mean": round(float(clf_cv_scores.mean()), 4),
        "deployed_classification_cv_f1_std": round(float(clf_cv_scores.std()), 4),
        "deployed_classification_confusion_matrix": confusion,
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Observed training ranges for the key raw inputs, used at prediction
    # time to warn when a submitted value falls well outside what the
    # model actually learned from -- a prediction for attendance=20 when
    # training data only ever saw 70-95 is an extrapolation, not an
    # interpolation, and the UI should say so rather than presenting it
    # with the same confidence as an in-distribution prediction.
    training_ranges = {
        "attendance": [float(X_train["Attendance_Final"].min()), float(X_train["Attendance_Final"].max())],
        "study_hours": [float(X_train["StudyHours_Final"].min()), float(X_train["StudyHours_Final"].max())],
        "previous_grade": [float(X_train["PreviousGrade"].min()), float(X_train["PreviousGrade"].max())],
    }

    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLS,
                "gender_map": GENDER_MAP,
                "support_map": SUPPORT_MAP,
                "gender_mode": gender_mode,
                "support_mode": support_mode,
                "training_ranges": training_ranges,
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
