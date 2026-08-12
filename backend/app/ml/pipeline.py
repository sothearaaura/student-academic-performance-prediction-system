"""
StudentPerformancePipeline
===========================
This module is the single source of truth for turning raw student input
(attendance, study hours, previous grade, etc.) into the exact feature
vector the models in `Student_Prediction_v3_corrected.ipynb` were trained
on. It is imported by BOTH:

  1. train_model.py   -> fits the imputer/scaler/model and wraps them here
  2. app/services/ml_service.py -> loads the saved .pkl and calls .predict()

Keeping this logic in one shared class guarantees the website's prediction
always matches what the notebook's pipeline would produce for the same
raw input -- there is no second, hand-copied preprocessing implementation
that could drift out of sync.

Feature order / encoding maps below are copied verbatim from Step 5 of the
notebook ("STEP 5 - Preprocessing").
"""
from __future__ import annotations

import numpy as np

FEATURE_COLS = [
    "Attendance_Final",
    "StudyHours_Final",
    "PreviousGrade",
    "ExtracurricularActivities",
    "Gender",
    "ParentalSupport",
    "Online_Classes",
    "Attendance_Study_Interaction",
    "Grade_Study_Ratio",
]

GENDER_MAP = {"Male": 0, "Female": 1}
SUPPORT_MAP = {"Low": 0, "Medium": 1, "High": 2}
PASS_THRESHOLD = 70.0


class StudentPerformancePipeline:
    """A fitted, picklable prediction pipeline for one task (regression or
    classification). Bundles the fitted SimpleImputer, StandardScaler and
    final estimator together with the fixed encoding maps used at training
    time, so a single joblib.load() + .predict(raw_dict) round-trip is all
    the Flask app ever needs to do.
    """

    def __init__(
        self,
        model,
        imputer,
        scaler,
        task: str,
        feature_cols=None,
        gender_map=None,
        support_map=None,
        gender_mode: str = "Male",
        support_mode: str = "Medium",
        model_name: str = "",
    ):
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        self.model = model
        self.imputer = imputer
        self.scaler = scaler
        self.task = task
        self.feature_cols = feature_cols or FEATURE_COLS
        self.gender_map = gender_map or GENDER_MAP
        self.support_map = support_map or SUPPORT_MAP
        self.gender_mode = gender_mode
        self.support_mode = support_mode
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Feature engineering (mirrors notebook Step 5-A / 5-D exactly)
    # ------------------------------------------------------------------
    def build_feature_row(self, raw: dict) -> list:
        attendance = float(raw["attendance"])
        study_hours = float(raw["study_hours"])
        previous_grade = float(raw["previous_grade"])
        extracurricular = float(raw.get("extracurricular", 0) or 0)
        online = int(bool(raw.get("online_classes", 0)))

        gender_raw = str(raw.get("gender") or self.gender_mode).strip().title()
        support_raw = str(raw.get("parental_support") or self.support_mode).strip().title()

        gender_enc = self.gender_map.get(gender_raw, self.gender_map[self.gender_mode])
        support_enc = self.support_map.get(support_raw, self.support_map[self.support_mode])

        interaction = attendance * study_hours
        ratio = previous_grade / (study_hours + 1)

        row = {
            "Attendance_Final": attendance,
            "StudyHours_Final": study_hours,
            "PreviousGrade": previous_grade,
            "ExtracurricularActivities": extracurricular,
            "Gender": gender_enc,
            "ParentalSupport": support_enc,
            "Online_Classes": online,
            "Attendance_Study_Interaction": interaction,
            "Grade_Study_Ratio": ratio,
        }
        return [row[c] for c in self.feature_cols]

    def transform(self, raw: dict) -> np.ndarray:
        import pandas as pd

        row = self.build_feature_row(raw)
        X = pd.DataFrame([row], columns=self.feature_cols)
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return X_scaled

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, raw: dict):
        X_scaled = self.transform(raw)
        if self.task == "regression":
            value = float(self.model.predict(X_scaled)[0])
            # A final grade is bounded [0, 100] by definition -- the model
            # was trained on data clipped to that range but was never given
            # an explicit output constraint, so it can (and did) extrapolate
            # slightly past 100 for very high inputs.
            value = max(0.0, min(100.0, value))
            return {"predicted_grade": round(value, 2)}

        pred_class = int(self.model.predict(X_scaled)[0])
        proba = np.asarray(self.model.predict_proba(X_scaled)[0], dtype=float)
        classes = np.asarray(getattr(self.model, "classes_", [0, 1]))
        probabilities = {int(label): float(probability) for label, probability in zip(classes, proba)}
        pass_probability = probabilities.get(1, 0.0)
        winning_probability = probabilities.get(pred_class, float(proba.max()))
        return {
            "pass_fail": "Pass" if pred_class == 1 else "Fail",
            "pass_probability": round(pass_probability * 100, 2),
            "confidence": round(winning_probability * 100, 2),
            "classifier_class": pred_class,
            "classifier_classes": [int(label) for label in classes],
        }

    def feature_importance(self) -> dict:
        if hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_
            values = coef.ravel() if hasattr(coef, "ravel") else coef
            values = np.abs(values)
        else:
            return {}
        return dict(zip(self.feature_cols, [float(v) for v in values]))
