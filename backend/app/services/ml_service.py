"""
Loads the trained regression/classification pipelines ONCE at app startup.
No retraining ever happens here -- train_model.py is the only place models
are fit. This module purely loads the .pkl files and serves predictions.
"""
import json
import os

import joblib

_reg_pipeline = None
_clf_pipeline = None
_metrics = None
_feature_names = None
_models_dir = None


def init_app(app):
    global _reg_pipeline, _clf_pipeline, _metrics, _feature_names, _models_dir
    _models_dir = app.config["MODELS_DIR"]

    reg_path = os.path.join(_models_dir, "regression_pipeline.pkl")
    clf_path = os.path.join(_models_dir, "classification_pipeline.pkl")
    metrics_path = os.path.join(_models_dir, "metrics.json")
    feat_path = os.path.join(_models_dir, "feature_names.json")

    if os.path.exists(reg_path):
        _reg_pipeline = joblib.load(reg_path)
    else:
        app.logger.warning(
            "regression_pipeline.pkl not found - run `python train_model.py` first"
        )

    if os.path.exists(clf_path):
        _clf_pipeline = joblib.load(clf_path)
    else:
        app.logger.warning(
            "classification_pipeline.pkl not found - run `python train_model.py` first"
        )

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _metrics = json.load(f)
    else:
        _metrics = {}

    if os.path.exists(feat_path):
        with open(feat_path) as f:
            _feature_names = json.load(f)
    else:
        _feature_names = {}


def is_ready() -> bool:
    return _reg_pipeline is not None and _clf_pipeline is not None


def get_metrics() -> dict:
    return _metrics or {}


def get_feature_names() -> dict:
    return _feature_names or {}


def _performance_level(predicted_grade: float) -> str:
    if predicted_grade >= 90:
        return "Excellent"
    if predicted_grade >= 80:
        return "Very Good"
    if predicted_grade >= 70:
        return "Good"
    if predicted_grade >= 60:
        return "Needs Improvement"
    return "At Risk"


def _risk_level(pass_fail: str, pass_probability: float) -> str:
    """Maps the prediction to the explicit Risk Level called for in the
    grade-prediction flow (Predicted Grade / Confidence Score / Risk Level /
    Suggestions for Improvement)."""
    if pass_fail == "Fail":
        return "High Risk"
    if pass_probability < 70:
        return "Medium Risk"
    return "Low Risk"


def _recommendations(raw: dict, predicted_grade: float, pass_fail: str) -> list:
    tips = []
    if raw.get("attendance", 100) < 80:
        tips.append("Attendance is below 80% — regular class attendance strongly correlates with better outcomes.")
    if raw.get("study_hours", 0) < 10:
        tips.append("Increase weekly study hours — aim for at least 10-15 focused hours per week.")
    if raw.get("previous_grade", 100) < 70:
        tips.append("Previous grade is a strong predictor — consider tutoring or reviewing foundational material.")
    if str(raw.get("parental_support", "Medium")).title() == "Low":
        tips.append("Encourage more parental/guardian involvement in academic planning.")
    if not raw.get("online_classes"):
        tips.append("Consider supplementing coursework with online classes for extra practice.")
    if pass_fail == "Fail":
        tips.append("This student is at risk of failing — recommend an early intervention meeting.")
    if not tips:
        tips.append("Great trajectory — keep up the current study habits and attendance.")
    return tips


def _weak_areas(raw: dict) -> list:
    weak = []
    if raw.get("attendance", 100) < 80:
        weak.append("Attendance")
    if raw.get("study_hours", 0) < 10:
        weak.append("Study Hours")
    if raw.get("previous_grade", 100) < 70:
        weak.append("Prior Academic Performance")
    if raw.get("extracurricular", 0) == 0:
        weak.append("Extracurricular Engagement")
    return weak or ["None identified"]


def predict_full(raw: dict) -> dict:
    """Runs both pipelines and returns a combined, UI-ready prediction result."""
    if not is_ready():
        raise RuntimeError(
            "ML pipelines are not loaded. Run `python train_model.py` inside backend/ first."
        )

    reg_result = _reg_pipeline.predict(raw)
    clf_result = _clf_pipeline.predict(raw)

    predicted_grade = reg_result["predicted_grade"]
    pass_fail = clf_result["pass_fail"]

    result = {
        "predicted_grade": predicted_grade,
        "pass_fail": pass_fail,
        "pass_probability": clf_result["pass_probability"],
        "confidence": clf_result["confidence"],
        "performance_level": _performance_level(predicted_grade),
        "risk_level": _risk_level(pass_fail, clf_result["pass_probability"]),
        "recommendations": _recommendations(raw, predicted_grade, pass_fail),
        "weak_areas": _weak_areas(raw),
        "regression_model_name": getattr(_reg_pipeline, "model_name", ""),
        "classifier_model_name": getattr(_clf_pipeline, "model_name", ""),
    }
    return result
