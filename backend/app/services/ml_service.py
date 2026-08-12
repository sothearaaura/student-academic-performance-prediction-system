"""
Loads the trained regression/classification pipelines ONCE at app startup.
No retraining ever happens here -- train_model.py is the only place models
are fit. This module purely loads the .pkl files and serves predictions.
"""
import json
import os

import joblib

from app.ml.pipeline import PASS_THRESHOLD as _FALLBACK_PASS_THRESHOLD

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


def current_pass_threshold() -> float:
    """The threshold actually used to train the currently-deployed
    classifier (stored in metrics.json at train time). This is what
    display-time Pass/Fail MUST compare against -- using a separate
    hardcoded constant here would silently drift out of sync with the
    admin-configured threshold the moment someone retrains with a
    different value, making Result inconsistent with the classifier's own
    trained decision boundary."""
    if _metrics and "pass_threshold" in _metrics:
        return float(_metrics["pass_threshold"])
    return _FALLBACK_PASS_THRESHOLD


def get_feature_names() -> dict:
    return _feature_names or {}


_LEVEL_ORDER = ["At Risk", "Needs Improvement", "Good", "Very Good", "Excellent"]


def _grade_only_level(predicted_grade: float, pass_threshold: float = None) -> str:
    """Level's tiers anchor to the SAME configurable pass threshold Result
    uses (not a separately hardcoded 70), so the "Good" tier always starts
    exactly at the threshold -- Level and Result structurally align at that
    boundary no matter what the admin sets the threshold to, rather than
    only being patched into agreement after the fact."""
    threshold = pass_threshold if pass_threshold is not None else current_pass_threshold()
    if predicted_grade >= threshold + 20:
        return "Excellent"
    if predicted_grade >= threshold + 10:
        return "Very Good"
    if predicted_grade >= threshold:
        return "Good"
    if predicted_grade >= threshold - 10:
        return "Needs Improvement"
    return "At Risk"


def _performance_level(predicted_grade: float, pass_fail: str = None) -> str:
    """The Level label is a bucket on the predicted grade, but the grade
    model and the pass/fail classifier are trained independently and can
    disagree (e.g. a grade in the "Very Good" range with a "Fail" result).
    Showing a positive-sounding label right next to a Fail result is
    genuinely misleading, so when a pass_fail verdict is available, the
    Level is reconciled against it: Fail can never show a level better than
    "Needs Improvement", and Pass can never show "At Risk"."""
    level = _grade_only_level(predicted_grade)
    if pass_fail is None:
        return level

    idx = _LEVEL_ORDER.index(level)
    needs_improvement_idx = _LEVEL_ORDER.index("Needs Improvement")

    if pass_fail == "Fail" and idx > needs_improvement_idx:
        return "Needs Improvement"
    if pass_fail == "Pass" and idx < needs_improvement_idx:
        return "Needs Improvement"
    return level


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
    attendance = raw.get("attendance", 100)
    if attendance < 60:
        tips.append(
            f"Critical attendance issue ({attendance:g}%) — attendance this low is a serious risk "
            "factor on its own; recommend an immediate meeting with the student and guardians."
        )
    elif attendance < 80:
        tips.append(
            f"Attendance ({attendance:g}%) needs improvement — regular class attendance strongly "
            "correlates with better outcomes. Aim for 80%+."
        )

    study_hours = raw.get("study_hours", 0)
    if study_hours < 5:
        tips.append(
            f"Study time ({study_hours:g} hrs/week) is critically low — this alone puts the student "
            "at serious risk regardless of other factors. Aim for at least 10-15 focused hours per week."
        )
    elif study_hours < 10:
        tips.append(
            f"Study hours ({study_hours:g} hrs/week) need improvement — aim for at least 10-15 "
            "focused hours per week."
        )

    previous_grade = raw.get("previous_grade", 100)
    if previous_grade < 50:
        tips.append(
            f"Previous grade ({previous_grade:g}) indicates a serious foundational gap — strongly "
            "recommend tutoring or remedial review before this student can realistically catch up."
        )
    elif previous_grade < 70:
        tips.append(
            f"Previous grade ({previous_grade:g}) needs improvement — consider tutoring or reviewing "
            "foundational material, since prior performance is a strong predictor of future grades."
        )

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


def _ood_warnings(raw: dict) -> list:
    """Flags inputs that fall outside the range the model actually saw
    during training -- a prediction for those is an extrapolation the
    model was never validated on, not a normal in-distribution estimate."""
    ranges = (_feature_names or {}).get("training_ranges")
    if not ranges:
        return []
    warnings = []
    labels = {"attendance": "Attendance", "study_hours": "Study Hours", "previous_grade": "Previous Grade"}
    for key, (lo, hi) in ranges.items():
        val = raw.get(key)
        if val is None:
            continue
        val = float(val)
        if val < lo or val > hi:
            warnings.append(
                f"{labels.get(key, key)} of {val:g} is outside the training data's observed "
                f"range ({lo:g}\u2013{hi:g}) -- this prediction is an extrapolation, not backed "
                f"by data the model was actually validated on."
            )
    return warnings


def predict_full(raw: dict) -> dict:
    """Runs both pipelines and returns a combined, UI-ready prediction result."""
    if not is_ready():
        raise RuntimeError(
            "ML pipelines are not loaded. Run `python train_model.py` inside backend/ first."
        )

    reg_result = _reg_pipeline.predict(raw)
    clf_result = _clf_pipeline.predict(raw)

    predicted_grade = reg_result["predicted_grade"]

    # Result (Pass/Fail) is derived directly from Predicted Grade, using the
    # SAME threshold the currently-deployed classifier was actually trained
    # with (read live from metrics.json, never a hardcoded constant) -- this
    # guarantees Result, the classifier's decision boundary, and the admin
    # Settings "Pass Threshold" value can never drift out of sync, and
    # ensures Result can't contradict Level either (both are grade-based).
    pass_threshold = current_pass_threshold()
    pass_fail = "Pass" if predicted_grade >= pass_threshold else "Fail"

    result = {
        "predicted_grade": predicted_grade,
        "pass_fail": pass_fail,
        # pass_probability/confidence remain the classifier's own estimate --
        # informational context (how confident that separate model is), not
        # what determines the Pass/Fail label shown above.
        "pass_probability": clf_result["pass_probability"],
        "confidence": clf_result["confidence"],
        "performance_level": _performance_level(predicted_grade, pass_fail),
        "risk_level": _risk_level(pass_fail, clf_result["pass_probability"]),
        "recommendations": _recommendations(raw, predicted_grade, pass_fail),
        "weak_areas": _weak_areas(raw),
        "ood_warnings": _ood_warnings(raw),
        "regression_model_name": getattr(_reg_pipeline, "model_name", ""),
        "classifier_model_name": getattr(_clf_pipeline, "model_name", ""),
    }
    return result


# Public wrappers: these are pure threshold/lookup logic, no model inference
# needed, so callers can use them to recompute display data (recommendations,
# weak areas, risk level) from an already-STORED prediction's saved inputs,
# without re-running the model or requiring it to be loaded.
def recommendations_for(raw: dict, predicted_grade: float, pass_fail: str) -> list:
    return _recommendations(raw, predicted_grade, pass_fail)


def weak_areas_for(raw: dict) -> list:
    return _weak_areas(raw)


def risk_level_for(pass_fail: str, pass_probability: float) -> str:
    return _risk_level(pass_fail, pass_probability)


def performance_level_for(predicted_grade: float, pass_fail: str = None) -> str:
    return _performance_level(predicted_grade, pass_fail)
