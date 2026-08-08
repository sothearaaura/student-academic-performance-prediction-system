import os
from datetime import datetime

from flask import Blueprint, current_app, flash, redirect, render_template, request, send_file, url_for
from flask_login import current_user, login_required

from app.extensions import db
from app.models import ROLE_STUDENT, ActivityLog, AppSetting, Prediction, Report
from app.services import ml_service, report_service
from app.utils import roles_required

student_bp = Blueprint("student", __name__, url_prefix="/student")


@student_bp.before_request
@login_required
@roles_required(ROLE_STUDENT)
def guard():
    pass


def _my_student_record():
    profile = current_user.student_profile
    if profile is None:
        return None
    return profile


@student_bp.route("/dashboard")
def dashboard():
    profile = _my_student_record()
    if profile is None:
        flash("Your account is not yet linked to a student record. Contact your admin.", "warning")
        return render_template("student/dashboard.html", profile=None, prediction=None, history=[], history_data=[], recommendations=[], weak_areas=[], total_predictions=0, avg_predicted_grade=None, highest_predicted_grade=None)

    # Only show a prediction if the student has actually made one (via the
    # New Prediction page) or an admin has run one for them -- the dashboard
    # never silently generates a prediction on its own. A brand-new student
    # who hasn't submitted anything yet should see an empty state, not a
    # fabricated-looking result.
    prediction = profile.latest_prediction()
    recommendations, weak_areas = [], []
    if prediction:
        raw_snapshot = {
            "attendance": prediction.attendance,
            "study_hours": prediction.study_hours,
            "previous_grade": prediction.previous_grade,
            "extracurricular": prediction.extracurricular,
            "gender": prediction.gender,
            "parental_support": prediction.parental_support,
            "online_classes": prediction.online_classes,
        }
        recommendations = ml_service.recommendations_for(raw_snapshot, prediction.predicted_grade, prediction.pass_fail)
        weak_areas = ml_service.weak_areas_for(raw_snapshot)
        prediction.risk_level = ml_service.risk_level_for(prediction.pass_fail, prediction.pass_probability)

    history = profile.predictions.limit(10).all()
    history_data = [h.to_dict() for h in history]

    all_predictions = profile.predictions.all()
    total_predictions = len(all_predictions)
    avg_predicted_grade = (
        round(sum(p.predicted_grade for p in all_predictions) / total_predictions, 1)
        if total_predictions else None
    )
    highest_predicted_grade = max((p.predicted_grade for p in all_predictions), default=None)

    return render_template(
        "student/dashboard.html",
        profile=profile,
        prediction=prediction,
        history=history,
        history_data=history_data,
        recommendations=recommendations,
        weak_areas=weak_areas,
        total_predictions=total_predictions,
        avg_predicted_grade=avg_predicted_grade,
        highest_predicted_grade=highest_predicted_grade,
        now_hour=datetime.utcnow().hour,
    )


@student_bp.route("/new-prediction", methods=["GET", "POST"])
def new_prediction():
    """Self-service 'what-if' predictor: the student can try out different
    hypothetical values (not necessarily their real stored data) and see an
    instant AI prediction, matching the New Prediction wizard. Results are
    saved to their history like any other prediction."""
    profile = _my_student_record()
    result = None
    raw = None

    if request.method == "POST":
        raw = {
            "attendance": float(request.form.get("attendance", 0) or 0),
            "study_hours": float(request.form.get("study_hours", 0) or 0),
            "previous_grade": float(request.form.get("previous_grade", 0) or 0),
            "extracurricular": float(request.form.get("extracurricular", 0) or 0),
            "gender": request.form.get("gender", "Male"),
            "parental_support": request.form.get("parental_support", "Medium"),
            "online_classes": bool(request.form.get("online_classes")),
        }
        try:
            result = ml_service.predict_full(raw)
        except RuntimeError as exc:
            flash(str(exc), "danger")
            return render_template("student/new_prediction.html", profile=profile, result=None)

        if profile:
            # Keep the student's actual profile in sync with what they just told
            # the system about themselves -- otherwise admin's Manage Students
            # page would keep showing stale/blank data even after the student
            # has clearly reported real numbers via a prediction.
            profile.attendance = raw["attendance"]
            profile.study_hours = raw["study_hours"]
            profile.previous_grade = raw["previous_grade"]
            profile.extracurricular = raw["extracurricular"]
            profile.gender = raw["gender"]
            profile.parental_support = raw["parental_support"]
            profile.online_classes = raw["online_classes"]
            profile.current_grade = result["predicted_grade"]

            stored = Prediction(
                student_id=profile.id,
                created_by_id=current_user.id,
                attendance=raw["attendance"],
                study_hours=raw["study_hours"],
                previous_grade=raw["previous_grade"],
                extracurricular=raw["extracurricular"],
                gender=raw["gender"],
                parental_support=raw["parental_support"],
                online_classes=raw["online_classes"],
                predicted_grade=result["predicted_grade"],
                pass_fail=result["pass_fail"],
                pass_probability=result["pass_probability"],
                confidence=result["confidence"],
                performance_level=result["performance_level"],
                regression_model_name=result["regression_model_name"],
                classifier_model_name=result["classifier_model_name"],
            )
            db.session.add(stored)
            db.session.commit()
            ActivityLog.log(current_user.id, "predict", "Student ran a self-service what-if prediction")

    return render_template("student/new_prediction.html", profile=profile, result=result, raw=raw if result else None)


@student_bp.route("/history")
def history():
    profile = _my_student_record()
    predictions = profile.predictions.all() if profile else []
    return render_template("student/history.html", predictions=predictions)


@student_bp.route("/about")
def about():
    metrics = ml_service.get_metrics()
    feature_names = ml_service.get_feature_names()
    settings = AppSetting.get()
    return render_template(
        "student/about.html",
        feature_count=len(feature_names.get("feature_columns", [])) or 9,
        best_classifier=metrics.get("best_classifier_model", "AI Model"),
        pass_threshold=settings.pass_threshold,
    )


@student_bp.route("/report")
def download_report():
    profile = _my_student_record()
    if profile is None:
        flash("No student record linked to your account yet.", "warning")
        return redirect(url_for("student.dashboard"))

    prediction = profile.latest_prediction()
    filename = f"my_report_{profile.student_code}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(current_app.config["EXPORTS_DIR"], filename)
    report_service.generate_student_report(profile, prediction, output_path, current_user.full_name)

    report = Report(
        student_id=profile.id,
        generated_by_id=current_user.id,
        report_type="student",
        file_path=output_path,
        file_name=filename,
    )
    db.session.add(report)
    db.session.commit()
    ActivityLog.log(current_user.id, "download_report", "Student downloaded own report")
    return send_file(output_path, as_attachment=True, download_name=filename)
