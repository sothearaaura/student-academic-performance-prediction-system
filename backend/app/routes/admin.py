import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from app.extensions import db
from app.models import (
    ROLE_ADMIN,
    ROLE_STUDENT,
    ActivityLog,
    AppSetting,
    LoginHistory,
    Prediction,
    Report,
    Student,
    User,
)
from app.services import ml_service, report_service
from app.utils import generate_student_code, parse_academic_inputs, roles_required

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.before_request
@login_required
@roles_required(ROLE_ADMIN)
def guard():
    pass


@admin_bp.route("/dashboard")
def dashboard():
    total_students = Student.query.count()
    total_predictions = Prediction.query.count()

    avg_grade_row = db.session.query(db.func.avg(Prediction.predicted_grade)).scalar()
    avg_grade = round(avg_grade_row, 2) if avg_grade_row else None

    pass_count = Prediction.query.filter_by(pass_fail="Pass").count()
    at_risk_count = Prediction.query.filter_by(pass_fail="Fail").count()
    accuracy_display = round((pass_count / total_predictions) * 100, 1) if total_predictions else 0

    recent_activity = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(10).all()
    metrics = ml_service.get_metrics()

    model_accuracy = None
    for row in metrics.get("classification_results", []):
        if row.get("Model") == metrics.get("best_classifier_model"):
            model_accuracy = round(row.get("Accuracy", 0) * 100, 2)
            break

    # simple 7-day prediction trend for chart
    today = datetime.utcnow().date()
    trend_labels, trend_values = [], []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = Prediction.query.filter(db.func.date(Prediction.created_at) == day).count()
        trend_labels.append(day.strftime("%a"))
        trend_values.append(count)

    grade_buckets = {"Excellent": 0, "Very Good": 0, "Good": 0, "Needs Improvement": 0, "At Risk": 0}
    for p in Prediction.query.all():
        grade_buckets[p.performance_level] = grade_buckets.get(p.performance_level, 0) + 1

    # Letter-grade distribution across all students' current standing
    letter_grades = {"A (90-100)": 0, "B (80-89)": 0, "C (70-79)": 0, "D (60-69)": 0, "F (0-59)": 0}
    attendance_buckets = {"<60%": 0, "60-70%": 0, "70-80%": 0, "80-90%": 0, "90%+": 0}
    study_hours_buckets = {"0-5": 0, "6-10": 0, "11-15": 0, "16-20": 0, "21+": 0}
    for s in Student.query.all():
        grade = s.current_grade if s.current_grade is not None else s.previous_grade
        if grade >= 90:
            letter_grades["A (90-100)"] += 1
        elif grade >= 80:
            letter_grades["B (80-89)"] += 1
        elif grade >= 70:
            letter_grades["C (70-79)"] += 1
        elif grade >= 60:
            letter_grades["D (60-69)"] += 1
        else:
            letter_grades["F (0-59)"] += 1

        if s.attendance < 60:
            attendance_buckets["<60%"] += 1
        elif s.attendance < 70:
            attendance_buckets["60-70%"] += 1
        elif s.attendance < 80:
            attendance_buckets["70-80%"] += 1
        elif s.attendance < 90:
            attendance_buckets["80-90%"] += 1
        else:
            attendance_buckets["90%+"] += 1

        if s.study_hours <= 5:
            study_hours_buckets["0-5"] += 1
        elif s.study_hours <= 10:
            study_hours_buckets["6-10"] += 1
        elif s.study_hours <= 15:
            study_hours_buckets["11-15"] += 1
        elif s.study_hours <= 20:
            study_hours_buckets["16-20"] += 1
        else:
            study_hours_buckets["21+"] += 1

    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(6).all()

    return render_template(
        "admin/dashboard.html",
        total_students=total_students,
        total_predictions=total_predictions,
        avg_grade=avg_grade,
        at_risk_count=at_risk_count,
        accuracy_display=accuracy_display,
        recent_activity=recent_activity,
        recent_predictions=recent_predictions,
        metrics=metrics,
        model_accuracy=model_accuracy,
        trend_labels=trend_labels,
        trend_values=trend_values,
        grade_buckets=grade_buckets,
        letter_grades=letter_grades,
        attendance_buckets=attendance_buckets,
        study_hours_buckets=study_hours_buckets,
        model_ready=ml_service.is_ready(),
    )


# ---------------------------------------------------------------- Users ----
@admin_bp.route("/users")
def users():
    role_filter = request.args.get("role", "")
    query = User.query
    if role_filter in (ROLE_ADMIN, ROLE_STUDENT):
        query = query.filter_by(role=role_filter)
    all_users = query.order_by(User.created_at.desc()).all()
    return render_template("admin/users.html", users=all_users, role_filter=role_filter)


@admin_bp.route("/users/create", methods=["GET", "POST"])
def create_user():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        role = request.form.get("role", ROLE_STUDENT)
        password = request.form.get("password", "")
        if role not in (ROLE_ADMIN, ROLE_STUDENT):
            flash("Invalid account role.", "danger")
            return redirect(url_for("admin.create_user"))
        if password and len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("admin.create_user"))

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("A user with that username or email already exists.", "danger")
            return redirect(url_for("admin.create_user"))

        user = User(full_name=full_name, username=username, email=email, role=role)
        user.set_password(password or "changeme123")
        db.session.add(user)
        db.session.commit()
        ActivityLog.log(current_user.id, "create_user", f"Created {role} account: {username}")
        flash(f"{role.title()} account '{username}' created.", "success")
        return redirect(url_for("admin.users"))

    return render_template("admin/user_form.html", user=None)


@admin_bp.route("/users/<int:user_id>/reset-password", methods=["POST"])
def reset_password(user_id):
    user = User.query.get_or_404(user_id)
    new_password = request.form.get("new_password") or "changeme123"
    user.set_password(new_password)
    db.session.commit()
    ActivityLog.log(current_user.id, "reset_password", f"Reset password for {user.username}")
    flash(f"Password for '{user.username}' has been reset.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/toggle-active", methods=["POST"])
def toggle_active(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("You cannot deactivate your own account.", "danger")
        return redirect(url_for("admin.users"))
    user.is_active_flag = not user.is_active_flag
    db.session.commit()
    ActivityLog.log(current_user.id, "toggle_active", f"Set {user.username} active={user.is_active_flag}")
    flash(f"User '{user.username}' is now {'active' if user.is_active_flag else 'inactive'}.", "info")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("You cannot delete your own account.", "danger")
        return redirect(url_for("admin.users"))
    username = user.username
    db.session.delete(user)
    db.session.commit()
    ActivityLog.log(current_user.id, "delete_user", f"Deleted user: {username}")
    flash(f"User '{username}' deleted.", "info")
    return redirect(url_for("admin.users"))


# ------------------------------------------------------------- Students ----
def _maybe_create_login(student: Student, form) -> str | None:
    """Optionally creates/links a student-role User account from the registration
    or edit form. Returns an error message string if it couldn't be created, else None."""
    if not form.get("create_login"):
        return None
    if student.user_id:
        return None  # already has an account

    username = form.get("login_username", "").strip()
    password = form.get("login_password", "").strip()
    if not username or not password:
        return "username and password are required to create a login."
    if len(password) < 6:
        return "password must be at least 6 characters."
    if User.query.filter((User.username == username) | (User.email == (student.email or ""))).first():
        return f"username '{username}' (or the student's email) is already in use."

    login_account = User(
        full_name=student.full_name,
        username=username,
        email=student.email or f"{username}@spas.edu",
        role=ROLE_STUDENT,
    )
    login_account.set_password(password)
    db.session.add(login_account)
    db.session.flush()
    student.user_id = login_account.id
    return None


@admin_bp.route("/students")
def students():
    q = request.args.get("q", "").strip()
    support_filter = request.args.get("support", "")
    query = Student.query
    if q:
        like = f"%{q}%"
        query = query.filter(db.or_(Student.full_name.ilike(like), Student.student_code.ilike(like)))
    if support_filter in ("Low", "Medium", "High"):
        query = query.filter_by(parental_support=support_filter)
    all_students = query.order_by(Student.created_at.desc()).all()

    imported_count = Student.query.filter(
        Student.student_code.like("STU%"),
        Student.user_id.is_(None),
    ).count()

    return render_template(
        "admin/students.html", students=all_students, q=q, support_filter=support_filter,
        imported_count=imported_count,
    )


@admin_bp.route("/students/create", methods=["GET", "POST"])
def create_student():
    if request.method == "POST":
        next_id = (Student.query.count() or 0) + 1
        code = generate_student_code(next_id)
        while Student.query.filter_by(student_code=code).first():
            next_id += 1
            code = generate_student_code(next_id)

        raw, validation_errors = parse_academic_inputs(request.form)
        full_name = request.form.get("full_name", "").strip()
        if not full_name:
            validation_errors.append("Full name is required.")
        if validation_errors:
            for error in validation_errors:
                flash(error, "danger")
            return render_template("admin/student_form.html", student=None)

        student = Student(
            student_code=code,
            full_name=full_name,
            gender=raw["gender"],
            email=request.form.get("email", "").strip() or None,
            attendance=raw["attendance"],
            study_hours=raw["study_hours"],
            previous_grade=raw["previous_grade"],
            extracurricular=raw["extracurricular"],
            parental_support=raw["parental_support"],
            online_classes=raw["online_classes"],
            created_by_id=current_user.id,
        )
        db.session.add(student)
        db.session.flush()  # get student.id before optionally linking a login account

        login_error = _maybe_create_login(student, request.form)

        db.session.commit()
        ActivityLog.log(current_user.id, "create_student", f"Registered student {student.full_name}")
        if login_error:
            flash(
                f"Student '{student.full_name}' registered with code {student.student_code}, "
                f"but the login account was not created: {login_error}",
                "warning",
            )
        else:
            flash(f"Student '{student.full_name}' registered with code {student.student_code}.", "success")
        return redirect(url_for("admin.students"))

    return render_template("admin/student_form.html", student=None)


@admin_bp.route("/students/<int:student_id>/edit", methods=["GET", "POST"])
def edit_student(student_id):
    student = Student.query.get_or_404(student_id)
    if request.method == "POST":
        raw, validation_errors = parse_academic_inputs(request.form, {
            "attendance": student.attendance, "study_hours": student.study_hours,
            "previous_grade": student.previous_grade, "extracurricular": student.extracurricular,
            "gender": student.gender, "parental_support": student.parental_support,
        })
        full_name = request.form.get("full_name", student.full_name).strip()
        if not full_name:
            validation_errors.append("Full name is required.")
        if validation_errors:
            for error in validation_errors:
                flash(error, "danger")
            return render_template("admin/student_form.html", student=student)

        student.full_name = full_name
        student.gender = raw["gender"]
        student.email = request.form.get("email", "").strip() or None
        student.attendance = raw["attendance"]
        student.study_hours = raw["study_hours"]
        student.previous_grade = raw["previous_grade"]
        student.extracurricular = raw["extracurricular"]
        student.parental_support = raw["parental_support"]
        student.online_classes = raw["online_classes"]

        login_error = _maybe_create_login(student, request.form)

        db.session.commit()
        ActivityLog.log(current_user.id, "edit_student", f"Updated student {student.full_name}")
        if login_error:
            flash(f"Student updated, but the login account was not created: {login_error}", "warning")
        else:
            flash("Student updated.", "success")
        return redirect(url_for("admin.students"))

    return render_template("admin/student_form.html", student=student)


@admin_bp.route("/students/<int:student_id>")
def student_detail(student_id):
    student = Student.query.get_or_404(student_id)
    history = student.predictions.limit(20).all()
    return render_template("admin/student_detail.html", student=student, history=history)


@admin_bp.route("/students/<int:student_id>/delete", methods=["POST"])
def delete_student(student_id):
    student = Student.query.get_or_404(student_id)
    name = student.full_name
    db.session.delete(student)
    db.session.commit()
    ActivityLog.log(current_user.id, "delete_student", f"Deleted student: {name}")
    flash(f"Student '{name}' deleted.", "info")
    return redirect(url_for("admin.students"))


@admin_bp.route("/students/clear-imported", methods=["POST"])
def clear_imported_students():
    """Bulk-deletes the demo students that seed.py auto-imports from the
    dataset (student codes STU00001-STU00950 with no real linked login).
    Any of those that a real student has since claimed via signup (i.e.
    has a linked user account) are deliberately left alone, so this can't
    accidentally delete someone's real, in-use account."""
    # Dataset imports use the exact generated name pattern "Student NNN".
    # Do not delete arbitrary admin-created STUxxxxx records that merely lack
    # a linked login. Linked records are still protected as an extra guard.
    imported = Student.query.filter(
        Student.student_code.like("STU%"),
        Student.full_name.like("Student %"),
        Student.user_id.is_(None),
    ).all()
    count = len(imported)
    for s in imported:
        db.session.delete(s)
    db.session.commit()
    ActivityLog.log(current_user.id, "clear_imported_students", f"Deleted {count} unclaimed imported students")
    flash(f"Deleted {count} imported dataset student(s). Any linked to a real login were kept.", "success")
    return redirect(url_for("admin.students"))


@admin_bp.route("/students/<int:student_id>/report")
def student_report(student_id):
    student = Student.query.get_or_404(student_id)
    prediction = student.latest_prediction()
    filename = f"student_{student.student_code}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(current_app.config["EXPORTS_DIR"], filename)
    report_service.generate_student_report(student, prediction, output_path, current_user.full_name)

    report = Report(
        student_id=student.id,
        generated_by_id=current_user.id,
        report_type="student",
        file_path=output_path,
        file_name=filename,
    )
    db.session.add(report)
    db.session.commit()
    ActivityLog.log(current_user.id, "generate_report", f"Generated report for {student.full_name}")
    return send_file(output_path, as_attachment=True, download_name=filename)


# ----------------------------------------------------------- Prediction ---
@admin_bp.route("/predict", methods=["GET", "POST"])
def predict():
    all_students = Student.query.order_by(Student.full_name).all()
    result = None
    raw = None
    selected_student_id = request.values.get("student_id", type=int)

    if request.method == "POST":
        raw, validation_errors = parse_academic_inputs(request.form)
        if validation_errors:
            for error in validation_errors:
                flash(error, "danger")
            return render_template(
                "admin/predict.html", students=all_students, result=None,
                selected_student_id=selected_student_id, raw=raw, metrics=ml_service.get_metrics()
            )
        try:
            result = ml_service.predict_full(raw)
        except RuntimeError as exc:
            flash(str(exc), "danger")
            return render_template(
                "admin/predict.html", students=all_students, result=None, selected_student_id=selected_student_id
            )

        student = None
        if selected_student_id:
            student = Student.query.filter_by(id=selected_student_id).first()

        prediction = Prediction(
            student_id=student.id if student else None,
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
        if student:
            student.current_grade = result["predicted_grade"]
        db.session.add(prediction)
        db.session.commit()
        ActivityLog.log(current_user.id, "predict", f"Ran prediction (student_id={selected_student_id})")

    return render_template(
        "admin/predict.html", students=all_students, result=result, raw=raw if result else None,
        selected_student_id=selected_student_id, metrics=ml_service.get_metrics(),
    )


# ---------------------------------------------------------- Predictions ----
@admin_bp.route("/predictions")
def predictions():
    q = request.args.get("q", "").strip()
    level_filter = request.args.get("level", "")

    query = Prediction.query
    if q:
        like = f"%{q}%"
        query = query.join(Student, isouter=True).filter(
            db.or_(Student.full_name.ilike(like), Student.student_code.ilike(like))
        )
    if level_filter:
        query = query.filter(Prediction.performance_level == level_filter)

    all_predictions = query.order_by(Prediction.created_at.desc()).limit(200).all()

    pass_threshold = ml_service.current_pass_threshold()
    stale_count = Prediction.query.filter(
        db.or_(
            db.and_(Prediction.predicted_grade >= pass_threshold, Prediction.pass_fail == "Fail"),
            db.and_(Prediction.predicted_grade < pass_threshold, Prediction.pass_fail == "Pass"),
        )
    ).count()

    return render_template(
        "admin/predictions.html", predictions=all_predictions, q=q, level_filter=level_filter,
        stale_count=stale_count,
    )


@admin_bp.route("/predictions/recalculate-results", methods=["POST"])
def recalculate_results():
    """One-time fix for predictions stored BEFORE Result was tied to
    Predicted Grade -- a code fix alone doesn't touch already-saved rows,
    so this recomputes pass_fail/performance_level/risk_level for every
    existing Prediction using the current (correct) grade-based logic."""
    pass_threshold = ml_service.current_pass_threshold()

    all_preds = Prediction.query.all()
    updated = 0
    for p in all_preds:
        correct_pass_fail = "Pass" if p.predicted_grade >= pass_threshold else "Fail"
        new_level = ml_service.performance_level_for(p.predicted_grade, correct_pass_fail)
        if p.pass_fail != correct_pass_fail or p.performance_level != new_level:
            p.pass_fail = correct_pass_fail
            p.performance_level = new_level
            updated += 1
    db.session.commit()
    ActivityLog.log(current_user.id, "recalculate_results", f"Recalculated {updated} stale prediction result(s)")
    flash(f"Recalculated {updated} prediction(s) whose Result/Level didn't match their Predicted Grade.", "success")
    return redirect(url_for("admin.predictions"))


@admin_bp.route("/predictions/<int:prediction_id>/delete", methods=["POST"])
def delete_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    db.session.delete(prediction)
    db.session.commit()
    ActivityLog.log(current_user.id, "delete_prediction", f"Deleted prediction #{prediction_id}")
    flash("Prediction deleted.", "info")
    return redirect(url_for("admin.predictions"))


# ------------------------------------------------------------ Analytics ---
@admin_bp.route("/analytics")
def analytics():
    all_students = Student.query.all()
    support_breakdown = {"Low": 0, "Medium": 0, "High": 0}
    for s in all_students:
        support_breakdown[s.parental_support] = support_breakdown.get(s.parental_support, 0) + 1

    pass_count = Prediction.query.filter_by(pass_fail="Pass").count()
    fail_count = Prediction.query.filter_by(pass_fail="Fail").count()

    attendance_vs_grade = [
        {"attendance": s.attendance, "grade": s.current_grade or s.previous_grade} for s in all_students
    ]

    return render_template(
        "admin/analytics.html",
        support_breakdown=support_breakdown,
        pass_count=pass_count,
        fail_count=fail_count,
        attendance_vs_grade=attendance_vs_grade,
        total_students=len(all_students),
    )


# --------------------------------------------------------------- Logs -----
@admin_bp.route("/logs")
def logs():
    activity = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(300).all()
    logins = LoginHistory.query.order_by(LoginHistory.timestamp.desc()).limit(300).all()
    return render_template("admin/logs.html", activity=activity, logins=logins)


# --------------------------------------------------------------- Reports --
@admin_bp.route("/reports")
def reports():
    all_reports = Report.query.order_by(Report.created_at.desc()).all()
    return render_template("admin/reports.html", reports=all_reports)


@admin_bp.route("/reports/generate", methods=["POST"])
def generate_report():
    total_students = Student.query.count()
    total_predictions = Prediction.query.count()
    avg_grade_row = db.session.query(db.func.avg(Prediction.predicted_grade)).scalar()
    metrics = ml_service.get_metrics()

    stats = {
        "total_students": total_students,
        "total_predictions": total_predictions,
        "average_grade": round(avg_grade_row, 2) if avg_grade_row else "-",
        "best_regression_model": metrics.get("best_regression_model", "-"),
        "best_classifier_model": metrics.get("best_classifier_model", "-"),
    }

    filename = f"admin_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(current_app.config["EXPORTS_DIR"], filename)
    report_service.generate_admin_report(stats, output_path)

    report = Report(
        student_id=None,
        generated_by_id=current_user.id,
        report_type="admin",
        file_path=output_path,
        file_name=filename,
    )
    db.session.add(report)
    db.session.commit()
    ActivityLog.log(current_user.id, "generate_report", "Generated system-wide admin report")
    flash("System report generated.", "success")
    return redirect(url_for("admin.reports"))


@admin_bp.route("/reports/all-students")
def all_students_report():
    all_students = Student.query.order_by(Student.full_name).all()
    pairs = [(s, s.latest_prediction()) for s in all_students]
    filename = f"all_students_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(current_app.config["EXPORTS_DIR"], filename)
    report_service.generate_all_students_report(pairs, current_user.full_name, output_path)

    report = Report(
        student_id=None,
        generated_by_id=current_user.id,
        report_type="admin",
        file_path=output_path,
        file_name=filename,
    )
    db.session.add(report)
    db.session.commit()
    ActivityLog.log(current_user.id, "generate_report", "Generated all-students report")
    return send_file(output_path, as_attachment=True, download_name=filename)


@admin_bp.route("/reports/<int:report_id>/download")
def download_report(report_id):
    report = Report.query.get_or_404(report_id)
    if not os.path.exists(report.file_path):
        flash("That report file is no longer available on disk.", "danger")
        return redirect(url_for("admin.reports"))
    return send_file(report.file_path, as_attachment=True, download_name=report.file_name)


# -------------------------------------------------------------- Dataset ---
ALLOWED_DATASET_EXTENSIONS = {".xlsx", ".xls"}


@admin_bp.route("/dataset")
def dataset():
    dataset_path = current_app.config["DATASET_PATH"]
    exists = os.path.exists(dataset_path)
    size_kb = round(os.path.getsize(dataset_path) / 1024, 1) if exists else 0
    modified_at = (
        datetime.fromtimestamp(os.path.getmtime(dataset_path)).strftime("%Y-%m-%d %H:%M")
        if exists
        else None
    )

    row_count = None
    if exists:
        try:
            import pandas as pd

            row_count = len(pd.read_excel(dataset_path))
        except Exception:
            row_count = None

    return render_template(
        "admin/dataset.html",
        exists=exists,
        size_kb=size_kb,
        dataset_path=dataset_path,
        modified_at=modified_at,
        row_count=row_count,
    )


@admin_bp.route("/dataset/upload", methods=["POST"])
def upload_dataset():
    file = request.files.get("dataset_file")
    if not file or file.filename == "":
        flash("Please choose a .xlsx file to upload.", "danger")
        return redirect(url_for("admin.dataset"))

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_DATASET_EXTENSIONS:
        flash("Only .xlsx or .xls files are accepted.", "danger")
        return redirect(url_for("admin.dataset"))

    dataset_path = current_app.config["DATASET_PATH"]
    dataset_dir = os.path.dirname(dataset_path)
    os.makedirs(dataset_dir, exist_ok=True)

    # Back up the current dataset before overwriting it
    if os.path.exists(dataset_path):
        backup_name = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(dataset_path)}"
        shutil.copy2(dataset_path, os.path.join(dataset_dir, backup_name))

    safe_name = secure_filename(file.filename)
    upload_tmp_path = os.path.join(dataset_dir, f"upload_{safe_name}")
    file.save(upload_tmp_path)

    # Validate it's actually a readable spreadsheet before committing to it
    try:
        import pandas as pd

        df_check = pd.read_excel(upload_tmp_path)
        if "FinalGrade" not in df_check.columns:
            raise ValueError("Uploaded file is missing the required 'FinalGrade' column.")
        # Validate against the same cleaning/feature contract used by
        # train_model.py, not just the target column. This prevents an upload
        # that looks valid in the UI from breaking the next retrain.
        from train_model import clean_data
        cleaned = clean_data(df_check)
        required = {"Gender", "ParentalSupport", "PreviousGrade", "ExtracurricularActivities", "FinalGrade"}
        if not {c for c in required if c in cleaned.columns} == required:
            missing = sorted(required - set(cleaned.columns))
            raise ValueError("Uploaded file is missing required columns after cleaning: " + ", ".join(missing))
        if not ("Attendance_Final" in cleaned.columns or "AttendanceRate" in cleaned.columns or "Attendance (%)" in df_check.columns):
            raise ValueError("Uploaded file is missing an attendance column.")
        if not ("StudyHours_Final" in cleaned.columns or "StudyHoursPerWeek" in cleaned.columns or "Study Hours" in df_check.columns):
            raise ValueError("Uploaded file is missing a study-hours column.")
        if cleaned.empty:
            raise ValueError("Uploaded dataset contains no usable rows after cleaning.")
    except Exception as exc:
        os.remove(upload_tmp_path)
        flash(f"Upload rejected: {exc}", "danger")
        return redirect(url_for("admin.dataset"))

    shutil.move(upload_tmp_path, dataset_path)
    ActivityLog.log(current_user.id, "upload_dataset", f"Replaced dataset with {file.filename} ({len(df_check)} rows)")
    flash(
        f"Dataset replaced ({len(df_check)} rows). The previous file was backed up. "
        "Click 'Retrain Model Now' on the ML Model page to train on this new data.",
        "success",
    )
    return redirect(url_for("admin.dataset"))


@admin_bp.route("/dataset/import-demo-students", methods=["POST"])
def import_demo_students():
    """Manually, deliberately imports the current dataset file as Student
    records. This is the ONLY way the bulk dataset ever becomes students --
    it never happens automatically, so it never fights an admin's decision
    to clear that data out."""
    from seed import import_dataset

    count = import_dataset(current_user)
    ActivityLog.log(current_user.id, "import_demo_students", f"Imported {count} students from dataset")
    if count:
        flash(f"Imported {count} student(s) from the dataset.", "success")
    else:
        flash("No new students imported (dataset already fully imported, or file not found).", "info")
    return redirect(url_for("admin.students"))


# ------------------------------------------------------------ ML Model ----
@admin_bp.route("/model")
def model_info():
    metrics = ml_service.get_metrics()
    feature_names = ml_service.get_feature_names()
    settings = AppSetting.get()
    return render_template(
        "admin/model.html",
        metrics=metrics,
        feature_names=feature_names,
        model_ready=ml_service.is_ready(),
        settings=settings,
    )


@admin_bp.route("/model/retrain", methods=["POST"])
def retrain_model():
    settings = AppSetting.get()
    backend_dir = os.path.dirname(current_app.config["MODELS_DIR"])
    train_script = os.path.join(backend_dir, "train_model.py")

    result = subprocess.run(
        [sys.executable, train_script, "--pass-threshold", str(settings.pass_threshold)],
        capture_output=True,
        text=True,
        cwd=backend_dir,
        timeout=600,
    )

    log_tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])

    if result.returncode != 0:
        ActivityLog.log(current_user.id, "retrain_model", "Retrain FAILED")
        flash(f"Retraining failed. Last output:\n{log_tail}", "danger")
        return redirect(url_for("admin.model_info"))

    # Reload the freshly-trained pipelines into the running app without a restart
    ml_service.init_app(current_app._get_current_object())

    ActivityLog.log(current_user.id, "retrain_model", f"Retrained with pass_threshold={settings.pass_threshold}")
    flash("Model retrained successfully and reloaded — new predictions now use the updated pipelines.", "success")
    return redirect(url_for("admin.model_info"))


# -------------------------------------------------------------- Settings --
@admin_bp.route("/settings", methods=["GET", "POST"])
def settings():
    app_settings = AppSetting.get()

    if request.method == "POST":
        app_settings.app_name = request.form.get("app_name", app_settings.app_name).strip() or app_settings.app_name
        try:
            new_threshold = float(request.form.get("pass_threshold", app_settings.pass_threshold))
            if 0 <= new_threshold <= 100:
                app_settings.pass_threshold = new_threshold
        except ValueError:
            pass
        try:
            new_timeout = int(request.form.get("session_timeout_minutes", app_settings.session_timeout_minutes))
            if new_timeout > 0:
                app_settings.session_timeout_minutes = new_timeout
        except ValueError:
            pass
        app_settings.maintenance_mode = bool(request.form.get("maintenance_mode"))

        db.session.commit()
        ActivityLog.log(current_user.id, "update_settings", "Updated system settings")
        flash(
            "Settings saved. Note: the Pass Threshold only affects the NEXT model retrain, "
            "not models already trained.",
            "success",
        )
        return redirect(url_for("admin.settings"))

    return render_template("admin/settings.html", settings=app_settings)
