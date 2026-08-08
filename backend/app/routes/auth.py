import secrets
from datetime import datetime, timedelta

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.extensions import db
from app.models import ROLE_ADMIN, ROLE_STUDENT, ActivityLog, AppSetting, LoginHistory, Student, User
from app.utils import generate_student_code

auth_bp = Blueprint("auth", __name__)


def _dashboard_redirect(user: User):
    if user.is_admin():
        return redirect(url_for("admin.dashboard"))
    return redirect(url_for("student.dashboard"))


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return _dashboard_redirect(current_user)

    settings = AppSetting.get()

    if request.method == "POST":
        identifier = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = User.query.filter(
            (User.username == identifier) | (User.email == identifier)
        ).first()

        success = bool(user and user.check_password(password) and user.is_active_flag)

        # Maintenance mode blocks everyone except admins from signing in
        if success and settings.maintenance_mode and user.role != ROLE_ADMIN:
            db.session.add(
                LoginHistory(
                    user_id=user.id, username_attempted=identifier,
                    ip_address=request.remote_addr, user_agent=request.headers.get("User-Agent", "")[:255],
                    success=False,
                )
            )
            db.session.commit()
            flash("The system is currently under maintenance. Please try again later.", "warning")
            return render_template("auth/login.html")

        db.session.add(
            LoginHistory(
                user_id=user.id if user else None,
                username_attempted=identifier,
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent", "")[:255],
                success=success,
            )
        )
        db.session.commit()

        if not success:
            if user and not user.is_active_flag:
                flash("This account has been deactivated. Contact an administrator.", "danger")
            else:
                flash("Invalid username/email or password.", "danger")
            return render_template("auth/login.html")

        login_user(user, remember=remember)
        session.permanent = True
        from flask import current_app

        current_app.permanent_session_lifetime = timedelta(minutes=settings.session_timeout_minutes)

        user.last_login_at = datetime.utcnow()
        db.session.commit()
        ActivityLog.log(user.id, "login", f"{user.role} logged in")
        flash(f"Welcome back, {user.full_name}!", "success")
        return _dashboard_redirect(user)

    return render_template("auth/login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    ActivityLog.log(current_user.id, "logout", f"{current_user.role} logged out")
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    """Self-service account creation for the Student role.

    Two paths, both handled by this one form:
      1. The student already has an academic record on file (entered by an
         admin) and knows their Student Code -> we link the new login to
         that existing record, so their real attendance/grades/etc. are
         used immediately.
      2. The student has no code yet -> leave it blank and a fresh, blank
         Student record is created alongside the account. An admin can fill
         in their academic data later from the Manage Students page.
    """
    if current_user.is_authenticated:
        return _dashboard_redirect(current_user)

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        student_code = request.form.get("student_code", "").strip().upper()

        errors = []
        if not full_name:
            errors.append("Full name is required.")
        if not username:
            errors.append("Username is required.")
        if not email:
            errors.append("Email is required.")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if User.query.filter((User.username == username) | (User.email == email)).first():
            errors.append(f"Username '{username}' or that email is already registered.")

        student_record = None
        if student_code:
            student_record = Student.query.filter_by(student_code=student_code).first()
            if student_record is None:
                errors.append(f"No student record found with code '{student_code}'. Leave blank to register fresh.")
            elif student_record.user_id is not None:
                errors.append(f"Student code '{student_code}' is already linked to another account.")

        if errors:
            for err in errors:
                flash(err, "danger")
            return render_template("auth/signup.html", form=request.form)

        new_user = User(full_name=full_name, username=username, email=email, role=ROLE_STUDENT)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.flush()

        if student_record:
            student_record.user_id = new_user.id
        else:
            next_id = (Student.query.count() or 0) + 1
            code = generate_student_code(next_id)
            while Student.query.filter_by(student_code=code).first():
                next_id += 1
                code = generate_student_code(next_id)
            student_record = Student(
                student_code=code,
                full_name=full_name,
                email=email,
                user_id=new_user.id,
            )
            db.session.add(student_record)

        db.session.commit()
        ActivityLog.log(new_user.id, "signup", f"New student account self-registered: {username}")
        flash("Account created! You can now log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/signup.html", form={})


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """Simplified self-service reset: verifies identity via username + email,
    then lets the user set a new password directly. In a production deployment
    this would instead email a signed, expiring token."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        new_password = request.form.get("new_password", "")

        user = User.query.filter_by(username=username, email=email).first()
        if not user:
            flash("No account matches that username and email combination.", "danger")
            return render_template("auth/forgot_password.html")

        if len(new_password) < 6:
            flash("New password must be at least 6 characters.", "danger")
            return render_template("auth/forgot_password.html")

        user.set_password(new_password)
        db.session.commit()
        ActivityLog.log(user.id, "password_reset", "Password reset via forgot-password form")
        flash("Password updated successfully. You can now log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/forgot_password.html")
