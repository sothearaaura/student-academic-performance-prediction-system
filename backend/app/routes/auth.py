import secrets
from datetime import datetime, timedelta
import os
import smtplib
from email.message import EmailMessage
from urllib.parse import urlparse

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.extensions import db
from app.models import (
    ROLE_ADMIN, ROLE_STUDENT, ActivityLog, AppSetting, LoginHistory,
    PasswordResetToken, Student, User,
)
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
    """Request a one-time, expiring password-reset link.

    The response is intentionally identical whether or not the email exists,
    preventing account enumeration. In production, SMTP must be configured so
    the reset link is delivered to the verified email address.
    """
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        generic = "If an account exists for that email, a password reset link has been sent."

        user = User.query.filter(db.func.lower(User.email) == email).first() if email else None
        if user and user.is_active_flag:
            smtp_host = os.environ.get("SMTP_HOST")
            smtp_port = int(os.environ.get("SMTP_PORT", "587"))
            smtp_user = os.environ.get("SMTP_USERNAME")
            smtp_password = os.environ.get("SMTP_PASSWORD")
            smtp_from = os.environ.get("SMTP_FROM") or smtp_user
            smtp_tls = os.environ.get("SMTP_TLS", "true").lower() == "true"

            if not all([smtp_host, smtp_user, smtp_password, smtp_from]):
                # Never create a reset token that cannot be delivered in production.
                if os.environ.get("RENDER") or os.environ.get("FLASK_ENV") == "production":
                    flash("Password reset is temporarily unavailable. Please contact an administrator.", "warning")
                    return render_template("auth/forgot_password.html")
                flash("Password reset email is not configured in this development environment.", "warning")
                return render_template("auth/forgot_password.html")

            raw_token, _ = PasswordResetToken.issue(user, minutes=30)
            db.session.commit()

            reset_url = url_for("auth.reset_password", token=raw_token, _external=True)
            msg = EmailMessage()
            msg["Subject"] = "EduPredict password reset"
            msg["From"] = smtp_from
            msg["To"] = user.email
            msg.set_content(
                f"Hello {user.full_name},\n\n"
                "We received a request to reset your EduPredict password.\n\n"
                f"Use this one-time link within 30 minutes:\n{reset_url}\n\n"
                "If you did not request this, you can safely ignore this email.\n"
                "The link can only be used once.\n\n"
                "EduPredict"
            )
            try:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as smtp:
                    if smtp_tls:
                        smtp.starttls()
                    smtp.login(smtp_user, smtp_password)
                    smtp.send_message(msg)
                ActivityLog.log(user.id, "password_reset_requested", "One-time password reset link sent")
            except Exception:
                db.session.rollback()
                current_app.logger.exception("Password reset email delivery failed")
                flash("Password reset is temporarily unavailable. Please try again later.", "warning")
                return render_template("auth/forgot_password.html")

        flash(generic, "info")
        return redirect(url_for("auth.login"))

    return render_template("auth/forgot_password.html")


@auth_bp.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    """Consume a single-use reset token and set a new password."""
    reset = PasswordResetToken.query.filter_by(
        token_hash=PasswordResetToken.hash_token(token), used_at=None
    ).first()

    if not reset or reset.expires_at <= datetime.utcnow() or not reset.user.is_active_flag:
        flash("This password reset link is invalid or has expired.", "danger")
        return redirect(url_for("auth.forgot_password"))

    if request.method == "POST":
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        if len(new_password) < 8:
            flash("New password must be at least 8 characters.", "danger")
            return render_template("auth/reset_password.html")
        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("auth/reset_password.html")

        # Mark the token used in the same transaction as the password change.
        reset.user.set_password(new_password)
        reset.used_at = datetime.utcnow()
        db.session.commit()
        ActivityLog.log(reset.user.id, "password_reset", "Password reset using one-time token")
        flash("Your password has been reset. You can now log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/reset_password.html")
