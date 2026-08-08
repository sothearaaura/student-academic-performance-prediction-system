from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app.extensions import db

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.is_admin():
            return redirect(url_for("admin.dashboard"))
        return redirect(url_for("student.dashboard"))
    return redirect(url_for("auth.login"))


@main_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        current_user.full_name = request.form.get("full_name", current_user.full_name).strip()
        current_user.email = request.form.get("email", current_user.email).strip()
        new_password = request.form.get("new_password", "").strip()
        if new_password:
            if len(new_password) < 6:
                flash("New password must be at least 6 characters.", "danger")
                return redirect(url_for("main.profile"))
            current_user.set_password(new_password)
        db.session.commit()
        flash("Profile updated successfully.", "success")
        return redirect(url_for("main.profile"))

    return render_template("shared/profile.html")
