from functools import wraps

from flask import abort
from flask_login import current_user


def roles_required(*roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(401)
            if current_user.role not in roles:
                abort(403)
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


def generate_student_code(next_id: int) -> str:
    return f"STU{next_id:05d}"


def parse_academic_inputs(form, defaults=None):
    """Parse and validate prediction/student academic inputs server-side.

    Browser min/max attributes are only UX; clients can bypass them. Keep the
    same domain bounds enforced on every route that can create a prediction or
    student record so invalid values never reach the ML model/database.
    """
    defaults = defaults or {}
    errors = []

    def number(name, label, lo=None, hi=None, default=0.0):
        raw = form.get(name, defaults.get(name, default))
        try:
            value = float(raw if raw not in (None, "") else default)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a valid number.")
            return default
        if value != value or value in (float("inf"), float("-inf")):
            errors.append(f"{label} must be finite.")
            return default
        if lo is not None and value < lo:
            errors.append(f"{label} must be at least {lo:g}.")
        if hi is not None and value > hi:
            errors.append(f"{label} must be at most {hi:g}.")
        return value

    attendance = number("attendance", "Attendance", 0, 100, 85)
    study_hours = number("study_hours", "Study hours", 0, 168, 15)
    previous_grade = number("previous_grade", "Previous grade", 0, 100, 75)
    extracurricular = number("extracurricular", "Extracurricular activities", 0, 100, 0)

    gender = str(form.get("gender", defaults.get("gender", "Male"))).strip().title()
    if gender not in ("Male", "Female"):
        errors.append("Gender must be Male or Female.")
        gender = "Male"

    parental_support = str(form.get("parental_support", defaults.get("parental_support", "Medium"))).strip().title()
    if parental_support not in ("Low", "Medium", "High"):
        errors.append("Parental support must be Low, Medium, or High.")
        parental_support = "Medium"

    return ({
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_grade": previous_grade,
        "extracurricular": extracurricular,
        "gender": gender,
        "parental_support": parental_support,
        "online_classes": bool(form.get("online_classes")),
    }, errors)
