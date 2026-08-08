from datetime import datetime

from app.extensions import db


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=True)
    created_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # Input snapshot (so history is reproducible even if the student record changes later)
    attendance = db.Column(db.Float)
    study_hours = db.Column(db.Float)
    previous_grade = db.Column(db.Float)
    extracurricular = db.Column(db.Float)
    gender = db.Column(db.String(10))
    parental_support = db.Column(db.String(10))
    online_classes = db.Column(db.Boolean)

    # Output
    predicted_grade = db.Column(db.Float, nullable=False)
    pass_fail = db.Column(db.String(4), nullable=False)  # "Pass" / "Fail"
    pass_probability = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    performance_level = db.Column(db.String(20), nullable=False)

    regression_model_name = db.Column(db.String(80))
    classifier_model_name = db.Column(db.String(80))

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "predicted_grade": self.predicted_grade,
            "pass_fail": self.pass_fail,
            "pass_probability": self.pass_probability,
            "confidence": self.confidence,
            "performance_level": self.performance_level,
            "created_at": self.created_at.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        }


class Report(db.Model):
    __tablename__ = "reports"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=True)
    generated_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    report_type = db.Column(db.String(30), nullable=False)  # student / admin
    file_path = db.Column(db.String(255), nullable=False)
    file_name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    generated_by = db.relationship("User", foreign_keys=[generated_by_id])


class LoginHistory(db.Model):
    __tablename__ = "login_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    username_attempted = db.Column(db.String(80))
    ip_address = db.Column(db.String(64))
    user_agent = db.Column(db.String(255))
    success = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)


class ActivityLog(db.Model):
    __tablename__ = "activity_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    action = db.Column(db.String(80), nullable=False)
    details = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    @staticmethod
    def log(user_id, action, details=""):
        entry = ActivityLog(user_id=user_id, action=action, details=details)
        db.session.add(entry)
        db.session.commit()
        return entry


class AppSetting(db.Model):
    """Singleton row (id is always 1) holding admin-editable system settings."""

    __tablename__ = "app_settings"

    id = db.Column(db.Integer, primary_key=True)
    app_name = db.Column(db.String(120), nullable=False, default="EduPredict")
    pass_threshold = db.Column(db.Float, nullable=False, default=70.0)
    session_timeout_minutes = db.Column(db.Integer, nullable=False, default=480)
    maintenance_mode = db.Column(db.Boolean, nullable=False, default=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @staticmethod
    def get():
        settings = AppSetting.query.get(1)
        if settings is None:
            settings = AppSetting(id=1)
            db.session.add(settings)
            db.session.commit()
        return settings
