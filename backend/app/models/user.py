from datetime import datetime

from flask_login import UserMixin

from app.extensions import bcrypt, db

ROLE_ADMIN = "admin"
ROLE_STUDENT = "student"
ROLES = (ROLE_ADMIN, ROLE_STUDENT)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=ROLE_STUDENT)
    is_active_flag = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login_at = db.Column(db.DateTime, nullable=True)

    student_profile = db.relationship(
        "Student", backref="account", uselist=False, foreign_keys="Student.user_id"
    )
    predictions_made = db.relationship("Prediction", backref="created_by", lazy="dynamic")
    login_history = db.relationship("LoginHistory", backref="user", lazy="dynamic")
    activity_logs = db.relationship("ActivityLog", backref="user", lazy="dynamic")

    # -- password helpers --
    def set_password(self, raw_password: str):
        self.password_hash = bcrypt.generate_password_hash(raw_password).decode("utf-8")

    def check_password(self, raw_password: str) -> bool:
        return bcrypt.check_password_hash(self.password_hash, raw_password)

    # flask-login override: use is_active_flag column instead of shadowing it
    @property
    def is_active(self):
        return self.is_active_flag

    def is_admin(self):
        return self.role == ROLE_ADMIN

    def is_student(self):
        return self.role == ROLE_STUDENT

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"
