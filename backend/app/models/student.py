from datetime import datetime

from app.extensions import db


class Student(db.Model):
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    student_code = db.Column(db.String(30), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(120), nullable=False)
    gender = db.Column(db.String(10), nullable=False, default="Male")
    email = db.Column(db.String(120), nullable=True)

    attendance = db.Column(db.Float, nullable=False, default=0)
    study_hours = db.Column(db.Float, nullable=False, default=0)
    previous_grade = db.Column(db.Float, nullable=False, default=0)
    extracurricular = db.Column(db.Float, nullable=False, default=0)
    parental_support = db.Column(db.String(10), nullable=False, default="Medium")
    online_classes = db.Column(db.Boolean, nullable=False, default=False)

    current_grade = db.Column(db.Float, nullable=True)

    created_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    predictions = db.relationship(
        "Prediction", backref="student", lazy="dynamic",
        cascade="all, delete-orphan", order_by="desc(Prediction.created_at)",
    )
    reports = db.relationship("Report", backref="student", lazy="dynamic", cascade="all, delete-orphan")

    def to_feature_dict(self):
        return {
            "attendance": self.attendance,
            "study_hours": self.study_hours,
            "previous_grade": self.previous_grade,
            "extracurricular": self.extracurricular,
            "gender": self.gender,
            "parental_support": self.parental_support,
            "online_classes": int(self.online_classes),
        }

    def latest_prediction(self):
        return self.predictions.first()

    def __repr__(self):
        return f"<Student {self.student_code} {self.full_name}>"
