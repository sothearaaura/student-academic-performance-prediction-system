"""
seed.py
=======
Creates all database tables and populates the system with:
  - one admin account
  - one demo student account (linked to a Student record)
  - a batch of Student records imported from the production dataset,
    managed directly by the admin, so the dashboards have real data to show

Run once after `python train_model.py`:
    python seed.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models import ROLE_ADMIN, ROLE_STUDENT, Student, User  # noqa: E402
from train_model import clean_data  # noqa: E402

IMPORT_ROW_LIMIT = None  # None = import every row of the dataset


def log(msg):
    print(f"[seed] {msg}")


def create_default_admin():
    admin = User.query.filter_by(username="admin").first()
    if not admin:
        admin = User(
            full_name="System Administrator",
            username="admin",
            email="admin@spas.edu",
            role=ROLE_ADMIN,
        )
        admin.set_password("Admin@123")
        db.session.add(admin)
        db.session.commit()
        log("Created admin / Admin@123")
    return admin


def import_dataset(admin: User):
    if Student.query.count() > 0:
        log("Students table already populated - skipping dataset import")
        return

    from flask import current_app

    dataset_path = current_app.config["DATASET_PATH"]
    if not os.path.exists(dataset_path):
        log(f"Dataset not found at {dataset_path} - skipping import")
        return

    df = pd.read_excel(dataset_path)
    df_clean = clean_data(df).reset_index(drop=True)
    if IMPORT_ROW_LIMIT is not None:
        df_clean = df_clean.head(IMPORT_ROW_LIMIT)

    for i, row in df_clean.iterrows():
        code = f"STU{i + 1:05d}"
        student = Student(
            student_code=code,
            full_name=f"Student {i + 1:03d}",
            gender=row["Gender"],
            attendance=float(row["Attendance_Final"]) if pd.notna(row["Attendance_Final"]) else 0.0,
            study_hours=float(row["StudyHours_Final"]) if pd.notna(row["StudyHours_Final"]) else 0.0,
            previous_grade=float(row["PreviousGrade"]) if pd.notna(row["PreviousGrade"]) else 0.0,
            extracurricular=float(row["ExtracurricularActivities"]) if pd.notna(row["ExtracurricularActivities"]) else 0.0,
            parental_support=row["ParentalSupport"] if row["ParentalSupport"] in ("Low", "Medium", "High") else "Medium",
            online_classes=bool(row["Online_Classes"]),
            current_grade=float(row["FinalGrade"]) if pd.notna(row["FinalGrade"]) else None,
            created_by_id=admin.id,
        )
        db.session.add(student)

    db.session.commit()
    log(f"Imported {len(df_clean)} students from dataset")


def create_demo_student_login():
    first_student = Student.query.order_by(Student.id).first()
    if first_student is None:
        return
    if first_student.user_id is not None:
        return
    if User.query.filter_by(username="student1").first():
        return

    student_user = User(
        full_name=first_student.full_name,
        username="student1",
        email="student1@spas.edu",
        role=ROLE_STUDENT,
    )
    student_user.set_password("Student@123")
    db.session.add(student_user)
    db.session.flush()

    first_student.user_id = student_user.id
    db.session.commit()
    log(f"Created student1 / Student@123, linked to {first_student.student_code}")


def seed_app(app):
    """Runs the full idempotent seed routine against an already-created app.
    Safe to call on every process start (e.g. from run.py) since every step
    checks for existing data first -- it never duplicates or overwrites."""
    with app.app_context():
        db.create_all()
        log("Database tables verified/created")
        admin = create_default_admin()
        import_dataset(admin)
        create_demo_student_login()
        log("Seed check complete.")


def main():
    app = create_app()
    seed_app(app)
    log("Login credentials:")
    log("  Admin:   admin / Admin@123")
    log("  Student: student1 / Student@123")


if __name__ == "__main__":
    main()
