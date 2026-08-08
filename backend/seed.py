"""
seed.py
=======
Two separate things happen here, deliberately kept apart:

1. seed_app(app) -- ESSENTIAL accounts only (admin + one demo student).
   This is safe to run on EVERY process boot (run.py does exactly that),
   including Render's free tier where the whole database gets wiped on
   every restart/sleep-wake cycle. It never imports the bulk dataset, so
   an admin's decision to clear demo data actually sticks across restarts
   instead of being silently undone every time the service sleeps.

2. import_dataset(admin) -- the heavy ~950-row dataset import. This is
   ONLY ever triggered deliberately: by running `python seed.py` directly
   (for local/fresh setup), or by an admin clicking "Import Dataset as
   Demo Students" in the UI. It never runs automatically on boot.

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


def create_demo_student_account(admin: User):
    """Creates exactly ONE lightweight demo Student + linked student1 login,
    with hardcoded sample data -- not dependent on the dataset import, so
    this always works even when the bulk dataset has never been imported
    (or was cleared by an admin)."""
    if User.query.filter_by(username="student1").first():
        return

    demo_student = Student.query.filter_by(student_code="STU00001").first()
    if demo_student is None:
        demo_student = Student(
            student_code="STU00001",
            full_name="Student 001",
            gender="Male",
            attendance=85.0,
            study_hours=15.0,
            previous_grade=78.0,
            extracurricular=1.0,
            parental_support="High",
            online_classes=False,
            current_grade=80.0,
            created_by_id=admin.id,
        )
        db.session.add(demo_student)
        db.session.flush()

    student_user = User(
        full_name=demo_student.full_name,
        username="student1",
        email="student1@spas.edu",
        role=ROLE_STUDENT,
    )
    student_user.set_password("Student@123")
    db.session.add(student_user)
    db.session.flush()

    demo_student.user_id = student_user.id
    db.session.commit()
    log(f"Created student1 / Student@123, linked to {demo_student.student_code}")


def import_dataset(admin: User) -> int:
    """Bulk-imports the production dataset as Student records. NEVER called
    automatically on boot -- only via explicit CLI run or an admin's
    deliberate button click. Returns the number of rows imported."""
    from flask import current_app

    dataset_path = current_app.config["DATASET_PATH"]
    if not os.path.exists(dataset_path):
        log(f"Dataset not found at {dataset_path} - skipping import")
        return 0

    df = pd.read_excel(dataset_path)
    df_clean = clean_data(df).reset_index(drop=True)
    if IMPORT_ROW_LIMIT is not None:
        df_clean = df_clean.head(IMPORT_ROW_LIMIT)

    existing_codes = {c for (c,) in db.session.query(Student.student_code).all()}
    imported = 0
    for i, row in df_clean.iterrows():
        code = f"STU{i + 1:05d}"
        if code in existing_codes:
            continue  # don't duplicate a row that's already there (e.g. the demo STU00001)
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
        imported += 1

    db.session.commit()
    log(f"Imported {imported} students from dataset")
    return imported


def seed_app(app):
    """ESSENTIAL accounts only -- admin + one demo student. Safe (and
    intended) to run on every single process boot, including Render's
    free-tier restarts, without ever re-adding the bulk dataset."""
    with app.app_context():
        db.create_all()
        log("Database tables verified/created")
        admin = create_default_admin()
        create_demo_student_account(admin)
        log("Seed check complete.")


def main():
    """CLI entry point for a deliberate local/fresh setup: essential
    accounts PLUS the full dataset import, since running this by hand
    implies you actually want the demo data populated."""
    app = create_app()
    seed_app(app)
    with app.app_context():
        admin = User.query.filter_by(username="admin").first()
        import_dataset(admin)
    log("Login credentials:")
    log("  Admin:   admin / Admin@123")
    log("  Student: student1 / Student@123")


if __name__ == "__main__":
    main()
