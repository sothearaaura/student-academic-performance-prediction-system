"""
Smoke tests covering auth, role-based access control, and the prediction flow.
Run from backend/:  pytest ../tests/test_smoke.py
(Requires `python train_model.py` and `python seed.py` to have been run first.)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest  # noqa: E402

from app import create_app  # noqa: E402


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def login(client, username, password):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=True)


def test_login_success(client):
    resp = login(client, "admin", "Admin@123")
    assert resp.status_code == 200


def test_login_failure(client):
    resp = login(client, "admin", "wrong-password")
    assert b"Invalid username" in resp.data


def test_admin_dashboard_accessible(client):
    login(client, "admin", "Admin@123")
    resp = client.get("/admin/dashboard")
    assert resp.status_code == 200


def test_student_cannot_access_admin(client):
    login(client, "student1", "Student@123")
    resp = client.get("/admin/dashboard")
    assert resp.status_code == 403


def test_admin_can_manage_students(client):
    login(client, "admin", "Admin@123")
    for path in ["/admin/students", "/admin/students/create", "/admin/predict", "/admin/analytics"]:
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} returned {resp.status_code}"


def test_admin_predict_flow(client):
    login(client, "admin", "Admin@123")
    resp = client.post(
        "/admin/predict",
        data={
            "attendance": "85",
            "study_hours": "15",
            "previous_grade": "75",
            "extracurricular": "1",
            "gender": "Male",
            "parental_support": "Medium",
        },
    )
    assert resp.status_code == 200


def test_admin_can_register_student_with_login(client):
    import uuid

    unique_suffix = uuid.uuid4().hex[:8]
    username = f"regtest_{unique_suffix}"

    login(client, "admin", "Admin@123")
    resp = client.post(
        "/admin/students/create",
        data={
            "full_name": "Regression Test Student",
            "gender": "Female",
            "attendance": "90",
            "study_hours": "12",
            "previous_grade": "80",
            "extracurricular": "1",
            "parental_support": "High",
            "create_login": "on",
            "login_username": username,
            "login_password": "Pass123",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"was not created" not in resp.data

    logout = client.get("/logout")
    assert logout.status_code == 302

    login_resp = login(client, username, "Pass123")
    assert login_resp.status_code == 200
    dash = client.get("/student/dashboard")
    assert b"not yet linked" not in dash.data


def test_student_can_download_report(client):
    login(client, "student1", "Student@123")
    resp = client.get("/student/report")
    assert resp.status_code == 200
    assert resp.content_type == "application/pdf"


def test_student_dashboard_shows_empty_state_until_real_prediction_made(client):
    """Regression test: the dashboard must NOT silently fabricate a prediction
    for a student who hasn't actually made one. A brand-new student should
    see an empty state; only after they submit New Prediction (or an admin
    predicts for them) should the dashboard show real numbers."""
    from app.extensions import db
    from app.models import Prediction, Student, User, ROLE_STUDENT

    with client.application.app_context():
        # Create a brand-new student + login with zero prediction history
        u = User(full_name="Fresh Student", username="freshdashtest", email="freshdashtest@example.com", role=ROLE_STUDENT)
        u.set_password("Pass123")
        db.session.add(u)
        db.session.flush()
        s = Student(student_code="STUFRESH1", full_name="Fresh Student", user_id=u.id)
        db.session.add(s)
        db.session.commit()
        baseline_count = Prediction.query.count()

    login(client, "freshdashtest", "Pass123")
    resp = client.get("/student/dashboard")
    assert resp.status_code == 200
    text = resp.data.decode()
    assert "No predictions yet." in text
    assert "haven't made a prediction yet" in text
    # Stat cards should show the em-dash placeholder, not a fabricated number
    assert 'stat-value">\u2014' in text

    with client.application.app_context():
        after_visit_count = Prediction.query.count()
    assert after_visit_count == baseline_count, "visiting the dashboard must never create a prediction on its own"

    # Now the student actually makes a prediction -- dashboard should reflect it
    client.post(
        "/student/new-prediction",
        data={"attendance": "85", "study_hours": "15", "previous_grade": "75", "extracurricular": "1", "gender": "Male", "parental_support": "Medium"},
    )
    resp = client.get("/student/dashboard")
    text = resp.data.decode()
    assert "No predictions yet." not in text
    assert "Risk Level" in text


def test_signup_claims_existing_student_record(client):
    """New user role feature: sign up for new user, login for old user.
    Claiming an existing admin-entered academic record via Student Code."""
    from app.extensions import db
    from app.models import Student

    with client.application.app_context():
        unclaimed = Student(student_code="STUCLAIMTEST", full_name="Unclaimed Record")
        db.session.add(unclaimed)
        db.session.commit()

    resp = client.post(
        "/signup",
        data={
            "full_name": "New Claimant",
            "username": "claimanttest",
            "email": "claimanttest@example.com",
            "password": "Pass123",
            "confirm_password": "Pass123",
            "student_code": "STUCLAIMTEST",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Account created" in resp.data

    login_resp = login(client, "claimanttest", "Pass123")
    assert login_resp.status_code == 200
    dash = client.get("/student/dashboard")
    assert b"not yet linked" not in dash.data


def test_signup_fresh_registration_without_code(client):
    resp = client.post(
        "/signup",
        data={
            "full_name": "Brand New Student",
            "username": "freshsignuptest",
            "email": "freshsignuptest@example.com",
            "password": "Pass123",
            "confirm_password": "Pass123",
            "student_code": "",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Account created" in resp.data

    login_resp = login(client, "freshsignuptest", "Pass123")
    assert login_resp.status_code == 200
    dash = client.get("/student/dashboard")
    assert dash.status_code == 200


def test_signup_rejects_mismatched_passwords(client):
    resp = client.post(
        "/signup",
        data={
            "full_name": "Mismatch",
            "username": "mismatchsignuptest",
            "email": "mismatchsignuptest@example.com",
            "password": "Pass123",
            "confirm_password": "Different123",
            "student_code": "",
        },
        follow_redirects=True,
    )
    assert b"do not match" in resp.data


def test_timestamps_are_utc_marked_for_client_side_localization(client):
    """Regression test for a reported bug: timestamps were displayed in raw
    server (UTC) time instead of the viewer's real local time. Every
    timestamp must carry a data-utc="...Z" attribute so app.js can convert
    it client-side to whatever timezone the actual visitor is in."""
    login(client, "student1", "Student@123")
    client.post(
        "/student/new-prediction",
        data={"attendance": "85", "study_hours": "15", "previous_grade": "75", "extracurricular": "1", "gender": "Male", "parental_support": "Medium"},
    )  # ensures at least one prediction exists
    resp = client.get("/student/history")
    text = resp.data.decode()
    assert 'class="local-time"' in text
    assert 'data-utc="' in text
    # The UTC marker must be a real, parseable ISO timestamp ending in Z
    import re
    from datetime import datetime

    m = re.search(r'data-utc="([^"]+)"', text)
    assert m is not None
    datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))  # raises if invalid


def test_admin_can_clear_imported_students_but_keeps_linked_ones(client):
    """Regression test: the bulk-clear action must delete unclaimed dataset
    imports but must NEVER delete a student record that's linked to a real
    login (e.g. student1, or anyone who signed up with a Student Code)."""
    from app.models import Student

    login(client, "admin", "Admin@123")

    with client.application.app_context():
        before_total = Student.query.count()
        before_unlinked_stu = Student.query.filter(
            Student.student_code.like("STU%"), Student.user_id.is_(None)
        ).count()
        # student1's linked record must survive
        linked_code = Student.query.filter_by(student_code="STU00001").first().student_code
        assert linked_code == "STU00001"

    resp = client.post("/admin/students/clear-imported", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Deleted" in resp.data

    with client.application.app_context():
        after_total = Student.query.count()
        still_unlinked_stu = Student.query.filter(
            Student.student_code.like("STU%"), Student.user_id.is_(None)
        ).count()
        # The linked demo student must still exist
        still_linked = Student.query.filter_by(student_code="STU00001").first()
        assert still_linked is not None
        assert still_linked.user_id is not None

    assert still_unlinked_stu == 0, "all unclaimed imported students should be gone"
    assert after_total == before_total - before_unlinked_stu, "only the unclaimed imported students should have been removed"


def test_new_prediction_syncs_student_profile_for_admin_visibility(client):
    """Regression test for a reported bug: a student's prediction inputs were
    stored on the Prediction record but never synced back to their actual
    Student profile, so admin's Manage Students page kept showing stale/blank
    data even after the student had clearly reported real numbers."""
    from app.models import Student

    resp = client.post(
        "/signup",
        data={
            "full_name": "Profile Sync Test",
            "username": "profilesynctest",
            "email": "profilesynctest@example.com",
            "password": "Pass123",
            "confirm_password": "Pass123",
            "student_code": "",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200

    with client.application.app_context():
        before = Student.query.filter_by(full_name="Profile Sync Test").first()
        assert before.attendance == 0
        assert before.current_grade is None

    login(client, "profilesynctest", "Pass123")
    client.post(
        "/student/new-prediction",
        data={"attendance": "88", "study_hours": "14", "previous_grade": "76", "extracurricular": "2", "gender": "Female", "parental_support": "High"},
    )

    with client.application.app_context():
        after = Student.query.filter_by(full_name="Profile Sync Test").first()
        assert after.attendance == 88
        assert after.study_hours == 14
        assert after.previous_grade == 76
        assert after.current_grade is not None

    client.get("/logout")
    login(client, "admin", "Admin@123")
    resp = client.get("/admin/students")
    text = resp.data.decode()
    assert "88.0" in text
    assert "14.0" in text


def test_student_code_generation_has_no_collision_after_bulk_delete(client):
    """Regression test: after clearing most student records (e.g. via the
    admin bulk-clear-imported action), generating a new student code must
    never collide with a surviving high-numbered code."""
    from app.models import Student

    login(client, "admin", "Admin@123")
    client.post("/admin/students/clear-imported")

    resp = client.post(
        "/admin/students/create",
        data={"full_name": "Code Collision Check", "gender": "Male", "attendance": "80", "study_hours": "10", "previous_grade": "70", "extracurricular": "1", "parental_support": "Medium"},
        follow_redirects=True,
    )
    assert resp.status_code == 200

    with client.application.app_context():
        codes = [s.student_code for s in Student.query.all()]
        assert len(codes) == len(set(codes)), "student codes must never collide, even after bulk deletion"


def test_admin_can_update_settings(client):
    login(client, "admin", "Admin@123")
    resp = client.post(
        "/admin/settings",
        data={"app_name": "Test App Name", "pass_threshold": "72", "session_timeout_minutes": "60"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Settings saved" in resp.data


def test_maintenance_mode_blocks_non_admin_login(client):
    login(client, "admin", "Admin@123")
    client.post(
        "/admin/settings",
        data={
            "app_name": "Test App",
            "pass_threshold": "70",
            "session_timeout_minutes": "480",
            "maintenance_mode": "on",
        },
    )
    client.get("/logout")

    resp = login(client, "student1", "Student@123")
    assert b"under maintenance" in resp.data

    # Admin must still be able to log in
    resp = login(client, "admin", "Admin@123")
    assert b"Welcome back" in resp.data

    # Turn it back off so it doesn't affect other test runs
    client.post(
        "/admin/settings",
        data={"app_name": "Test App", "pass_threshold": "70", "session_timeout_minutes": "480"},
    )
