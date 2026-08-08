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


def test_student_dashboard_generates_realtime_prediction(client):
    """Matches the flow: Student -> Browse Predicted Grade -> System retrieves
    records -> runs Prediction Model -> generates and displays the grade,
    confidence, and risk level -- with zero admin action required first."""
    from app.models import Prediction

    with client.application.app_context():
        baseline_count = Prediction.query.count()

    login(client, "student1", "Student@123")
    resp = client.get("/student/dashboard")
    assert resp.status_code == 200
    assert b"Risk Level" in resp.data
    assert b"Confidence Score" in resp.data
    assert b"Suggestions for Improvement" in resp.data
    # No em-dash placeholder -- a real prediction was generated, not a "nothing yet" state
    assert 'stat-value">\u2014' not in resp.data.decode()

    with client.application.app_context():
        after_first_visit = Prediction.query.count()
    assert after_first_visit in (baseline_count, baseline_count + 1), (
        "first visit should store at most one new prediction "
        "(zero if student1 already had one from a prior test)"
    )

    # Repeat visits with unchanged data should not spam duplicate history entries
    client.get("/student/dashboard")
    client.get("/student/dashboard")
    with client.application.app_context():
        after_repeat_visits = Prediction.query.count()
    assert after_repeat_visits == after_first_visit, "unchanged data should not create duplicate predictions"


def test_signup_claims_existing_student_record(client):
    """New user role feature: sign up for new user, login for old user.
    Claiming an existing admin-entered academic record via Student Code."""
    resp = client.post(
        "/signup",
        data={
            "full_name": "New Claimant",
            "username": "claimanttest",
            "email": "claimanttest@example.com",
            "password": "Pass123",
            "confirm_password": "Pass123",
            "student_code": "STU00003",
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
    client.get("/student/dashboard")  # ensures at least one prediction exists
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
