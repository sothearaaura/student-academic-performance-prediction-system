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
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as c:
        yield c


def login(client, username, password):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=True)


def test_classifier_probability_uses_pass_class_label_not_array_position():
    import numpy as np
    from app.ml.pipeline import StudentPerformancePipeline

    class FakeModel:
        classes_ = np.array([1, 0])

        def predict(self, _):
            return np.array([0])

        def predict_proba(self, _):
            return np.array([[0.8, 0.2]])

    class Identity:
        def transform(self, value):
            return value

    pipeline = StudentPerformancePipeline(FakeModel(), Identity(), Identity(), "classification")
    result = pipeline.predict({"attendance": 80, "study_hours": 10, "previous_grade": 75})

    assert result["pass_fail"] == "Fail"
    assert result["pass_probability"] == 80.0
    assert result["confidence"] == 20.0
    assert result["classifier_classes"] == [1, 0]


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


def test_performance_level_never_conflicts_with_pass_fail_result(client):
    """Regression test for reported UX confusion: a positive-sounding Level
    (e.g. 'Very Good') must never be shown alongside a 'Fail' result, and
    'At Risk' must never be shown alongside a 'Pass' result, even though the
    grade regressor and the pass/fail classifier are trained independently
    and can genuinely disagree."""
    from app.services.ml_service import _performance_level

    assert _performance_level(80.3, "Fail") == "Needs Improvement"
    assert _performance_level(95.0, "Fail") == "Needs Improvement"
    assert _performance_level(45.0, "Pass") == "Needs Improvement"
    # Already-consistent cases must pass through unchanged
    assert _performance_level(95.0, "Pass") == "Excellent"
    assert _performance_level(45.0, "Fail") == "At Risk"


def test_deployed_regression_model_is_monotonic_in_key_features(client):
    """Regression test for a reported bug: the auto-selected 'best Test R2'
    regression model predicted WORSE grades for MORE attendance/study hours,
    because Test R2 was near-zero for every candidate on this dataset and
    picking blindly by that metric surfaced noise, not signal. The deployed
    model must never predict a worse outcome as attendance, study_hours, or
    previous_grade increases (all else held equal)."""
    from app.services import ml_service

    base = {
        "attendance": 50, "study_hours": 10, "previous_grade": 50,
        "extracurricular": 1, "gender": "Male", "parental_support": "Medium",
        "online_classes": True,
    }

    for feature, values in [
        ("attendance", [0, 25, 50, 75, 100]),
        ("study_hours", [0, 8, 15, 22, 30]),
        ("previous_grade", [0, 25, 50, 75, 100]),
    ]:
        prev_grade = None
        for v in values:
            raw = dict(base)
            raw[feature] = v
            predicted = ml_service.predict_full(raw)["predicted_grade"]
            if prev_grade is not None:
                assert predicted >= prev_grade - 0.01, (
                    f"Monotonicity violated: increasing {feature} to {v} "
                    f"dropped the predicted grade from {prev_grade} to {predicted}"
                )
            prev_grade = predicted


def test_result_always_matches_predicted_grade_not_separate_classifier(client):
    """Regression test for a reported bug: Result (Pass/Fail) came from a
    separately-trained classifier's own probability estimate, which could
    (and did) contradict the Predicted Grade -- e.g. grade 80.3 showing
    'Fail'. Result must now be a direct function of Predicted Grade, using
    the same PASS_THRESHOLD Level's tiers are built on."""
    from app.ml.pipeline import PASS_THRESHOLD
    from app.services import ml_service

    for grade_target, gender, support in [
        (85, "Male", "High"), (72, "Female", "Medium"), (40, "Male", "Low"),
    ]:
        raw = {
            "attendance": 85, "study_hours": 15, "previous_grade": grade_target,
            "extracurricular": 1, "gender": gender, "parental_support": support,
            "online_classes": False,
        }
        res = ml_service.predict_full(raw)
        expected = "Pass" if res["predicted_grade"] >= PASS_THRESHOLD else "Fail"
        assert res["pass_fail"] == expected, (
            f"grade={res['predicted_grade']} produced {res['pass_fail']}, expected {expected}"
        )


def test_admin_can_fix_stale_prediction_results(client):
    """Regression test for the exact reported screenshot: a prediction
    stored before the fix (grade 80.3, Result 'Fail', Level 'Very Good')
    must be correctable via the admin recalculate-results action, without
    needing to delete and regenerate it."""
    from app.extensions import db
    from app.models import Prediction

    with client.application.app_context():
        stale = Prediction(
            student_id=None, created_by_id=None,
            attendance=78, study_hours=12, previous_grade=82, extracurricular=1,
            gender="Male", parental_support="Medium", online_classes=False,
            predicted_grade=80.3, pass_fail="Fail", pass_probability=44.38,
            confidence=55.62, performance_level="Very Good",
            regression_model_name="test", classifier_model_name="test",
        )
        db.session.add(stale)
        db.session.commit()
        stale_id = stale.id

    login(client, "admin", "Admin@123")
    resp = client.post("/admin/predictions/recalculate-results", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Recalculated" in resp.data

    with client.application.app_context():
        fixed = db.session.get(Prediction, stale_id)
        assert fixed.pass_fail == "Pass"
        assert fixed.performance_level == "Very Good"


def test_performance_score_ring_shows_consistent_pass_chance_not_confidence(client):
    """Regression test for an audit finding: the 'Performance Score' donut
    was bound to `confidence` (max class probability), which can represent
    confidence in a FAIL outcome yet was always rendered as a green,
    positively-framed ring. It must show pass_probability (a consistently
    meaningful 'chance of passing') and be colored according to the actual
    Result, not always green."""
    login(client, "admin", "Admin@123")

    # A clear, IN-RANGE Fail case: ring must NOT be styled green
    resp = client.post(
        "/admin/predict",
        data={"attendance": "40", "study_hours": "3", "previous_grade": "30",
              "extracurricular": "0", "gender": "Male", "parental_support": "Low"},
    )
    text = resp.data.decode()
    assert "Classifier Est." in text
    assert "independent" in text.lower()  # must be clearly labeled as a separate signal
    assert "pr-label\">Score" not in text  # old, misleading label must be gone
    assert "var(--danger)" in text  # ring colored red/danger for an in-range Fail case

    # An out-of-distribution case: ring must be muted/grey, not falsely red or green,
    # and the estimate must be marked unreliable rather than presented with confidence
    resp_ood = client.post(
        "/admin/predict",
        data={"attendance": "5", "study_hours": "1", "previous_grade": "2",
              "extracurricular": "0", "gender": "Male", "parental_support": "Low"},
    )
    text_ood = resp_ood.data.decode()
    assert "var(--muted)" in text_ood
    assert "unreliable" in text_ood.lower()

    # A clear Pass case: ring should be styled green
    resp2 = client.post(
        "/admin/predict",
        data={"attendance": "95", "study_hours": "25", "previous_grade": "95",
              "extracurricular": "3", "gender": "Male", "parental_support": "High"},
    )
    text2 = resp2.data.decode()
    assert "var(--success)" in text2  # ring colored green for a Pass case


def test_out_of_distribution_inputs_are_flagged(client):
    """Regression test: a prediction request with values well outside the
    training data's observed range must be flagged as an extrapolation,
    not silently presented with the same confidence as a normal in-range
    prediction."""
    from app.services import ml_service

    ood_raw = {
        "attendance": 20, "study_hours": 5, "previous_grade": 10,
        "extracurricular": 0, "gender": "Male", "parental_support": "Medium",
        "online_classes": True,
    }
    result = ml_service.predict_full(ood_raw)
    assert len(result["ood_warnings"]) > 0

    in_range_raw = {
        "attendance": 85, "study_hours": 17, "previous_grade": 75,
        "extracurricular": 1, "gender": "Male", "parental_support": "Medium",
        "online_classes": True,
    }
    result2 = ml_service.predict_full(in_range_raw)
    assert len(result2["ood_warnings"]) == 0


def test_predicted_grade_is_clipped_to_valid_range(client):
    """Regression test: a predicted grade must never fall outside [0, 100]
    -- the underlying regressor was never given an explicit output
    constraint and could (and did) extrapolate past 100 for very high
    inputs."""
    from app.services import ml_service

    extreme_high = {
        "attendance": 100, "study_hours": 40, "previous_grade": 100,
        "extracurricular": 10, "gender": "Male", "parental_support": "High",
        "online_classes": True,
    }
    result = ml_service.predict_full(extreme_high)
    assert 0 <= result["predicted_grade"] <= 100


def test_admin_stale_prediction_checks_use_deployed_threshold(client):
    """Admin stale-result repair must use the currently deployed model
    threshold, not the fallback 70.0 constant, after a retrain at another
    threshold."""
    from app.services import ml_service
    from app.models import Prediction
    from app.extensions import db

    original = ml_service._metrics
    ml_service._metrics = dict(original or {}, pass_threshold=80.0)
    try:
        with client.application.app_context():
            stale = Prediction(
                predicted_grade=75, pass_fail="Pass", performance_level="Good",
                pass_probability=50, confidence=50,
                attendance=80, study_hours=10, previous_grade=70,
                extracurricular=1, gender="Male", parental_support="Medium",
                online_classes=False, regression_model_name="test", classifier_model_name="test",
            )
            db.session.add(stale); db.session.commit(); stale_id = stale.id
        login(client, "admin", "Admin@123")
        client.post("/admin/predictions/recalculate-results", follow_redirects=True)
        with client.application.app_context():
            fixed = db.session.get(Prediction, stale_id)
            assert fixed.pass_fail == "Fail"
    finally:
        ml_service._metrics = original


def test_student_what_if_prediction_does_not_overwrite_profile(client):
    """The self-service predictor is explicitly hypothetical; it must save
    the prediction snapshot without replacing the student's authoritative
    profile fields."""
    from app.extensions import db
    from app.models import Student
    login(client, "student1", "Student@123")
    with client.application.app_context():
        student = Student.query.filter_by(student_code="STU00001").first()
        before = (student.attendance, student.study_hours, student.previous_grade)
    resp = client.post("/student/new-prediction", data={
        "attendance": "30", "study_hours": "2", "previous_grade": "20",
        "extracurricular": "0", "gender": "Male",
        "parental_support": "Low",
    })
    assert resp.status_code == 200
    with client.application.app_context():
        student = Student.query.filter_by(student_code="STU00001").first()
        assert (student.attendance, student.study_hours, student.previous_grade) == before


def test_pass_threshold_reads_from_deployed_model_not_hardcoded_constant(client):
    """Regression test: Result must be derived from the threshold the
    CURRENTLY DEPLOYED classifier was actually trained with (stored in
    metrics.json), not a separate hardcoded constant that could silently
    drift out of sync after a retrain with a different admin-configured
    threshold."""
    from app.services import ml_service

    threshold = ml_service.current_pass_threshold()
    metrics = ml_service.get_metrics()
    assert threshold == metrics["pass_threshold"]

    # A grade exactly at the threshold must be Pass; just below must be Fail
    raw_at = {"attendance": 85, "study_hours": 17, "previous_grade": 75, "extracurricular": 1,
              "gender": "Male", "parental_support": "Medium", "online_classes": True}
    result = ml_service.predict_full(raw_at)
    expected = "Pass" if result["predicted_grade"] >= threshold else "Fail"
    assert result["pass_fail"] == expected


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


def test_about_page_uses_deployed_threshold_not_admin_setting(client):
    from app.services import ml_service
    from app.models import AppSetting
    from app.extensions import db
    original = ml_service._metrics
    ml_service._metrics = dict(original or {}, pass_threshold=82.0)
    try:
        with client.application.app_context():
            settings = AppSetting.get()
            settings.pass_threshold = 65.0
            db.session.commit()
        login(client, "student1", "Student@123")
        resp = client.get("/student/about")
        assert b"82" in resp.data
        assert b"65" not in resp.data
    finally:
        ml_service._metrics = original


def test_password_reset_token_is_single_use(client):
    from app.models import User, PasswordResetToken
    from app.extensions import db
    with client.application.app_context():
        user = User.query.filter_by(username="student1").first()
        raw, record = PasswordResetToken.issue(user, minutes=30)
        db.session.commit()
        found = PasswordResetToken.consume(raw)
        assert found is record
        db.session.commit()
        assert PasswordResetToken.consume(raw) is None
