# EduPredict — Student Academic Performance Prediction System

*(Formerly branded "SPAS" internally — renamed to EduPredict with a new logo across the whole app.)*

A full-stack, production-style web application that predicts a student's
final grade and Pass/Fail outcome from behavioral and demographic data,
built on top of the ML pipeline in
`notebooks/Student_Prediction_v3_corrected.ipynb`.

**Stack:** Flask + SQLAlchemy (PostgreSQL in production; SQLite for local development) + Flask-Login + scikit-learn/XGBoost + Bootstrap 5 + Chart.js

## 1. Project Structure

```
project/
├── backend/
│   ├── app/
│   │   ├── ml/                 # Shared preprocessing/prediction pipeline class
│   │   ├── models/             # SQLAlchemy models (User, Student, Prediction, ...)
│   │   ├── routes/             # auth / admin / student blueprints
│   │   ├── services/           # ml_service.py (loads pkl), report_service.py (PDF)
│   │   ├── templates/          # Jinja2 + Bootstrap 5 templates, per role
│   │   ├── static/             # CSS/JS
│   │   └── __init__.py         # App factory
│   ├── trained_models/         # regression_pipeline.pkl, classification_pipeline.pkl, metrics.json, feature_names.json
│   ├── dataset/                # Student_Performance_Cleaned.xlsx (production dataset)
│   ├── exports/                # Generated PDF reports land here
│   ├── config.py
│   ├── train_model.py          # Run ONCE to train & export the ML pipelines
│   ├── seed.py                 # Creates tables + demo accounts + imports dataset as students
│   ├── run.py                  # Flask entry point (also auto-seeds on every boot)
│   └── requirements.txt
├── database/                   # SQLite file lives here (app.db)
├── notebooks/                  # Original Jupyter notebook (source of the ML pipeline)
├── docs/                       # Architecture notes
└── tests/                      # Smoke tests
```

## 2. Setup

## 1.5. REQUIRED: PostgreSQL for production

Production deployments **must use PostgreSQL**. The application intentionally
refuses to start on Render/production when `DATABASE_URL` is missing or points
to SQLite. SQLite remains available only for local development and testing.

**Render setup:**
1. Create a managed PostgreSQL database with your preferred provider.
2. Set `DATABASE_URL` in the Render service Environment settings.
3. Set `SECRET_KEY` to a long random value.
4. Configure the SMTP variables below for password-reset email delivery.
5. Deploy. The application fails fast rather than silently falling back to
   ephemeral SQLite storage.

The schema is created on startup with `db.create_all()`. For future schema
changes in a long-lived production database, use a real migration tool.

### Password reset email configuration

Forgot-password uses a **single-use, 30-minute reset token**. Only a SHA-256
hash of the token is stored; outstanding tokens are invalidated when a newer
one is issued, and a token is consumed when the password is changed.

Production requires:
- `SMTP_HOST`
- `SMTP_PORT` (normally `587`)
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM`
- `SMTP_TLS=true`

The application returns the same generic response whether or not an account
exists for the submitted email, reducing account-enumeration risk.

## 2. Setup

```bash
cd backend
python3 -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# 1. Train the models (reads dataset/, writes trained_models/*.pkl + metrics.json)
python train_model.py

# 2. Create the database, demo accounts, and import the FULL dataset (all rows)
#    as managed students
python seed.py

# 3. Run the app
python run.py
```

Visit **http://localhost:5000**.

## 3. Accounts

**Demo accounts (created by seed.py):**

| Role    | Username | Password    |
|---------|----------|-------------|
| Admin   | admin    | Admin@123   |
| Student | student1 | Student@123 |

**Student self-registration**: new students can create their own account at
`/signup` (linked from the login page). Two paths:
- **Claim an existing record** — if an admin already entered their academic
  data, they enter the Student Code they were given and their new login is
  linked to that real data immediately.
- **Register fresh** — leave the Student Code blank and a new, blank student
  record is created alongside the account; an admin fills in their academic
  data later from the Manage Students page.

Admin accounts are never self-service — only an existing admin can create
another admin account, from the Users page.

## 4. How prediction stays in sync with the notebook

`app/ml/pipeline.py` defines `StudentPerformancePipeline`, a single class used by
**both** `train_model.py` (at training time) and `app/services/ml_service.py`
(at request time). It bundles the fitted `SimpleImputer`, `StandardScaler`,
final estimator, and the exact encoding maps/feature order from the notebook's
Step 5. There is no second, hand-written copy of the preprocessing logic that
could drift out of sync — the website's prediction for a given input is
identical to what the notebook's pipeline would produce.

To retrain on an updated dataset: use the "Upload & Replace" and "Retrain
Model Now" buttons on the admin Dataset/ML Model pages, or manually replace
`backend/dataset/Student_Performance_Cleaned.xlsx` and re-run
`python train_model.py`.

## 5. Roles & Permissions

This is a **two-role system**: Admin manages everything; Student has a
read-only self-service view.

- **Admin** — full system access: register/edit/delete students, run
  predictions, view prediction history and analytics, generate per-student
  and system-wide PDF reports, manage user accounts, view login/activity
  logs, view model metrics, upload a replacement dataset (with automatic
  backup of the previous file), trigger a live model retrain with one click
  (no server restart needed), and edit real system settings (app name,
  Pass/Fail threshold used on the next retrain, session timeout, and a
  maintenance-mode switch that blocks non-admin logins).
- **Student** — logs in and gets a **real-time prediction generated automatically**
  the moment they view their dashboard (no admin action required first): their
  current stored academic data is run through the same ML pipeline as admin's
  Predict page, and the result — Predicted Grade, Confidence Score, Risk Level,
  and Suggestions for Improvement — is generated and displayed immediately. It's
  also stored to their history automatically, but only when it's new information
  (the underlying data changed since their last stored prediction), so repeat
  dashboard visits don't spam duplicate history entries. Students also have a
  **New Prediction** page — a self-service "what-if" wizard where they can try
  hypothetical values and see an instant AI prediction with a circular
  confidence ring, personalized recommendations, and an input summary — plus
  an **About** page explaining the system. Can download their own PDF report.
  Cannot see other students or admin pages.

A student gets a login account either by an admin checking "Create a login
account for this student" when registering/editing them, or via the seed
script's demo `student1` account.

## 6. Database Schema

`users`, `students`, `predictions`, `reports`, `login_history`,
`activity_logs`, `app_settings` — see `app/models/` for exact columns and
relationships.

## 7. Notes for the written report / defense

### Dataset audit (August 2026) — the original dataset had no learnable signal

A full audit found the originally-provided `FinalGrade` target had **no genuine
relationship** to the input features — not just a weak one:

- `FinalGrade` contained only **10 unique discrete values** (62, 68, 72, 78,
  80, 85, 87, 88, 90, 92) across all 950 rows, each appearing a roughly
  uniform ~77–109 times.
- Pearson correlation between every numeric feature and `FinalGrade` was
  effectively zero: Attendance r=−0.018, Study Hours r=+0.033, Previous
  Grade r=+0.003, Extracurricular r=−0.030.
- Mean `FinalGrade` was flat (79–81) across every quintile of attendance,
  study hours, and previous grade — confirmed with a groupby, ruling out a
  non-linear relationship a simple correlation could miss.

This explains why every regression model tried — Linear, Ridge, Lasso,
Decision Tree, Random Forest, Gradient Boosting, XGBoost — scored a
**negative Test R²**: a negative R² specifically means the model performs
*worse* than always predicting the mean, which is the expected outcome
when trying to fit a target that has no real relationship to the inputs.
No algorithm choice, hyperparameter tuning, or feature engineering could
have fixed this — the ground truth itself didn't depend on these features.

**Fix**: since no alternative real-world dataset was available, `FinalGrade`
was reconstructed using a transparent, documented formula based on the
*same* students' actual feature values (attendance, study hours, previous
grade, parental support), plus genuine Gaussian noise so it isn't a
trivially perfect/deterministic relationship:

```
FinalGrade = 0.25 × Attendance
           + 0.25 × min(StudyHours, 25)/25 × 100
           + 0.50 × PreviousGrade
           + ParentalSupport bonus (Low=0, Medium=5, High=10)
           + Normal(0, 4.5) noise
           , clipped to [0, 100]
```

The **original file is preserved** at
`dataset/original_backup/Student_Performance_Cleaned_ORIGINAL_synthetic_flat_target.xlsx`
for comparison. This reconstruction should be clearly disclosed in any
report/defense as a documented synthetic target, not real-world ground
truth — the point is to demonstrate a genuinely working end-to-end ML
pipeline (data → training → validation → monotonic constraints → serving),
which the original data made impossible to validate honestly.

**Before / after retraining on the reconstructed dataset:**

| Metric | Before (original data) | After (reconstructed data) |
|---|---|---|
| Regression Test R² | **−0.1991** | **0.8100** |
| Regression Test RMSE | ~9.9 | 4.45 |
| Regression Test MAE | ~8.1 | 3.51 |
| Regression 5-fold CV R² | not computed | **0.7682 ± 0.0397** |
| Classification F1 | 0.8397 | **0.9515** |
| Classification Accuracy | — | 0.9185 |
| Classification AUC | ~0.49 (near-random) | **0.9653** |
| Classification 5-fold CV F1 | not computed | **0.9638 ± 0.0129** |

5-fold cross-validation (`sklearn.model_selection.cross_val_score`) uses the
deployed model configuration, leak-free preprocessing, and the same SMOTE
training recipe inside each classification fold. This confirms the metrics
are robust across different splits rather than relying on one lucky train/test
partition — see the `deployed_*_cv_*` fields in `trained_models/metrics.json`.

### Model selection: monotonic constraints deployed over raw "best R²"

`train_model.py` deliberately does **not** just deploy whichever model wins
on raw Test R²/F1. Early testing (against the original broken data)
surfaced whichever model won by that metric predicted a *worse* grade for
*more* attendance and *more* study hours — backwards relationships, because
a near-zero-R² model is essentially fitting noise, and noise has no
obligation to point the right direction. An additional **XGBoost
(Monotonic)** candidate is always trained using XGBoost's native
`monotone_constraints`, forcing attendance, study hours, previous grade,
and parental support to never predict a worse outcome as they increase —
and this is always deployed over the raw leaderboard winner. One derived
feature, `Grade_Study_Ratio = PreviousGrade / (StudyHours + 1)`, is
excluded entirely from the monotonic model's inputs because it structurally
decreases as study hours increases, which no per-column constraint can fix
(it gets its own reduced feature set + separately-fit imputer/scaler, see
`MONOTONIC_FEATURE_COLS` in `train_model.py`).

Verified two ways: a targeted sensitivity sweep across attendance/study
hours/previous grade, and a 30-trial × 3-feature randomized stress test —
both showed zero monotonicity violations (see
`tests/test_smoke.py::test_deployed_regression_model_is_monotonic_in_key_features`).

### Consistency fixes across Predicted Grade / Result / Level / Probability / Risk

Several fields were previously computed by *independent* models/logic paths
that could silently contradict each other (e.g. a grade of 80.3 — "Very
Good" by any reasonable bucket — showing Result: "Fail," because Result
came from a separately-trained classifier's own probability estimate, not
from the grade at all):

- **Result** is now derived directly from **Predicted Grade** compared
  against the pass threshold stored with the currently deployed model in
  `metrics.json`. Admin Settings explicitly applies a changed threshold on
  the **next retrain**, so the live Settings value can intentionally differ
  from the currently deployed model until retraining is completed. Admin
  stale-result/recalculation checks use the deployed threshold, not a
  hardcoded 70.
- **Level**'s tier boundaries anchor to that same threshold (`Good` starts
  exactly at the threshold, `Very Good`/`Excellent` are +10/+20 above it),
  so Level and Result structurally agree rather than needing to be patched
  into agreement after the fact.
- **Pass Probability** is the separately-trained classifier's probability for
  the class labeled `1` (Pass), resolved through the estimator's `classes_`
  metadata rather than assuming array position. **Classifier Confidence** is
  the probability of the classifier's winning class. Both are independent
  signals; the displayed Result is determined by Predicted Grade and the
  deployed pass threshold. With the reconstructed dataset giving the
  classifier real signal to learn from, these signals agree with the
  grade-based Result in the overwhelming majority of cases anyway.
- **Recommendations** were audited against real submitted values (not
  hardcoded/sample data) — confirmed the "online classes" recommendation
  correctly does not appear when a student's actual input says they're
  already taking online classes.
- The "Performance Score" ring was previously bound to `confidence`
  (`max(class probability)`), which can represent confidence in a **Fail**
  outcome yet was always rendered in green with encouraging text — now
  shows `pass_probability` specifically and is colored red/green to match
  the actual Result.

### Out-of-distribution (OOD) warnings

`train_model.py` now saves the observed min/max of attendance, study hours,
and previous grade from the actual training data to
`trained_models/feature_names.json`. `ml_service.predict_full()` checks
submitted values against these ranges and returns `ood_warnings` — surfaced
as a prominent red banner on both the admin and student prediction pages —
whenever a value falls outside what the model was ever trained on, making
clear that prediction is an extrapolation rather than a normal
in-distribution estimate.

### Output range fix

The regression model was never given an explicit output constraint and
could (and did) predict slightly above 100 for very high inputs (e.g.
104.02 on a max-inputs test case) even though the training data itself was
clipped to [0, 100]. `StudentPerformancePipeline.predict()` now clips the
final output to `[0, 100]`.

`train_model.py` mirrors the notebook's leak-free methodology: cleaning is
done before the split, imputation/encoding statistics (mode, median, scaler)
are fit on the training set only, and SMOTE is applied only to the training
fold before classifier training.

### Second audit pass — broadened feature distributions + remaining gaps closed

A follow-up, more detailed 22-point audit request found the first pass had
fixed the *target* (`FinalGrade`) but left the *input feature distributions*
too narrow — Attendance was still only 50–98%, Previous Grade only 60–90 —
meaning genuinely poor-performing students were never represented in
training at all, which is exactly why a 20%-attendance test case landed
out-of-distribution. Feature distributions were regenerated using wider,
more realistic truncated-normal distributions (Attendance 26–100%, Study
Hours 0–39, Previous Grade 21–100), keeping the same transparent, documented
`FinalGrade` formula applied to this broader population.

**Metrics after broadening (most current):**

| Metric | Value |
|---|---|
| Regression Test R² | 0.8247 |
| Regression Test RMSE / MAE | 5.24 / 4.26 |
| Regression 5-fold CV R² | 0.8404 ± 0.0361 |
| Classification Accuracy | 0.8474 |
| Classification Precision / Recall | 0.8722 / 0.9062 |
| Classification F1 | 0.8889 |
| Classification AUC | 0.9293 |
| Classification 5-fold CV F1 | 0.9274 ± 0.0119 |
| Confusion Matrix | TN=45, FP=17, FN=12, TP=116 (see `metrics.json`) |

**Test A/B/C results (exact scenarios from the audit request):**

| | Attendance | Study Hrs | Prev Grade | Predicted Grade | Result | Level | OOD? |
|---|---|---|---|---|---|---|---|
| A — Poor | 20% | 5 | 10 | **32.95** | Fail | At Risk | Yes (2 warnings) |
| B — Average | 75% | 15 | 65 | **71.93** | Pass | Good | No |
| C — Strong | 95% | 25 | 90 | **100.0** | Pass | Excellent | No |

Grade strictly increases A→B→C, confirming the model responds correctly to
these factors in the expected direction.

**Additional fixes in this pass:**
- **Recommendations are now severity-tiered**, not generic — e.g.
  attendance below 60% now says "critical attendance issue... recommend an
  immediate meeting," not the same generic "below 80%" text regardless of
  how far below. Same tiering added for study hours and previous grade.
- **Confusion matrix** added to classification validation (`metrics.json` →
  `deployed_classification_confusion_matrix`).
- **OOD predictions no longer show a falsely confident colored ring** — when
  an input is flagged out-of-distribution, the classifier estimate ring
  turns grey/muted instead of red or green, with an explicit "* this
  estimate is unreliable" note, rather than presenting an extrapolated
  number with the same visual confidence as an in-distribution one.

**Remaining limitations (honest, not hidden):**
- `FinalGrade` is still a documented, formula-based reconstruction, not
  real-world outcome data — appropriate for demonstrating a genuinely
  working ML pipeline, but should be disclosed as such in any report/defense,
  not presented as real student outcomes.
- Regression accounts for roughly 82-84% of variance (R²≈0.82-0.84) with the
  noise level chosen (σ=4.5) — intentionally not a perfect/deterministic fit,
  since a real academic-outcome predictor never would be either.
- The classifier's own independent probability estimate can still
  occasionally diverge from the grade-derived Result in genuinely
  borderline cases (e.g. Test B: Result=Pass at grade 71.93, but classifier
  estimate only 41.8%) — this is disclosed via the "Classifier Est." label
  and explanatory note, not hidden, since forcing them to always agree
  would mean fabricating one of the two numbers.


### Prediction semantics

The Student **New Prediction** page is a what-if simulator. Submitted values are
saved as an immutable prediction snapshot but do **not** overwrite the student's
authoritative profile. Admin-created predictions may update a student's current
grade when they are explicitly linked to that student.
