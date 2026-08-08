# EduPredict — Student Academic Performance Prediction System

*(Formerly branded "SPAS" internally — renamed to EduPredict with a new logo across the whole app.)*

A full-stack, production-style web application that predicts a student's
final grade and Pass/Fail outcome from behavioral and demographic data,
built on top of the ML pipeline in
`notebooks/Student_Prediction_v3_corrected.ipynb`.

**Stack:** Flask + SQLAlchemy (SQLite) + Flask-Login + scikit-learn/XGBoost + Bootstrap 5 + Chart.js

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

- Regression models achieve low R² (best: see `trained_models/metrics.json`)
  because the dataset's engineered/behavioral features have weak linear
  correlation with `FinalGrade` — this matches the notebook's own conclusion.
  Pass/Fail classification is more tractable than exact grade regression.
- `train_model.py` mirrors the notebook's leak-free methodology: cleaning is
  done before the split, imputation/encoding statistics (mode, median, scaler)
  are fit on the training set only, and SMOTE is applied only to the training
  fold before classifier training.
