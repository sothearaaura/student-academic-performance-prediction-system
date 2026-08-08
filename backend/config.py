import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # project root
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production-3f8a9c2e")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(DATABASE_DIR, 'app.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Student_Performance_Cleaned.xlsx")
    EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 8  # 8 hours

    PASS_THRESHOLD = 70.0
