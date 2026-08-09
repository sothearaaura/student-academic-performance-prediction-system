import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # project root
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")


def _normalized_database_uri():
    """Reads DATABASE_URL from the environment (set this to your Neon/Postgres
    connection string on Render for real data persistence -- SQLite alone
    doesn't survive Render's free-tier restarts). Falls back to a local
    SQLite file for local development where that's not an issue.

    Some providers hand out URLs starting with 'postgres://', but modern
    SQLAlchemy requires the 'postgresql://' scheme -- this normalizes that
    automatically so it doesn't turn into a confusing connection error."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        return f"sqlite:///{os.path.join(DATABASE_DIR, 'app.db')}"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production-3f8a9c2e")
    SQLALCHEMY_DATABASE_URI = _normalized_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Postgres connections can go stale after periods of inactivity (exactly
    # the free-tier sleep/wake pattern this app runs under) -- pre-ping
    # transparently discards a dead connection and opens a fresh one instead
    # of raising an error on the next query.
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Student_Performance_Cleaned.xlsx")
    EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 8  # 8 hours

    PASS_THRESHOLD = 70.0
