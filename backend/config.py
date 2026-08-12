import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # project root
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")

_DEV_SECRET_KEY_FALLBACK = "dev-secret-key-change-in-production-3f8a9c2e"


def _normalized_database_uri():
    """Reads DATABASE_URL from the environment (set this to your Neon/Postgres
    connection string on Render for real data persistence -- SQLite alone
    doesn't survive Render's free-tier restarts). Falls back to a local
    SQLite file for local development where that's not an issue.

    Some providers hand out URLs starting with 'postgres://', but modern
    SQLAlchemy requires the 'postgresql://' scheme -- this normalizes that
    automatically so it doesn't turn into a confusing connection error."""
    url = os.environ.get("DATABASE_URL")
    is_production = bool(os.environ.get("RENDER")) or os.environ.get("FLASK_ENV") == "production"
    if not url:
        if is_production:
            raise RuntimeError(
                "DATABASE_URL is required in production. Configure a persistent "
                "PostgreSQL database before starting EduPredict. SQLite is allowed "
                "only for local development."
            )
        return f"sqlite:///{os.path.join(DATABASE_DIR, 'app.db')}"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if is_production and not url.startswith(("postgresql://", "postgresql+psycopg://", "postgresql+psycopg2://")):
        raise RuntimeError(
            "Production DATABASE_URL must be a PostgreSQL connection string."
        )
    return url


def _resolve_secret_key() -> str:
    """SECRET_KEY signs session cookies -- if it's predictable, sessions can
    be forged. The dev fallback below is intentionally public (it's sitting
    right here in the source), which is fine for local SQLite development
    but would be a real vulnerability if silently used in production.

    DATABASE_URL being set is this project's existing signal for "this is a
    real deployment, not just someone running it locally" (see
    _normalized_database_uri above) -- reuse that same signal here: require
    SECRET_KEY explicitly whenever DATABASE_URL is present, and fail loudly
    at startup with a clear, actionable message rather than silently
    running with a key anyone can find in this file."""
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    if os.environ.get("DATABASE_URL"):
        raise RuntimeError(
            "SECRET_KEY environment variable is not set. Refusing to start with a "
            "real database configured (DATABASE_URL is set) using the public dev "
            "fallback key, since that would let anyone forge session cookies. "
            "Set SECRET_KEY in your Render Environment tab (or wherever you set "
            "DATABASE_URL) -- render.yaml already generates one automatically if "
            "you deploy via the Render Blueprint, but a manually-created service "
            "needs it added by hand."
        )
    return _DEV_SECRET_KEY_FALLBACK


class Config:
    SECRET_KEY = _resolve_secret_key()
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
