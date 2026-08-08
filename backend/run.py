from app import create_app

app = create_app()

# Free-tier hosts (e.g. Render's Free plan) reset the local filesystem on
# every restart/wake-from-sleep, which wipes the SQLite database. Rather than
# depend on a paid "pre-deploy command" feature to reseed it, we seed right
# here on every process boot. seed_app() is fully idempotent -- it checks
# for existing data before creating anything, so this is safe to run every
# single time, including on a warm restart where the data is already there.
try:
    from seed import seed_app

    seed_app(app)
except Exception as exc:  # pragma: no cover - defensive: never crash the app over seeding
    app.logger.error(f"Auto-seed on startup failed: {exc}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
