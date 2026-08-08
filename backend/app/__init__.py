import os

from flask import Flask

from app.extensions import bcrypt, db, login_manager
from app.services import ml_service


def create_app(config_object="config.Config"):
    app = Flask(__name__)
    app.config.from_object(config_object)

    from config import DATABASE_DIR

    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(app.config["EXPORTS_DIR"], exist_ok=True)
    os.makedirs(app.config["MODELS_DIR"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)

    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from app.routes.admin import admin_bp
    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    from app.routes.student import student_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(student_bp)

    with app.app_context():
        db.create_all()  # idempotent: creates any new tables (e.g. app_settings) without touching existing ones
        ml_service.init_app(app)

    @app.context_processor
    def inject_globals():
        from app.models import AppSetting

        try:
            settings = AppSetting.get()
            return {"app_name": settings.app_name}
        except Exception:
            # Table may not exist yet on a very first request in edge cases; fall back safely.
            return {"app_name": "EduPredict"}

    @app.template_filter("isoutc")
    def isoutc_filter(dt):
        """Renders a datetime (always stored/naive in UTC in this app) as an
        ISO 8601 string with an explicit UTC 'Z' marker, so client-side JS
        can convert it to the viewer's actual local timezone for display."""
        if dt is None:
            return ""
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

    @app.errorhandler(403)
    def forbidden(e):
        from flask import render_template

        return render_template("shared/error.html", code=403, message="You don't have permission to access this page."), 403

    @app.errorhandler(404)
    def not_found(e):
        from flask import render_template

        return render_template("shared/error.html", code=404, message="Page not found."), 404

    return app
