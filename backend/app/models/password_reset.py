import hashlib
import secrets
from datetime import datetime, timedelta

from app.extensions import db


class PasswordResetToken(db.Model):
    __tablename__ = "password_reset_tokens"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    token_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    used_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship("User", backref=db.backref("password_reset_tokens", lazy="dynamic"))

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @classmethod
    def issue(cls, user, minutes=30):
        # Invalidate outstanding tokens for this account before issuing a new one.
        now = datetime.utcnow()
        cls.query.filter_by(user_id=user.id, used_at=None).update(
            {"used_at": now}, synchronize_session=False
        )
        raw = secrets.token_urlsafe(48)
        record = cls(
            user_id=user.id,
            token_hash=cls.hash_token(raw),
            expires_at=now + timedelta(minutes=minutes),
        )
        db.session.add(record)
        return raw, record

    @classmethod
    def consume(cls, raw: str):
        record = cls.query.filter_by(token_hash=cls.hash_token(raw), used_at=None).first()
        if not record or record.expires_at <= datetime.utcnow():
            return None
        record.used_at = datetime.utcnow()
        return record
