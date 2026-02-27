"""Auth integration with Supabase and a local fallback.

When SUPABASE_URL and SUPABASE_ANON_KEY are set, auth uses Supabase.
Otherwise auth uses a local file-backed user store (for dev/demo).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)
LOCAL_AUTH_SECRET_ENV = "LOCAL_AUTH_SECRET"
LOCAL_AUTH_DEFAULT_SECRET = "ec3-local-dev-secret-change-me"
LOCAL_AUTH_TTL_HOURS = 24 * 7


class AuthRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class AuthResponse(BaseModel):
    access_token: str
    user_id: str
    email: str
    refresh_token: str | None = None
    expires_at: int | None = None  # Unix timestamp


class RefreshRequest(BaseModel):
    refresh_token: str


class LocalAuthStore:
    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict:
        if not self.path.exists():
            return {"users": []}
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"users": []}
        users = parsed.get("users") if isinstance(parsed, dict) else []
        if not isinstance(users, list):
            users = []
        return {"users": [u for u in users if isinstance(u, dict)]}

    def _save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().lower()

    def _find_by_email(self, payload: dict, email: str) -> dict | None:
        norm = self._normalize_email(email)
        for user in payload.get("users", []):
            if self._normalize_email(str(user.get("email", ""))) == norm:
                return user
        return None

    def create_user(self, email: str, password: str) -> dict:
        payload = self._load()
        if self._find_by_email(payload, email):
            raise HTTPException(409, "Account already exists.")

        salt = secrets.token_bytes(16)
        password_hash = _derive_password_hash(password, salt)
        user = {
            "user_id": secrets.token_hex(16),
            "email": self._normalize_email(email),
            "salt_b64": base64.b64encode(salt).decode("ascii"),
            "password_hash_b64": base64.b64encode(password_hash).decode("ascii"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload["users"].append(user)
        self._save(payload)
        return user

    def verify_user(self, email: str, password: str) -> dict | None:
        payload = self._load()
        user = self._find_by_email(payload, email)
        if not user:
            return None
        salt_b64 = str(user.get("salt_b64", ""))
        stored_hash_b64 = str(user.get("password_hash_b64", ""))
        if not salt_b64 or not stored_hash_b64:
            return None
        try:
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(stored_hash_b64.encode("ascii"))
        except Exception:
            return None
        candidate = _derive_password_hash(password, salt)
        if not hmac.compare_digest(candidate, expected):
            return None
        return user


def _derive_password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)


def _local_auth_secret() -> str:
    return os.getenv(LOCAL_AUTH_SECRET_ENV, LOCAL_AUTH_DEFAULT_SECRET)


def _issue_local_token(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=LOCAL_AUTH_TTL_HOURS)).timestamp()),
        "iss": "ec3-local-auth",
    }
    return _encode_local_token(payload, _local_auth_secret())


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode((raw + padding).encode("ascii"))


def _encode_local_token(payload: dict, secret: str) -> str:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_part = _b64url_encode(body)
    signature = hmac.new(secret.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
    sig_part = _b64url_encode(signature)
    return f"{payload_part}.{sig_part}"


def _decode_local_token(token: str, secret: str) -> dict:
    try:
        payload_part, sig_part = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("Malformed token.") from exc

    expected_sig = hmac.new(secret.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
    given_sig = _b64url_decode(sig_part)
    if not hmac.compare_digest(given_sig, expected_sig):
        raise ValueError("Invalid token signature.")

    try:
        payload = json.loads(_b64url_decode(payload_part).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid token payload.") from exc

    exp = int(payload.get("exp", 0))
    if exp <= int(datetime.now(timezone.utc).timestamp()):
        raise ValueError("Token expired.")

    return payload


def create_auth_router(settings: Any) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["auth"])

    if settings.auth_enabled:
        from supabase import create_client

        supabase = create_client(settings.supabase_url, settings.supabase_anon_key)

        @router.get("/status")
        async def auth_status():
            return {
                "enabled": True,
                "provider": "supabase",
                "threads_sync": bool(settings.supabase_service_role_key),
            }

        @router.post("/signup", response_model=AuthResponse)
        async def signup(req: AuthRequest):
            try:
                res = supabase.auth.sign_up({"email": req.email, "password": req.password})
                if res.user is None:
                    raise HTTPException(400, "Signup failed — check email/password requirements.")
                session = res.session
                if not session:
                    return AuthResponse(
                        access_token="",
                        user_id=res.user.id,
                        email=res.user.email or str(req.email),
                    )
                exp = int(session.expires_at) if getattr(session, "expires_at", None) else None
                return AuthResponse(
                    access_token=session.access_token,
                    user_id=res.user.id,
                    email=res.user.email or str(req.email),
                    refresh_token=getattr(session, "refresh_token", None) or "",
                    expires_at=exp,
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("signup_failed", extra={"error": str(exc)})
                raise HTTPException(400, str(exc)) from exc

        @router.post("/login", response_model=AuthResponse)
        async def login(req: AuthRequest):
            try:
                res = supabase.auth.sign_in_with_password({"email": req.email, "password": req.password})
                if res.user is None or res.session is None:
                    raise HTTPException(401, "Invalid credentials.")
                session = res.session
                exp = int(session.expires_at) if getattr(session, "expires_at", None) else None
                return AuthResponse(
                    access_token=session.access_token,
                    user_id=res.user.id,
                    email=res.user.email or str(req.email),
                    refresh_token=getattr(session, "refresh_token", None) or "",
                    expires_at=exp,
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("login_failed", extra={"error": str(exc)})
                raise HTTPException(401, str(exc)) from exc

        @router.post("/logout")
        async def logout():
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            return {"ok": True}

        @router.post("/forgot-password")
        async def forgot_password(req: ForgotPasswordRequest):
            try:
                supabase.auth.reset_password_for_email(req.email)
                return {"ok": True, "message": "If an account exists, you will receive a reset link."}
            except Exception as exc:
                logger.warning("forgot_password_failed", extra={"error": str(exc)})
                return {"ok": True, "message": "If an account exists, you will receive a reset link."}

        @router.post("/refresh")
        async def refresh(req: RefreshRequest):
            try:
                res = supabase.auth.refresh_session(req.refresh_token)
                if not res.session:
                    raise HTTPException(401, "Refresh failed")
                session = res.session
                exp = int(session.expires_at) if getattr(session, "expires_at", None) else None
                return {
                    "access_token": session.access_token,
                    "refresh_token": getattr(session, "refresh_token", None) or req.refresh_token,
                    "expires_at": exp,
                }
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("refresh_failed", extra={"error": str(exc)})
                raise HTTPException(401, "Session expired") from exc

        @router.get("/me")
        async def me(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
            if not creds:
                raise HTTPException(401, "Not authenticated.")
            user = _verify_token(creds.credentials, settings)
            return {"user_id": user["sub"], "email": user.get("email", "")}

        return router

    local_store = LocalAuthStore(settings.project_root / "data" / "local_auth_users.json")

    @router.get("/status")
    async def auth_status():
        return {"enabled": True, "provider": "local", "threads_sync": False}

    @router.post("/signup", response_model=AuthResponse)
    async def signup(req: AuthRequest):
        if len(req.password) < 6:
            raise HTTPException(400, "Password must be at least 6 characters.")
        user = local_store.create_user(str(req.email), req.password)
        token = _issue_local_token(user["user_id"], user["email"])
        return AuthResponse(access_token=token, user_id=user["user_id"], email=user["email"])

    @router.post("/login", response_model=AuthResponse)
    async def login(req: AuthRequest):
        user = local_store.verify_user(str(req.email), req.password)
        if not user:
            raise HTTPException(401, "Invalid credentials.")
        token = _issue_local_token(user["user_id"], user["email"])
        return AuthResponse(access_token=token, user_id=user["user_id"], email=user["email"])

    @router.post("/logout")
    async def logout():
        return {"ok": True}

    @router.post("/forgot-password")
    async def forgot_password(req: ForgotPasswordRequest):
        return {"ok": True, "message": "Password reset is not available with local auth. Contact support."}

    @router.post("/refresh")
    async def refresh(req: RefreshRequest):
        raise HTTPException(400, "Session refresh not available with local auth")

    @router.get("/me")
    async def me(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
        if not creds:
            raise HTTPException(401, "Not authenticated.")
        user = _verify_token(creds.credentials, settings)
        return {"user_id": user["sub"], "email": user.get("email", "")}

    return router


def _verify_token(token: str, settings: Any) -> dict:
    if settings.auth_enabled:
        if not settings.supabase_jwt_secret:
            from jose import jwt as jose_jwt

            payload = jose_jwt.decode(
                token,
                settings.supabase_jwt_secret or "secret",
                algorithms=["HS256"],
                options={"verify_signature": False},
            )
            return payload

        from jose import JWTError, jwt as jose_jwt

        try:
            payload = jose_jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
            return payload
        except JWTError as exc:
            raise HTTPException(401, f"Invalid token: {exc}") from exc

    try:
        return _decode_local_token(token, _local_auth_secret())
    except ValueError as exc:
        raise HTTPException(401, f"Invalid token: {exc}") from exc


def get_optional_user(settings: Any):
    async def _dep(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> dict | None:
        if not creds:
            return None
        try:
            return _verify_token(creds.credentials, settings)
        except HTTPException:
            return None

    return _dep


def require_auth(settings: Any):
    async def _dep(creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> dict | None:
        if not creds:
            raise HTTPException(401, "Authentication required.")
        return _verify_token(creds.credentials, settings)

    return _dep
