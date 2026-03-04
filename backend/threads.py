"""Thread and message persistence with Supabase.

When SUPABASE_SERVICE_ROLE_KEY is set, threads and messages are synced to Supabase.
Otherwise the API returns 503 and the frontend falls back to localStorage.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth import get_optional_user, require_auth

logger = logging.getLogger(__name__)


class ThreadCreate(BaseModel):
    title: str = "New chat"


class ThreadPatch(BaseModel):
    title: str | None = None


class MessageCreate(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str = ""
    response_payload: dict | None = None


class ThreadTruncate(BaseModel):
    """Truncate a thread: keep only the first `keep_count` messages.

    Optionally update the content of the last kept message (for inline edits).
    """
    keep_count: int
    updated_content: str | None = None


def create_threads_router(settings: Any) -> APIRouter:
    router = APIRouter(prefix="/api/threads", tags=["threads"])
    user_dep = require_auth(settings)

    def get_supabase():
        if not (settings.supabase_url and settings.supabase_service_role_key):
            return None
        from supabase import create_client

        return create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )

    @router.get("")
    async def list_threads(user: dict = Depends(user_dep)):
        supabase = get_supabase()
        if not supabase:
            raise HTTPException(503, "Thread sync not configured")

        user_id = user.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        try:
            res = (
                supabase.table("threads")
                .select("id, title, created_at, updated_at")
                .eq("user_id", user_id)
                .order("updated_at", desc=True)
                .execute()
            )
            rows = res.data or []
        except Exception as exc:
            logger.exception("threads_list_failed")
            raise HTTPException(500, str(exc)) from exc

        return {
            "threads": [
                {
                    "id": str(r["id"]),
                    "title": r.get("title", "New chat"),
                    "createdAt": r.get("created_at"),
                    "updatedAt": r.get("updated_at"),
                }
                for r in rows
            ],
        }

    @router.post("")
    async def create_thread(
        body: ThreadCreate | None = None,
        user: dict = Depends(user_dep),
    ):
        supabase = get_supabase()
        if not supabase:
            raise HTTPException(503, "Thread sync not configured")

        user_id = user.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        payload = {
            "user_id": user_id,
            "title": (body.title if body else "New chat").strip() or "New chat",
        }
        try:
            res = supabase.table("threads").insert(payload).execute()
            row = (res.data or [{}])[0]
        except Exception as exc:
            logger.exception("thread_create_failed")
            raise HTTPException(500, str(exc)) from exc

        return {
            "id": str(row.get("id", "")),
            "title": row.get("title", "New chat"),
            "createdAt": row.get("created_at"),
            "updatedAt": row.get("updated_at"),
        }

    @router.get("/{thread_id}")
    async def get_thread(thread_id: str, user: dict = Depends(user_dep)):
        supabase = get_supabase()
        if not supabase:
            raise HTTPException(503, "Thread sync not configured")

        user_id = user.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        try:
            res = (
                supabase.table("threads")
                .select("id, title, created_at, updated_at")
                .eq("id", thread_id)
                .eq("user_id", user_id)
                .maybe_single()
                .execute()
            )
            row = res.data
        except Exception as exc:
            logger.exception("thread_get_failed")
            raise HTTPException(500, str(exc)) from exc

        if not row:
            raise HTTPException(404, "Thread not found")

        msgs_res = (
            supabase.table("messages")
            .select("id, role, content, response_payload, created_at")
            .eq("thread_id", thread_id)
            .order("created_at", desc=False)
            .execute()
        )
        messages = (msgs_res.data or [])

        return {
            "id": str(row["id"]),
            "title": row.get("title", "New chat"),
            "createdAt": row.get("created_at"),
            "updatedAt": row.get("updated_at"),
            "messages": [
                {
                    "id": str(m["id"]),
                    "role": m.get("role", "user"),
                    "content": m.get("content", ""),
                    "responsePayload": m.get("response_payload"),
                    "createdAt": m.get("created_at"),
                }
                for m in messages
            ],
        }

    @router.patch("/{thread_id}")
    async def patch_thread(
        thread_id: str,
        body: ThreadPatch,
        user: dict = Depends(user_dep),
    ):
        supabase = get_supabase()
        if not supabase:
            raise HTTPException(503, "Thread sync not configured")

        user_id = user.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        updates = {}
        if body.title is not None:
            updates["title"] = body.title.strip() or "New chat"
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        if not updates:
            res = (
                supabase.table("threads")
                .select("id, title, created_at, updated_at")
                .eq("id", thread_id)
                .eq("user_id", user_id)
                .maybe_single()
                .execute()
            )
            row = res.data
            if not row:
                raise HTTPException(404, "Thread not found")
            return {
                "id": str(row["id"]),
                "title": row.get("title", "New chat"),
                "createdAt": row.get("created_at"),
                "updatedAt": row.get("updated_at"),
            }

        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            res = (
                supabase.table("threads")
                .update({"title": updates["title"], "updated_at": now_iso})
                .eq("id", thread_id)
                .eq("user_id", user_id)
                .execute()
            )
        except Exception as exc:
            logger.exception("thread_patch_failed")
            raise HTTPException(500, str(exc)) from exc

        row = (res.data or [{}])[0]
        if not row:
            raise HTTPException(404, "Thread not found")
        return {
            "id": str(row.get("id", "")),
            "title": row.get("title", "New chat"),
            "createdAt": row.get("created_at"),
            "updatedAt": row.get("updated_at"),
        }

    @router.post("/{thread_id}/messages")
    async def add_message(
        thread_id: str,
        body: MessageCreate,
        user: dict = Depends(user_dep),
    ):
        supabase = get_supabase()
        if not supabase:
            raise HTTPException(503, "Thread sync not configured")

        user_id = user.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        if body.role not in ("user", "assistant"):
            raise HTTPException(400, "role must be 'user' or 'assistant'")

        # Verify thread belongs to user
        thread_res = (
            supabase.table("threads")
            .select("id")
            .eq("id", thread_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if not thread_res.data:
            raise HTTPException(404, "Thread not found")

        payload = {
            "thread_id": thread_id,
            "role": body.role,
            "content": body.content or "",
            "response_payload": body.response_payload,
        }
        try:
            msg_res = supabase.table("messages").insert(payload).execute()
            msg_row = (msg_res.data or [{}])[0]

            # Update thread updated_at
            now_iso = datetime.now(timezone.utc).isoformat()
            supabase.table("threads").update({"updated_at": now_iso}).eq(
                "id", thread_id
            ).eq("user_id", user_id).execute()
        except Exception as exc:
            logger.exception("message_add_failed")
            raise HTTPException(500, str(exc)) from exc

        return {
            "id": str(msg_row.get("id", "")),
            "role": msg_row.get("role", body.role),
            "content": msg_row.get("content", body.content),
            "responsePayload": msg_row.get("response_payload"),
            "createdAt": msg_row.get("created_at"),
        }

    @router.post("/{thread_id}/truncate")
    async def truncate_thread(
        thread_id: str,
        body: ThreadTruncate,
        user: dict = Depends(user_dep),
    ):
        """Delete messages after a given position and optionally update the last kept message.

        Used when the user edits a previous message and resubmits — all
        subsequent messages must be removed and the edited message content
        updated in the database.
        """
        supabase = get_supabase()
        if not supabase:
            raise HTTPException(503, "Thread sync not configured")

        user_id = user.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        # Verify thread belongs to user
        thread_res = (
            supabase.table("threads")
            .select("id")
            .eq("id", thread_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if not thread_res.data:
            raise HTTPException(404, "Thread not found")

        try:
            # Fetch all messages ordered by creation time
            msgs_res = (
                supabase.table("messages")
                .select("id, created_at")
                .eq("thread_id", thread_id)
                .order("created_at", desc=False)
                .execute()
            )
            all_msgs = msgs_res.data or []

            if body.keep_count < 0:
                raise HTTPException(400, "keep_count must be >= 0")

            # Delete messages beyond keep_count
            to_delete = all_msgs[body.keep_count:]
            if to_delete:
                delete_ids = [m["id"] for m in to_delete]
                supabase.table("messages").delete().in_("id", delete_ids).execute()

            # Optionally update the content of the last kept message
            if body.updated_content is not None and 0 < body.keep_count <= len(all_msgs):
                last_kept = all_msgs[body.keep_count - 1]
                supabase.table("messages").update(
                    {"content": body.updated_content}
                ).eq("id", last_kept["id"]).execute()

            # Update thread timestamp
            now_iso = datetime.now(timezone.utc).isoformat()
            supabase.table("threads").update({"updated_at": now_iso}).eq(
                "id", thread_id
            ).eq("user_id", user_id).execute()

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("thread_truncate_failed")
            raise HTTPException(500, str(exc)) from exc

        return {"ok": True, "kept": min(body.keep_count, len(all_msgs)), "deleted": len(to_delete)}

    return router
