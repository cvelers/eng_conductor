from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
    ) -> None:
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _call_chat_completions(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4000,
        reasoning_effort: str | None = None,
    ) -> str:
        if not self.available:
            raise RuntimeError(f"{self.provider_name} API key is not configured.")

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        # For thinking models (Gemini 3.x, OpenAI o-series) this controls
        # how many tokens the model spends on internal reasoning.  "low"
        # keeps the thinking budget small so more of max_tokens is available
        # for the actual output — critical for lightweight calls like
        # classification and greetings.
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"{self.provider_name} returned no choices.")

        choice = choices[0]
        finish_reason = str(choice.get("finish_reason", "")).strip().lower()

        content = choice.get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise RuntimeError(f"{self.provider_name} response content is invalid.")

        # Log token usage so we can see the thinking-vs-output split.
        usage = data.get("usage", {})
        logger.info(
            "llm_call_complete",
            extra={
                "provider": self.provider_name,
                "model": self.model,
                "finish_reason": finish_reason,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "content_len": len(content),
                "content_preview": content[:100],
            },
        )

        if finish_reason == "length":
            logger.warning(
                "llm_output_truncated",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "content_preview": content[:120],
                },
            )

        return content

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        reasoning_effort: str | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def generate_multimodal(
        self,
        *,
        system_prompt: str,
        content_parts: list[dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        reasoning_effort: str | None = None,
    ) -> str:
        """Generate with multimodal content (text + images via OpenAI vision API)."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]
        return self._call_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
