from __future__ import annotations

import logging

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

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        if not self.available:
            raise RuntimeError(f"{self.provider_name} API key is not configured.")

        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

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
        if finish_reason == "length":
            logger.warning(
                "llm_output_hit_token_limit",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "max_tokens": max_tokens,
                },
            )
            raise RuntimeError(
                f"{self.provider_name} output hit token limit (max_tokens={max_tokens})."
            )

        content = choice.get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise RuntimeError(f"{self.provider_name} response content is invalid.")
        return content
