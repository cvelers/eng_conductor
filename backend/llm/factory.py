from __future__ import annotations

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.llm.gemini_provider import GeminiProvider
from backend.llm.mock_provider import MockProvider
from backend.llm.openrouter_provider import OpenRouterProvider


def get_orchestrator_provider(settings: Settings) -> LLMProvider:
    provider = settings.orchestrator_provider.lower()
    if provider == "gemini":
        return GeminiProvider(
            api_key=settings.orchestrator_api_key,
            model=settings.orchestrator_model,
            base_url=settings.orchestrator_base_url,
            default_reasoning_effort=settings.orchestrator_reasoning_effort or None,
        )
    if provider == "openrouter":
        return OpenRouterProvider(
            api_key=settings.orchestrator_api_key,
            model=settings.orchestrator_model,
            base_url=settings.orchestrator_base_url,
        )
    if provider == "mock":
        return MockProvider()
    raise ValueError(f"Unsupported orchestrator provider: {settings.orchestrator_provider}")


def get_tool_writer_provider(settings: Settings) -> LLMProvider:
    """Separate LLM for tool generation. Falls back to orchestrator config if unset."""
    provider = (settings.tool_writer_provider or settings.orchestrator_provider).lower()
    model = settings.tool_writer_model or settings.orchestrator_model
    api_key = settings.tool_writer_api_key or settings.orchestrator_api_key
    base_url = settings.tool_writer_base_url or settings.orchestrator_base_url

    if provider == "gemini":
        effort = (
            settings.tool_writer_reasoning_effort
            or settings.orchestrator_reasoning_effort
        ) or None
        return GeminiProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            default_reasoning_effort=effort,
        )
    if provider == "openrouter":
        return OpenRouterProvider(api_key=api_key, model=model, base_url=base_url)
    if provider == "mock":
        return MockProvider()
    raise ValueError(f"Unsupported tool writer provider: {provider}")


def get_search_provider(settings: Settings) -> LLMProvider:
    provider = settings.search_provider.lower()
    if provider == "openrouter":
        return OpenRouterProvider(
            api_key=settings.search_api_key,
            model=settings.search_model,
            base_url=settings.search_base_url,
        )
    if provider == "gemini":
        return GeminiProvider(
            api_key=settings.search_api_key,
            model=settings.search_model,
            base_url=settings.search_base_url,
            default_reasoning_effort=settings.search_reasoning_effort or None,
        )
    if provider == "mock":
        return MockProvider()
    raise ValueError(f"Unsupported search provider: {settings.search_provider}")
