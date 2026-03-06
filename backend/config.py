from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    log_level: str = "INFO"

    orchestrator_provider: str = "gemini"
    orchestrator_model: str = "gemini-3.1-pro"
    orchestrator_api_key: str = ""
    orchestrator_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    orchestrator_reasoning_effort: str = ""  # Gemini only: low, medium, high

    search_provider: str = "openrouter"
    search_model: str = "moonshotai/kimi-k2.5"
    search_api_key: str = ""
    search_base_url: str = "https://openrouter.ai/api/v1"
    search_decompose_max_tokens: int = 1200
    search_reasoning_effort: str = ""  # Gemini only: low, medium, high

    tool_writer_provider: str = ""
    tool_writer_model: str = ""
    tool_writer_api_key: str = ""
    tool_writer_base_url: str = ""
    tool_writer_reasoning_effort: str = ""  # Gemini only: low, medium, high

    agentic_search_enabled: bool = True
    recursive_retrieval_enabled: bool = False
    embeddings_enabled: bool = False
    max_retrieval_iters: int = 3
    top_k_clauses: int = 20

    default_steel_grade: str = "S355"
    default_section_name: str = "IPE300"
    default_gamma_m0: float = 1.0
    default_med_knm: float = 120.0
    default_ned_kn: float = 200.0

    document_registry_path: Path | None = None
    tool_registry_path: Path | None = None
    orchestrator_thread_log_path: Path | None = None

    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_service_role_key: str = ""

    @property
    def auth_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def resolved_document_registry_path(self) -> Path:
        if self.document_registry_path:
            return self.document_registry_path
        return self.project_root / "data" / "document_registry.json"

    @property
    def resolved_tool_registry_path(self) -> Path:
        if self.tool_registry_path:
            return self.tool_registry_path
        return self.project_root / "tools" / "tool_registry.json"

    @property
    def resolved_orchestrator_thread_log_path(self) -> Path:
        if self.orchestrator_thread_log_path:
            return self.orchestrator_thread_log_path
        return self.project_root / "logs" / "orchestrator_threads.json"

    @classmethod
    def load(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[1]
        return cls(
            project_root=project_root,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            orchestrator_provider=os.getenv("ORCHESTRATOR_PROVIDER", "gemini"),
            orchestrator_model=os.getenv("ORCHESTRATOR_MODEL", "gemini-3.1-pro"),
            orchestrator_api_key=os.getenv("ORCHESTRATOR_API_KEY", ""),
            orchestrator_base_url=os.getenv(
                "ORCHESTRATOR_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            orchestrator_reasoning_effort=(
                os.getenv("ORCHESTRATOR_REASONING_EFFORT", "").strip() or ""
            ),
            search_provider=os.getenv("SEARCH_PROVIDER", "openrouter"),
            search_model=os.getenv("SEARCH_MODEL", "moonshotai/kimi-k2.5"),
            search_api_key=os.getenv("SEARCH_API_KEY", ""),
            search_base_url=os.getenv("SEARCH_BASE_URL", "https://openrouter.ai/api/v1"),
            search_decompose_max_tokens=_to_int(
                os.getenv("SEARCH_DECOMPOSE_MAX_TOKENS"), 1200
            ),
            search_reasoning_effort=(
                os.getenv("SEARCH_REASONING_EFFORT", "").strip() or ""
            ),
            tool_writer_provider=os.getenv("TOOL_WRITER_PROVIDER", ""),
            tool_writer_model=os.getenv("TOOL_WRITER_MODEL", ""),
            tool_writer_api_key=os.getenv("TOOL_WRITER_API_KEY", ""),
            tool_writer_base_url=os.getenv("TOOL_WRITER_BASE_URL", ""),
            tool_writer_reasoning_effort=(
                os.getenv("TOOL_WRITER_REASONING_EFFORT", "").strip() or ""
            ),
            agentic_search_enabled=_to_bool(os.getenv("AGENTIC_SEARCH_ENABLED"), True),
            recursive_retrieval_enabled=_to_bool(
                os.getenv("RECURSIVE_RETRIEVAL_ENABLED"), False
            ),
            embeddings_enabled=_to_bool(os.getenv("EMBEDDINGS_ENABLED"), False),
            max_retrieval_iters=_to_int(os.getenv("MAX_RETRIEVAL_ITERS"), 3),
            top_k_clauses=_to_int(os.getenv("TOP_K_CLAUSES"), 6),
            default_steel_grade=os.getenv("DEFAULT_STEEL_GRADE", "S355"),
            default_section_name=os.getenv("DEFAULT_SECTION_NAME", "IPE300"),
            default_gamma_m0=float(os.getenv("DEFAULT_GAMMA_M0", "1.0")),
            default_med_knm=float(os.getenv("DEFAULT_MED_KNM", "120.0")),
            default_ned_kn=float(os.getenv("DEFAULT_NED_KN", "200.0")),
            document_registry_path=(
                Path(os.environ["DOCUMENT_REGISTRY_PATH"])
                if os.getenv("DOCUMENT_REGISTRY_PATH")
                else None
            ),
            tool_registry_path=(
                Path(os.environ["TOOL_REGISTRY_PATH"])
                if os.getenv("TOOL_REGISTRY_PATH")
                else None
            ),
            orchestrator_thread_log_path=(
                Path(os.environ["ORCHESTRATOR_THREAD_LOG_PATH"])
                if os.getenv("ORCHESTRATOR_THREAD_LOG_PATH")
                else None
            ),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            supabase_jwt_secret=os.getenv("SUPABASE_JWT_SECRET", ""),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        )

    def with_overrides(self, **kwargs: object) -> "Settings":
        return replace(self, **kwargs)
