"""项目运行配置。

所有可调参数集中在此处，通过环境变量或 `.env` 覆盖默认值。
模型侧统一走 OpenAI 兼容接口，默认预留通义千问地址，API Key 由 `.env` 注入。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """平台配置项（环境变量名与字段名一致，大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ 服务
    app_name: str = "Robot Ops RAG & Text2SQL Platform"
    app_version: str = "1.0.0"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    cors_origins: str = "*"  # 逗号分隔；演示环境允许全部来源

    # ------------------------------------------------------------------ LLM
    # 任意 OpenAI 兼容服务（DeepSeek、通义千问、OpenAI、硅基流动等）
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""
    chat_model: str = "qwen-plus"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # ------------------------------------------------------------- Embedding
    embed_base_url: str = ""  # 留空时复用 LLM_BASE_URL
    embed_api_key: str = ""  # 留空时复用 LLM_API_KEY
    embed_model: str = "text-embedding-v3"
    embed_dim: int = Field(default=1024, gt=0)  # 必须与向量模型输出维度一致
    embed_batch_size: int = Field(default=16, ge=1, le=128)

    # ---------------------------------------------------------------- Qdrant
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "robot_ops_docs"

    # -------------------------------------------------------------- 检索参数
    chunk_size: int = Field(default=480, ge=120, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=600)
    top_k: int = Field(default=4, ge=1, le=10)
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # ----------------------------------------------------------------- Redis
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_connect_timeout_seconds: float = Field(default=1.0, gt=0)
    cache_ttl_seconds: int = Field(default=600, ge=1)
    cache_max_memory_entries: int = Field(default=256, ge=1)

    # ---------------------------------------------------------------- SQLite
    sqlite_path: Path = Path("db/robot_data.db")
    sql_max_rows: int = Field(default=200, ge=1, le=1000)
    sql_timeout_seconds: float = Field(default=5.0, gt=0)

    # ------------------------------------------------------------------ 文档
    knowledge_dir: Path = Path("knowledge_docs")
    upload_dir: Path = Path("knowledge_docs/uploaded")
    max_upload_mb: int = Field(default=2, ge=1, le=20)

    @field_validator("sqlite_path", "knowledge_dir", "upload_dir", mode="after")
    @classmethod
    def _to_absolute(cls, value: Path) -> Path:
        """把相对路径统一解析到 rag_platform/ 目录下，避免受启动目录影响。"""
        return value if value.is_absolute() else (BASE_DIR / value)

    # ---------------------------------------------------------------- 派生值
    @property
    def embed_base_url_resolved(self) -> str:
        """Embedding 服务地址：优先专用配置，其次复用 LLM 地址。"""
        return (self.embed_base_url or self.llm_base_url).rstrip("/")

    @property
    def embed_api_key_resolved(self) -> str:
        return self.embed_api_key or self.llm_api_key

    @property
    def sqlite_readonly_uri(self) -> str:
        """SQLite 只读 URI，执行层只允许读，物理上拒绝写入。"""
        return f"file:{self.sqlite_path.as_posix()}?mode=ro"

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [item.strip() for item in self.cors_origins.split(",") if item.strip()]
        return origins or ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例配置，便于 FastAPI 依赖注入与测试覆盖。"""
    return Settings()
