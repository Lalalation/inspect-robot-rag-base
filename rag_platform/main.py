"""FastAPI 服务入口。

对外提供文档上传、RAG 问答、Text2SQL 查询与健康检查四类能力。
本项目是独立的运维知识服务，不依赖 ROS2 / rclpy / Gazebo，
后续可由机器人上层 Agent 通过 HTTP 直接调用。
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from qdrant_client import AsyncQdrantClient

from cache import CacheClient
from config import BASE_DIR, Settings, get_settings
from db.init_db import ensure_database
from document_loader import DocumentLoader, EmbeddingDimensionError
from llm_client import LLMClient, LLMError
from retrieval import RAGService
from sql_generator import Text2SQLService

settings = get_settings()


# --------------------------------------------------------------------- 日志
def _setup_logging(config: Settings) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=config.log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        backtrace=False,
        diagnose=False,
    )
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_dir / "app.log"),
        level=config.log_level.upper(),
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


# --------------------------------------------------------------- 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    config = settings
    _setup_logging(config)
    logger.info("启动 {} v{}", config.app_name, config.app_version)

    for directory in (config.knowledge_dir, config.upload_dir, config.sqlite_path.parent, BASE_DIR / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    db_info = ensure_database(config.sqlite_path)
    logger.info("SQLite 就绪：{}，{} 条记录", db_info["path"], db_info["rows"])

    llm = LLMClient(config)
    cache = CacheClient(config)
    qdrant = AsyncQdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
    loader = DocumentLoader(config, llm, qdrant)

    app.state.settings = config
    app.state.llm = llm
    app.state.cache = cache
    app.state.qdrant = qdrant
    app.state.loader = loader
    app.state.rag = RAGService(config, llm, cache, qdrant)
    app.state.sql_service = Text2SQLService(config, llm, cache)

    await cache.ping()

    if not config.llm_api_key:
        logger.warning("未配置 LLM_API_KEY：请在 .env 中填写后再调用问答 / SQL 接口")
    if not config.embed_api_key_resolved:
        logger.warning("未配置 Embedding API Key：请在 .env 中填写后再上传或检索文档")

    try:
        await loader.ensure_collection()
        chunks = await loader.count_chunks()
        if chunks == 0:
            logger.info("向量集合为空，开始自动入库内置知识库……")
            summary = await loader.ingest_directory()
            logger.info(
                "内置知识库入库完成：{} 篇 / {} 个切片 / {} 个失败",
                summary["documents"],
                summary["chunks"],
                len(summary["failed"]),
            )
        else:
            logger.info("向量集合已有 {} 个切片，跳过自动入库", chunks)
    except EmbeddingDimensionError as exc:
        logger.error("向量维度不一致，RAG 暂不可用：{}", exc)
    except Exception as exc:  # noqa: BLE001 - Qdrant 未启动时服务仍可提供健康检查
        logger.warning("Qdrant 初始化失败，RAG 将以降级状态运行：{}", exc)

    yield

    logger.info("关闭服务，释放连接")
    await llm.close()
    await cache.close()
    await qdrant.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "面向巡检机器人运维人员的 RAG 问答与 Text2SQL 数据查询服务。"
        "独立部署，不包含 ROS2、rclpy、Gazebo 相关代码，"
        "可作为机器人上层 Agent 的知识库服务。"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端静态资源与运维控制台页面（与 API 同源，无需额外 CORS 配置）
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------- 模型
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, examples=["电机过温应该怎么处理？"])
    top_k: int | None = Field(default=None, ge=1, le=10, description="召回切片数量，默认取配置值")

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value


class SQLQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=300, examples=["最近24小时电量低于20%的记录"])

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value


# --------------------------------------------------------------------- 中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info("{} {} -> {} ({}ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# --------------------------------------------------------------- 异常处理
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    logger.error("模型服务异常：{}", exc)
    return JSONResponse(status_code=503, content={"detail": f"模型服务暂不可用：{exc}"})


@app.exception_handler(EmbeddingDimensionError)
async def embedding_error_handler(request: Request, exc: EmbeddingDimensionError) -> JSONResponse:
    logger.error("向量维度异常：{}", exc)
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# --------------------------------------------------------------------- 路由
@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """返回运维控制台页面。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
async def health(request: Request) -> dict[str, Any]:
    config: Settings = request.app.state.settings
    llm_info = await request.app.state.llm.health()
    redis_ok = await request.app.state.cache.ping()

    qdrant_info: dict[str, Any] = {
        "status": "down",
        "collection": config.qdrant_collection,
        "chunks": 0,
    }
    try:
        exists = await request.app.state.qdrant.collection_exists(config.qdrant_collection)
        qdrant_info = {
            "status": "up" if exists else "down",
            "collection": config.qdrant_collection,
            "chunks": await request.app.state.loader.count_chunks() if exists else 0,
        }
    except Exception as exc:  # noqa: BLE001
        qdrant_info["detail"] = type(exc).__name__

    sqlite_info = await asyncio.to_thread(_sqlite_health, config.sqlite_readonly_uri)
    healthy = (
        qdrant_info["status"] == "up"
        and redis_ok
        and llm_info["status"] == "up"
        and sqlite_info["status"] == "up"
    )
    return {
        "status": "ok" if healthy else "degraded",
        "version": config.app_version,
        "qdrant": qdrant_info,
        "redis": {"status": "up" if redis_ok else "degraded", "backend": request.app.state.cache.backend},
        "llm": llm_info,
        "sqlite": sqlite_info,
    }


@app.post(f"{settings.api_prefix}/documents/upload", tags=["documents"])
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """上传 md/txt 文档并立即切片入库。"""
    config: Settings = request.app.state.settings
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in {".md", ".txt"}:
        raise HTTPException(status_code=400, detail="仅支持 .md 或 .txt 文档")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > config.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件不得超过 {config.max_upload_mb} MB")

    target = config.upload_dir / filename
    await asyncio.to_thread(target.write_bytes, content)

    try:
        result = await request.app.state.loader.ingest_file(target)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("文档入库失败：{}", filename)
        raise HTTPException(status_code=500, detail=f"文档入库失败：{exc}") from exc

    return {"ingested": True, **result}


@app.post(f"{settings.api_prefix}/documents/reindex", tags=["documents"])
async def reindex_documents(request: Request) -> dict[str, Any]:
    """重建向量集合并重新入库内置知识库。"""
    try:
        summary = await request.app.state.loader.reindex_all()
    except (LLMError, EmbeddingDimensionError):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("重建知识库失败")
        raise HTTPException(status_code=503, detail=f"重建知识库失败：{exc}") from exc
    return {"reindexed": True, **summary}


@app.post(f"{settings.api_prefix}/rag/ask", tags=["rag"])
async def rag_ask(request: Request, payload: AskRequest) -> dict[str, Any]:
    """基于内置运维文档回答问题，返回严格 JSON 结构。"""
    result = await request.app.state.rag.ask(payload.question, payload.top_k)
    if result.get("degraded"):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post(f"{settings.api_prefix}/sql/query", tags=["sql"])
async def sql_query(request: Request, payload: SQLQueryRequest) -> dict[str, Any]:
    """自然语言转只读 SQL 并执行，返回 SQL、解释与结果。"""
    result = await request.app.state.sql_service.query(payload.question)
    if result.get("degraded"):
        raise HTTPException(status_code=503, detail=result)
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result)
    return result


# --------------------------------------------------------------------- 工具
def _sqlite_health(readonly_uri: str) -> dict[str, Any]:
    try:
        connection = sqlite3.connect(readonly_uri, uri=True, timeout=2)
        try:
            rows = connection.execute("SELECT COUNT(*) FROM robot_run_record").fetchone()[0]
        finally:
            connection.close()
        return {"status": "up", "rows": int(rows)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "down", "detail": type(exc).__name__}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=False)
