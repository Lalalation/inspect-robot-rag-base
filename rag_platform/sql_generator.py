"""Text2SQL 服务。

LangGraph 状态图：

    START → generate → validate ─┬─(通过)→ execute → finalize → END
                                  ├─(失败且可重试)→ regenerate → validate
                                  └─(失败且重试耗尽)→ finalize → END

安全三道防线：
1. 提示词只允许单条 SELECT；
2. 静态校验：单语句、禁注释、关键字黑名单、表白名单、禁 SELECT *；
3. 执行层：SQLite ``mode=ro`` 只读 URI + 查询超时 + 行数截断。
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from cache import CacheClient, cache_key
from config import Settings
from llm_client import LLMClient, LLMError, LLMJSONError
from llm_prompts import TEXT2SQL_SYSTEM_PROMPT, build_text2sql_user_prompt

_ALLOWED_TABLE = "robot_run_record"

_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "grant",
    "revoke",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "sqlite_master",
    "sqlite_schema",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
_TABLE_REF_RE = re.compile(r"\b(?:from|join)\s+([A-Za-z0-9_.\"`\[\]]+)", re.IGNORECASE)
_CTE_NAME_RE = re.compile(r"(?:\bwith\b|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(", re.IGNORECASE)
_SELECT_STAR_RE = re.compile(r"\bselect\s+\*", re.IGNORECASE)
_STRING_LITERAL_RE = re.compile(r"'(?:''|[^'])*'")

_MAX_EXPLAIN_CHARS = 160


class SQLValidationError(ValueError):
    """生成的 SQL 未通过静态安全校验。"""


def _mask_string_literals(sql: str) -> str:
    """把字符串常量替换为空串，避免字面量里的关键字造成误判。"""
    return _STRING_LITERAL_RE.sub("''", sql)


def validate_sql(sql: str, max_rows: int) -> str:
    """校验并规范化 SQL，返回可执行的只读语句。

    校验不通过时抛出 :class:`SQLValidationError`。
    """
    normalized = (sql or "").strip()
    if not normalized:
        raise SQLValidationError("模型未生成 SQL")

    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    if ";" in normalized:
        raise SQLValidationError("禁止多语句")

    masked = _mask_string_literals(normalized)

    if "--" in masked or "/*" in masked or "*/" in masked:
        raise SQLValidationError("禁止 SQL 注释")

    lowered = masked.lower().lstrip()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise SQLValidationError("只允许 SELECT 查询")

    forbidden = _FORBIDDEN_RE.search(masked)
    if forbidden:
        raise SQLValidationError(f"命中高危关键字：{forbidden.group(0).upper()}")

    if _SELECT_STAR_RE.search(masked):
        raise SQLValidationError("禁止 SELECT *，必须显式列出列名")

    table_refs = _TABLE_REF_RE.findall(masked)
    if not table_refs:
        raise SQLValidationError(f"查询必须引用数据表 {_ALLOWED_TABLE}")
    cte_names = {name.lower() for name in _CTE_NAME_RE.findall(masked)}
    for ref in table_refs:
        table = ref.strip('"`[]').split(".")[-1].lower()
        if table != _ALLOWED_TABLE and table not in cte_names:
            raise SQLValidationError(f"禁止访问非白名单表：{ref}")

    if "limit" not in lowered:
        normalized = f"{normalized} LIMIT {max_rows}"
    return normalized


class SQLState(TypedDict, total=False):
    """Text2SQL 状态图的数据结构。"""

    question: str
    candidate_sql: str
    sql: str
    explain: str
    validation_error: str | None
    retry_count: int
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    degraded: bool
    error: str | None


class Text2SQLService:
    """自然语言 → SQL → 校验 → 只读执行。"""

    def __init__(self, settings: Settings, llm: LLMClient, cache: CacheClient) -> None:
        self._settings = settings
        self._llm = llm
        self._cache = cache
        self._graph = self._build_graph()

    async def query(self, question: str) -> dict[str, Any]:
        """对外查询入口。"""
        started = time.perf_counter()
        key = cache_key("sql", question)

        cached = await self._cache.get_json(key)
        if cached is not None:
            return {
                **cached,
                "cached": True,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }

        state: SQLState = {
            "question": question,
            "candidate_sql": "",
            "sql": "",
            "explain": "",
            "validation_error": None,
            "retry_count": 0,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "degraded": False,
            "error": None,
        }
        result = await self._graph.ainvoke(state)
        response = self._to_response(result)
        response["cached"] = False
        response["latency_ms"] = int((time.perf_counter() - started) * 1000)

        if response.get("sql") and not response.get("error") and not response.get("degraded"):
            await self._cache.set_json(
                key,
                {
                    "sql": response["sql"],
                    "explain": response["explain"],
                    "columns": response["columns"],
                    "rows": response["rows"],
                    "row_count": response["row_count"],
                    "truncated": response["truncated"],
                },
            )
        logger.info("Text2SQL 完成：rows={} 耗时={}ms", response["row_count"], response["latency_ms"])
        return response

    # ------------------------------------------------------------ 状态图构建
    def _build_graph(self) -> Any:
        builder = StateGraph(SQLState)
        builder.add_node("generate", self._generate)
        builder.add_node("validate", self._validate)
        builder.add_node("regenerate", self._regenerate)
        builder.add_node("execute", self._execute)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "generate")
        builder.add_edge("generate", "validate")
        builder.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {"execute": "execute", "regenerate": "regenerate", "finalize": "finalize"},
        )
        builder.add_edge("regenerate", "validate")
        builder.add_edge("execute", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    # ------------------------------------------------------------------ 节点
    async def _generate(self, state: SQLState) -> dict[str, Any]:
        return await self._call_model(state["question"], None, retry_count=0)

    async def _regenerate(self, state: SQLState) -> dict[str, Any]:
        return await self._call_model(
            state["question"],
            state.get("validation_error"),
            retry_count=int(state.get("retry_count", 0)) + 1,
        )

    async def _call_model(
        self,
        question: str,
        validation_error: str | None,
        retry_count: int,
    ) -> dict[str, Any]:
        prompt = build_text2sql_user_prompt(question, datetime.now(), validation_error)
        try:
            raw = await self._llm.chat_json(TEXT2SQL_SYSTEM_PROMPT, prompt, max_tokens=500)
        except LLMJSONError as exc:
            logger.warning("Text2SQL JSON 解析失败：{}", exc)
            return self._failure("JSONParseError", f"模型输出不是合法 JSON：{exc}")
        except LLMError as exc:
            logger.warning("Text2SQL 模型调用失败：{}", exc)
            return self._failure("LLMError", str(exc))

        sql = raw.get("sql")
        explain = raw.get("explain")
        explain_text = explain.strip() if isinstance(explain, str) else ""
        if not isinstance(sql, str) or not sql.strip():
            if explain_text:
                # 模型按提示词主动拒绝（写操作 / 无关问题）：属于请求被拦截，不是服务降级
                return {
                    "candidate_sql": "",
                    "sql": "",
                    "explain": explain_text[:_MAX_EXPLAIN_CHARS],
                    "retry_count": 99,
                    "validation_error": explain_text,
                    "degraded": False,
                    "error": f"请求被拒绝：{explain_text}"[:240],
                }
            return self._failure("SchemaError", "模型返回缺少 sql 字段")
        return {
            "candidate_sql": sql.strip(),
            "explain": explain_text[:_MAX_EXPLAIN_CHARS],
            "retry_count": retry_count,
            "validation_error": None,
            "degraded": False,
            "error": None,
        }

    def _failure(self, kind: str, message: str) -> dict[str, Any]:
        return {
            "candidate_sql": "",
            "sql": "",
            "explain": "",
            "retry_count": 99,
            "degraded": True,
            "error": f"{kind}: {message}"[:240],
        }

    async def _validate(self, state: SQLState) -> dict[str, Any]:
        # 模型调用层已失败时直接放行，避免把真实错误覆盖成「模型未生成 SQL」
        if state.get("degraded") or not (state.get("candidate_sql") or "").strip():
            return {}
        candidate = state.get("candidate_sql", "")
        try:
            safe_sql = validate_sql(candidate, self._settings.sql_max_rows)
        except SQLValidationError as exc:
            logger.warning("SQL 校验未通过：{}；candidate={}", exc, candidate[:160])
            return {"sql": "", "validation_error": str(exc), "error": str(exc)}
        return {"sql": safe_sql, "validation_error": None, "error": None}

    def _route_after_validate(self, state: SQLState) -> str:
        if state.get("sql"):
            return "execute"
        if state.get("degraded"):
            return "finalize"
        if int(state.get("retry_count", 0)) < 1:
            return "regenerate"
        return "finalize"

    async def _execute(self, state: SQLState) -> dict[str, Any]:
        sql = state.get("sql", "")
        if not sql:
            return {}
        try:
            columns, rows, truncated = await asyncio.to_thread(self._run_query, sql)
        except sqlite3.Error as exc:
            message = "查询超时" if "interrupted" in str(exc).lower() else str(exc)
            logger.warning("SQL 执行失败：{}；sql={}", message, sql[:160])
            return {"columns": [], "rows": [], "row_count": 0, "truncated": False, "error": f"SQL 执行失败：{message}"}
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "error": None,
        }

    async def _finalize(self, state: SQLState) -> dict[str, Any]:
        return {}

    # -------------------------------------------------------------- 执行层
    def _run_query(self, sql: str) -> tuple[list[str], list[dict[str, Any]], bool]:
        """在线程中执行只读查询，避免阻塞事件循环。"""
        deadline = time.monotonic() + self._settings.sql_timeout_seconds
        connection = sqlite3.connect(
            self._settings.sqlite_readonly_uri,
            uri=True,
            timeout=self._settings.sql_timeout_seconds,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.set_progress_handler(
                lambda: int(time.monotonic() > deadline),
                10_000,
            )
            cursor = connection.execute(sql)  # sqlite3 本身也拒绝多语句
            columns = [item[0] for item in cursor.description or []]
            fetched = cursor.fetchmany(self._settings.sql_max_rows + 1)
            truncated = len(fetched) > self._settings.sql_max_rows
            rows = [dict(row) for row in fetched[: self._settings.sql_max_rows]]
            return columns, rows, truncated
        finally:
            connection.close()

    # ------------------------------------------------------------------ 输出
    @staticmethod
    def _to_response(state: SQLState) -> dict[str, Any]:
        response = {
            "sql": state.get("sql") or state.get("candidate_sql") or "",
            "explain": state.get("explain") or "",
            "columns": state.get("columns") or [],
            "rows": state.get("rows") or [],
            "row_count": int(state.get("row_count", 0)),
            "truncated": bool(state.get("truncated", False)),
            "degraded": bool(state.get("degraded", False)),
        }
        if state.get("error"):
            response["error"] = str(state["error"])[:240]
        return response
