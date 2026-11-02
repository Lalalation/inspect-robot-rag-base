"""RAG 检索问答服务。

用 LangGraph 组织成一张四节点状态图：

    START → retrieve ─┬─(有召回)→ generate → parse → finalize → END
                      └─(无召回/检索异常)→ finalize

这样「召回为空」不会浪费一次 LLM 调用，同时每条分支都可单独测试。
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger
from qdrant_client import AsyncQdrantClient

from cache import CacheClient, cache_key
from config import Settings
from llm_client import LLMClient, LLMError, LLMJSONError
from llm_prompts import NO_DATA_ANSWER, RAG_SYSTEM_PROMPT, build_rag_user_prompt

_RETRIEVAL_FAILED_ANSWER = "检索服务暂不可用，请稍后重试。"
_MODEL_FAILED_ANSWER = "模型服务暂不可用，请稍后重试。"
_MAX_ANSWER_CHARS = 1200


class RAGState(TypedDict, total=False):
    """状态图在节点之间传递的数据。"""

    question: str
    top_k: int
    contexts: list[dict[str, Any]]
    raw: dict[str, Any]
    found: bool
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    degraded: bool
    error: str | None


class RAGService:
    """检索 + 生成 + 校验 + 缓存。"""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        cache: CacheClient,
        qdrant: AsyncQdrantClient,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._cache = cache
        self._qdrant = qdrant
        self._graph = self._build_graph()

    async def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """对外问答入口：命中缓存直接返回，否则执行状态图。"""
        started = time.perf_counter()
        effective_top_k = top_k or self._settings.top_k
        key = cache_key("rag", question, top_k=effective_top_k)

        cached = await self._cache.get_json(key)
        if cached is not None:
            return {
                **cached,
                "cached": True,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }

        state: RAGState = {
            "question": question,
            "top_k": effective_top_k,
            "contexts": [],
            "found": False,
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "degraded": False,
            "error": None,
        }
        result = await self._graph.ainvoke(state)
        response = self._to_response(result)
        response["cached"] = False
        response["latency_ms"] = int((time.perf_counter() - started) * 1000)

        if not response["degraded"]:
            await self._cache.set_json(
                key,
                {
                    "found": response["found"],
                    "answer": response["answer"],
                    "sources": response["sources"],
                    "confidence": response["confidence"],
                },
            )
        logger.info(
            "RAG 问答完成：found={} cached=False 耗时={}ms",
            response["found"],
            response["latency_ms"],
        )
        return response

    # ------------------------------------------------------------ 状态图构建
    def _build_graph(self) -> Any:
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("parse", self._parse)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"generate": "generate", "finalize": "finalize"},
        )
        builder.add_edge("generate", "parse")
        builder.add_edge("parse", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    # ------------------------------------------------------------------ 节点
    async def _retrieve(self, state: RAGState) -> dict[str, Any]:
        """向量化问题并召回 Top-K 切片。"""
        question = state["question"]
        top_k = state.get("top_k") or self._settings.top_k
        try:
            vector = (await self._llm.embed_texts([question]))[0]
            response = await self._qdrant.query_points(
                collection_name=self._settings.qdrant_collection,
                query=vector,
                limit=top_k,
                score_threshold=self._settings.score_threshold,
                with_payload=True,
            )
            hits = response.points
        except Exception as exc:  # noqa: BLE001 - 降级返回统一文案
            logger.warning("RAG 检索失败，降级返回：{}: {}", type(exc).__name__, exc)
            return {
                "contexts": [],
                "found": False,
                "answer": _RETRIEVAL_FAILED_ANSWER,
                "sources": [],
                "confidence": 0.0,
                "degraded": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

        contexts: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            contexts.append(
                {
                    "chunk_id": payload.get("chunk_id", -1),
                    "source": payload.get("source", "unknown"),
                    "title": payload.get("title", ""),
                    "text": payload.get("text", ""),
                    "score": float(hit.score),
                }
            )

        if not contexts:
            logger.info("RAG 无召回：question={!r}", question[:80])
            return {
                "contexts": [],
                "found": False,
                "answer": NO_DATA_ANSWER,
                "sources": [],
                "confidence": 0.0,
                "degraded": False,
                "error": None,
            }
        return {"contexts": contexts, "degraded": False, "error": None}

    def _route_after_retrieve(self, state: RAGState) -> str:
        if state.get("degraded") or not state.get("contexts"):
            return "finalize"
        return "generate"

    async def _generate(self, state: RAGState) -> dict[str, Any]:
        """把召回内容交给模型，要求严格 JSON。"""
        prompt = build_rag_user_prompt(state["question"], state["contexts"])
        try:
            raw = await self._llm.chat_json(RAG_SYSTEM_PROMPT, prompt, max_tokens=700)
        except LLMJSONError as exc:
            logger.warning("RAG 输出 JSON 解析失败：{}", exc)
            return {
                "found": False,
                "answer": _MODEL_FAILED_ANSWER,
                "sources": [],
                "confidence": 0.0,
                "degraded": True,
                "error": f"JSONParseError: {exc}",
            }
        except LLMError as exc:
            logger.warning("RAG 模型调用失败：{}", exc)
            return {
                "found": False,
                "answer": _MODEL_FAILED_ANSWER,
                "sources": [],
                "confidence": 0.0,
                "degraded": True,
                "error": str(exc)[:200],
            }
        return {"raw": raw}

    async def _parse(self, state: RAGState) -> dict[str, Any]:
        """校验模型 JSON：字段类型、引用真实性、答案兜底。"""
        raw = state.get("raw") or {}
        found = bool(raw.get("found", False))
        answer = str(raw.get("answer", "")).strip()

        if not found:
            return {"found": False, "answer": NO_DATA_ANSWER, "sources": [], "confidence": 0.0}

        if not answer:
            return {
                "found": False,
                "answer": NO_DATA_ANSWER,
                "sources": [],
                "confidence": 0.0,
                "degraded": True,
                "error": "模型返回空答案",
            }

        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[:_MAX_ANSWER_CHARS] + "…"

        # 只保留召回切片中真实存在的引用，避免模型编造来源。
        # 注意 chunk_id 只在单个文档内唯一，必须同时匹配 source，否则不同文档的同号切片会串用。
        valid_pairs: dict[tuple[str, int], dict[str, Any]] = {}
        by_chunk: dict[int, list[dict[str, Any]]] = {}
        for context in state.get("contexts", []):
            try:
                context_chunk_id = int(context.get("chunk_id", -1))
            except (TypeError, ValueError):
                continue
            valid_pairs[(str(context.get("source", "")), context_chunk_id)] = context
            by_chunk.setdefault(context_chunk_id, []).append(context)

        sources: list[dict[str, Any]] = []
        raw_sources = raw.get("sources")
        if isinstance(raw_sources, list):
            for item in raw_sources:
                if not isinstance(item, dict):
                    continue
                chunk_id = item.get("chunk_id")
                if isinstance(chunk_id, str) and chunk_id.isdigit():
                    chunk_id = int(chunk_id)
                if not isinstance(chunk_id, int):
                    continue
                source_name = str(item.get("source") or "").strip()
                if source_name:
                    context = valid_pairs.get((source_name, chunk_id))
                else:
                    # 模型未给 source 时，只有该 chunk_id 在本次召回中唯一才允许引用
                    candidates = by_chunk.get(chunk_id) or []
                    context = candidates[0] if len(candidates) == 1 else None
                if context is None:
                    continue
                sources.append(
                    {
                        "source": context["source"],
                        "title": context["title"],
                        "chunk_id": chunk_id,
                        "score": round(float(context["score"]), 4),
                    }
                )
                if len(sources) >= 3:
                    break

        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "found": True,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "degraded": False,
            "error": None,
        }

    async def _finalize(self, state: RAGState) -> dict[str, Any]:
        # 兜底字段，保证任何分支的响应结构一致
        return {
            "found": bool(state.get("found", False)),
            "answer": state.get("answer") or NO_DATA_ANSWER,
            "sources": state.get("sources") or [],
            "confidence": float(state.get("confidence", 0.0)),
        }

    # ------------------------------------------------------------------ 输出
    @staticmethod
    def _to_response(state: RAGState) -> dict[str, Any]:
        response = {
            "found": bool(state.get("found", False)),
            "answer": state.get("answer") or NO_DATA_ANSWER,
            "sources": state.get("sources") or [],
            "confidence": float(state.get("confidence", 0.0)),
            "degraded": bool(state.get("degraded", False)),
        }
        if state.get("error"):
            response["error"] = str(state["error"])[:240]
        return response
