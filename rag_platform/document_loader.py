"""文档入库流水线：加载 → 切片 → 向量化 → 写入 Qdrant。

切分策略：
1. 先按 Markdown 一至四级标题切成语义段；
2. 段内再按 ``CHUNK_SIZE`` 做滑窗，优先在句号/换行处断开；
3. 每个切片携带来源文件名、小节标题与稳定的 chunk_id，便于问答引用溯源。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient, models

from config import Settings
from llm_client import LLMClient

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")
_SUPPORTED_SUFFIXES = {".md", ".txt"}
_UUID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class EmbeddingDimensionError(RuntimeError):
    """向量维度与已有集合不一致。"""


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    source: str
    title: str
    text: str


class DocumentLoader:
    """负责知识库文档的解析、切片与向量入库。"""

    def __init__(self, settings: Settings, llm: LLMClient, qdrant: AsyncQdrantClient) -> None:
        self._settings = settings
        self._llm = llm
        self._qdrant = qdrant

    # ------------------------------------------------------------ 集合管理
    async def ensure_collection(self, recreate: bool = False) -> None:
        """确保集合存在，并校验向量维度；``recreate=True`` 会重建集合。"""
        collection = self._settings.qdrant_collection
        exists = await self._qdrant.collection_exists(collection)

        if exists and not recreate:
            info = await self._qdrant.get_collection(collection)
            size: int | None = None
            vectors = info.config.params.vectors
            if isinstance(vectors, models.VectorParams):
                size = vectors.size
            elif isinstance(vectors, dict) and vectors:
                size = next(iter(vectors.values())).size
            if size is not None and size != self._settings.embed_dim:
                raise EmbeddingDimensionError(
                    f"Qdrant 集合维度为 {size}，但 EMBED_DIM={self._settings.embed_dim}。"
                    "请修改 .env 后调用 POST /api/v1/documents/reindex 重建集合。"
                )
            return

        if exists and recreate:
            logger.warning("重建 Qdrant 集合：{}", collection)
            await self._qdrant.delete_collection(collection)

        await self._qdrant.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=self._settings.embed_dim,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("创建 Qdrant 集合：{}（{} 维，COSINE）", collection, self._settings.embed_dim)

    async def count_chunks(self) -> int:
        try:
            result = await self._qdrant.count(collection_name=self._settings.qdrant_collection, exact=True)
            return int(result.count)
        except Exception as exc:  # noqa: BLE001 - 健康检查不抛错
            logger.warning("统计向量数量失败：{}", exc)
            return 0

    # ------------------------------------------------------------ 文档入库
    async def ingest_file(self, path: Path, source: str | None = None) -> dict[str, Any]:
        """入库单个文档，返回 ``{"source": ..., "chunks": n}``。"""
        text = self._read_text(path)
        source_name = source or self._source_name(path)
        chunks = self.split_document(text, source_name)
        if not chunks:
            raise ValueError(f"文档内容为空或无法切片：{path}")

        vectors = await self._llm.embed_texts([chunk.text for chunk in chunks])
        now = datetime.now(timezone.utc).isoformat()
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_UUID_NAMESPACE, f"{source_name}#{chunk.chunk_id}")),
                vector=vector,
                payload={
                    "source": chunk.source,
                    "title": chunk.title,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "ingested_at": now,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        # 先删同源旧切片，避免文档更新后残留旧版本内容
        await self._qdrant.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source_name),
                        )
                    ]
                )
            ),
            wait=True,
        )
        await self._qdrant.upsert(
            collection_name=self._settings.qdrant_collection,
            points=points,
            wait=True,
        )
        logger.info("入库完成：{}，切片 {} 个", source_name, len(chunks))
        return {"source": source_name, "chunks": len(chunks)}

    async def ingest_directory(self, directory: Path | None = None) -> dict[str, Any]:
        """递归入库目录下全部 md/txt，单文件失败不阻塞整体流程。"""
        target = directory or self._settings.knowledge_dir
        files = sorted(
            path
            for path in target.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        documents = 0
        chunks = 0
        failed: list[dict[str, str]] = []
        for path in files:
            try:
                result = await self.ingest_file(path)
                documents += 1
                chunks += int(result["chunks"])
            except Exception as exc:  # noqa: BLE001 - 汇总失败项返回给调用方
                logger.exception("入库失败：{}", path)
                failed.append({"source": path.name, "error": str(exc)})
        return {"documents": documents, "chunks": chunks, "failed": failed}

    async def reindex_all(self) -> dict[str, Any]:
        """删除并重建集合，然后重新入库内置知识库（可解决维度切换问题）。"""
        await self.ensure_collection(recreate=True)
        return await self.ingest_directory()

    # -------------------------------------------------------------- 切片逻辑
    def split_document(self, text: str, source: str) -> list[Chunk]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        sections: list[tuple[str, str]] = []
        current_title = Path(source).stem
        buffer: list[str] = []
        for line in normalized.split("\n"):
            match = _HEADING_RE.match(line.strip())
            if match:
                if buffer:
                    sections.append((current_title, "\n".join(buffer).strip()))
                current_title = match.group(2).strip()
                buffer = []
            else:
                buffer.append(line)
        if buffer:
            sections.append((current_title, "\n".join(buffer).strip()))

        chunks: list[Chunk] = []
        for title, body in sections:
            if not body:
                continue
            content = f"【{title}】\n{body}"
            for piece in self._sliding_window(content):
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        source=source,
                        title=title,
                        text=piece,
                    )
                )
        return chunks

    def _sliding_window(self, text: str) -> list[str]:
        size = self._settings.chunk_size
        overlap = min(self._settings.chunk_overlap, size - 1)
        if len(text) <= size:
            return [text.strip()] if text.strip() else []

        result: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                window = text[start:end]
                # 尽量在自然边界断开，避免把一句话切成两半
                cut = max(
                    window.rfind("\n"),
                    window.rfind("。"),
                    window.rfind("；"),
                    window.rfind("！"),
                    window.rfind("？"),
                )
                if cut >= int(size * 0.6):
                    end = start + cut + 1
            piece = text[start:end].strip()
            if piece:
                result.append(piece)
            if end >= len(text):
                break
            next_start = end - overlap
            start = next_start if next_start > start else start + 1
        return result

    # ---------------------------------------------------------------- 工具
    def _read_text(self, path: Path) -> str:
        limit = self._settings.max_upload_mb * 1024 * 1024
        if path.stat().st_size > limit:
            raise ValueError(f"文件超过 {self._settings.max_upload_mb} MB 限制：{path.name}")
        return path.read_bytes().decode("utf-8", errors="ignore")

    def _source_name(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self._settings.knowledge_dir.resolve()).as_posix()
        except ValueError:
            return path.name
