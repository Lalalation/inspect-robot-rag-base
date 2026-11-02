"""问答与 SQL 结果缓存。

优先使用 Redis；Redis 不可用时自动降级为进程内 TTL 缓存，
保证演示环境下即使没启动 Redis，接口也不会 500。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

from config import Settings

#: Redis 故障后 30 秒再尝试重连，避免每个请求都等待连接超时。
_RETRY_INTERVAL_SECONDS = 30.0


def cache_key(module: str, question: str, **extra: Any) -> str:
    """生成稳定缓存键：模块 + 归一化问题 + 额外参数。"""
    payload = json.dumps(
        {"module": module, "question": question.strip().lower(), **extra},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"robot_ops:{module}:{digest}"


class CacheClient:
    """带自动降级的 JSON 缓存客户端。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=settings.redis_connect_timeout_seconds,
            socket_timeout=settings.redis_connect_timeout_seconds,
        )
        self._memory: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._redis_ok: bool | None = None
        self._retry_at: float = 0.0

    async def ping(self) -> bool:
        try:
            pong = await self._redis.ping()
            if self._redis_ok is not True:
                logger.info("Redis 连接成功：{}", self._settings.redis_url)
            self._redis_ok = True
            return bool(pong)
        except Exception as exc:  # noqa: BLE001 - 降级是预期行为
            if self._redis_ok is not False:
                logger.warning("Redis 不可用，缓存降级为进程内 TTL 缓存：{}", exc)
            self._redis_ok = False
            self._retry_at = time.monotonic() + _RETRY_INTERVAL_SECONDS
            return False

    async def get_json(self, key: str) -> dict[str, Any] | None:
        if self._redis_should_try():
            raw, available = await self._redis_get(key)
            if available:
                return self._loads(raw)
        return self._memory_get(key)

    async def set_json(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        expire = ttl if ttl is not None else self._settings.cache_ttl_seconds
        if self._redis_should_try():
            try:
                await self._redis.set(key, payload, ex=expire)
                self._redis_ok = True
                return
            except Exception as exc:  # noqa: BLE001
                self._mark_redis_down(exc)
        self._memory_set(key, payload, expire)

    async def close(self) -> None:
        await self._redis.aclose()

    @property
    def backend(self) -> str:
        return "redis" if self._redis_ok else "memory"

    # ---------------------------------------------------------------- 内部
    def _redis_should_try(self) -> bool:
        if self._redis_ok is False and time.monotonic() < self._retry_at:
            return False
        return True

    async def _redis_get(self, key: str) -> tuple[str | None, bool]:
        """返回 (原始值, Redis 是否可用)。可用但未命中时原始值为 None。"""
        try:
            raw = await self._redis.get(key)
            if self._redis_ok is not True:
                logger.info("Redis 已恢复，重新启用 Redis 缓存")
            self._redis_ok = True
            return raw, True
        except Exception as exc:  # noqa: BLE001
            self._mark_redis_down(exc)
            return None, False

    def _mark_redis_down(self, exc: Exception) -> None:
        if self._redis_ok is not False:
            logger.warning("Redis 操作失败，降级为进程内缓存：{}", exc)
        self._redis_ok = False
        self._retry_at = time.monotonic() + _RETRY_INTERVAL_SECONDS

    def _memory_set(self, key: str, payload: str, ttl: int) -> None:
        self._memory[key] = (time.monotonic() + ttl, payload)
        self._memory.move_to_end(key)
        while len(self._memory) > self._settings.cache_max_memory_entries:
            self._memory.popitem(last=False)

    def _memory_get(self, key: str) -> dict[str, Any] | None:
        item = self._memory.get(key)
        if item is None:
            return None
        expire_at, payload = item
        if time.monotonic() >= expire_at:
            self._memory.pop(key, None)
            return None
        self._memory.move_to_end(key)
        return self._loads(payload)

    @staticmethod
    def _loads(raw: str | None) -> dict[str, Any] | None:
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
