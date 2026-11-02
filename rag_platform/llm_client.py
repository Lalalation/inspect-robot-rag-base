"""统一的大模型客户端。

对接任意 OpenAI 兼容接口（DeepSeek / 通义千问 / OpenAI / 硅基流动等）：
``/v1/chat/completions`` 与 ``/v1/embeddings``。

对外只暴露两个能力：``chat_json`` 与 ``embed_texts``。
所有调用自带超时、指数退避重试与 JSON 解析兜底；API Key 由 ``.env`` 注入。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
from loguru import logger

from config import Settings


class LLMError(RuntimeError):
    """模型服务不可用、超时或返回结构异常。"""


class LLMJSONError(LLMError):
    """模型返回内容无法解析为 JSON 对象。"""


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中尽力提取一个 JSON 对象。

    依次尝试：整体解析 → 代码围栏内解析 → 首个 ``{`` 到末个 ``}`` 的子串解析。
    仍失败则抛出 :class:`LLMJSONError`，由调用方决定重试或降级。
    """
    if not text or not text.strip():
        raise LLMJSONError("模型返回为空")

    candidate = text.strip().lstrip("\ufeff").strip()

    # 1) 严格 JSON
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) 模型偶尔会包一层 ```json ... ```
    for match in _CODE_FENCE_RE.finditer(candidate):
        body = match.group(1).strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    # 3) 前后夹杂解释文字时，截取最外层花括号
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    snippet = candidate[:120].replace("\n", " ")
    raise LLMJSONError(f"模型输出不是合法 JSON 对象，开头片段：{snippet!r}")


class LLMClient:
    """Chat 与 Embedding 的统一入口。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # trust_env=False：避免系统代理拦截 127.0.0.1 的本地模型请求
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
            trust_env=False,
        )

    # ------------------------------------------------------------ 生命周期
    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------ Chat 接口
    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        """调用对话模型并返回已解析的 JSON 对象。

        重试策略：网络类错误与 5xx 退避重试；JSON 解析失败附带纠正提示重试。
        """
        attempts = self._settings.llm_max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            prompt = user_prompt
            if attempt > 1 and isinstance(last_error, LLMJSONError):
                prompt = f"{user_prompt}\n\n注意：上一次输出不是合法 JSON，本次只能返回一个 JSON 对象。"
            try:
                raw = await self._chat_completions(system_prompt, prompt, max_tokens)
            except Exception as exc:  # noqa: BLE001 - 统一转成 LLMError
                last_error = exc
                if attempt >= attempts or not self._is_retryable(exc):
                    raise LLMError(f"{type(exc).__name__}: {exc}") from exc
                delay = min(0.5 * (2 ** (attempt - 1)), 4.0)
                logger.warning("LLM 调用失败（第 {} 次），{:.1f}s 后重试：{}", attempt, delay, exc)
                await asyncio.sleep(delay)
                continue

            try:
                return extract_json_object(raw)
            except LLMJSONError as exc:
                last_error = exc
                if attempt >= attempts:
                    raise
                logger.warning("LLM 输出 JSON 解析失败（第 {} 次），准备重试", attempt)
                await asyncio.sleep(0.2)

        raise LLMError(f"LLM 调用失败：{last_error}")

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 429 or exc.response.status_code >= 500
        return False

    async def _chat_completions(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        url = self._openai_url("chat/completions")
        payload: dict[str, Any] = {
            "model": self._settings.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._settings.llm_temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        logger.debug("OpenAI-compatible chat: model={}, prompt_chars={}", self._settings.chat_model, len(user_prompt))
        response = await self._http.post(url, json=payload, headers=self._auth_headers(self._settings.llm_api_key))

        # 部分兼容服务不支持 response_format，降级去掉该字段再试一次
        if response.status_code in (400, 422):
            payload.pop("response_format", None)
            response = await self._http.post(
                url, json=payload, headers=self._auth_headers(self._settings.llm_api_key)
            )

        if response.status_code >= 400:
            raise LLMError(f"模型服务 HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("模型服务返回结构不符合 OpenAI 兼容格式") from exc
        if not content:
            raise LLMError("模型返回内容为空")
        return content

    # ------------------------------------------------------- Embedding 接口
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，自动分批并校验维度。"""
        if not texts:
            return []

        if not self._settings.embed_base_url_resolved:
            raise LLMError("Embedding 服务地址为空：请配置 EMBED_BASE_URL 或 LLM_BASE_URL。")

        vectors: list[list[float]] = []
        batch_size = self._settings.embed_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_vectors = await self._embed_completions(batch)
            self._check_dimension(batch_vectors)
            vectors.extend(batch_vectors)
        return vectors

    async def _embed_completions(self, texts: list[str]) -> list[list[float]]:
        url = self._openai_url("embeddings", base_url=self._settings.embed_base_url_resolved)
        payload = {"model": self._settings.embed_model, "input": texts}
        response = await self._http.post(
            url, json=payload, headers=self._auth_headers(self._settings.embed_api_key_resolved)
        )
        if response.status_code >= 400:
            raise LLMError(f"Embedding 服务 HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise LLMError("Embedding 返回数量与输入不一致")
        items = sorted(items, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    def _check_dimension(self, vectors: list[list[float]]) -> None:
        if not vectors:
            return
        actual = len(vectors[0])
        expected = self._settings.embed_dim
        if actual != expected:
            raise LLMError(
                f"Embedding 维度不匹配：模型输出 {actual} 维，配置 EMBED_DIM={expected}。"
                "请修改 .env 中的 EMBED_DIM，并调用 POST /api/v1/documents/reindex 重建向量集合。"
            )

    # ------------------------------------------------------------ 健康检查
    async def health(self) -> dict[str, Any]:
        """返回可公开的模型状态，不包含任何密钥内容。"""
        info: dict[str, Any] = {
            "provider": "openai-compatible",
            "chat_model": self._settings.chat_model,
            "embed_model": self._settings.embed_model,
            "api_key_configured": bool(self._settings.llm_api_key),
            "auth_mode": "bearer" if self._settings.llm_api_key else "missing_key",
            "status": "down",
        }
        try:
            response = await self._http.get(
                self._openai_url("models"),
                headers=self._auth_headers(self._settings.llm_api_key),
                timeout=3.0,
            )
            info["status"] = "up" if response.status_code < 400 else "down"
            if response.status_code >= 400:
                info["detail"] = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001 - 健康检查只关心结果
            info["detail"] = type(exc).__name__
        return info

    # ---------------------------------------------------------------- 工具
    def _auth_headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _openai_url(self, path: str, base_url: str | None = None) -> str:
        base = (base_url or self._settings.llm_base_url).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/{path}"
        return f"{base}/v1/{path}"
