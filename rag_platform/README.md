# 机器人运维私有化 RAG & Text2SQL 平台

面向巡检机器人运维人员的私有化智能问答与数据查询服务：把故障 SOP / 运维手册做成私有知识库，支持自然语言问答；同时把「最近 24 小时电量低于 20% 的机器人」这类问题自动转成只读 SQL 并返回结果。

> **定位声明**：本项目是**独立的运维知识平台**，只提供 HTTP REST 接口，可作为机器人上层 Agent 的知识库服务被调用。仓库内**不包含、也不依赖 ROS2 / rclpy / Gazebo 及任何机器人导航代码**，可脱离机器人本体单独部署与演示。

---

## 1. 能力概览

| 模块 | 能力 | 关键实现 |
| --- | --- | --- |
| 运维控制台 | 问答、查询、文档管理三合一的中文 Web 界面 | 原生 HTML/CSS/JS，由 FastAPI 同源托管，无构建步骤 |
| 文档入库 | 加载 Markdown / TXT，切片、向量化写入 Qdrant | Markdown 标题分段 + 定长滑窗，同源旧切片覆盖更新 |
| RAG 问答 | 只依据召回文档作答，无资料返回「暂无资料」 | LangGraph 四节点图 + 严格 JSON + 引用真实性校验 |
| Text2SQL | 中文问题转 SQLite 只读 SELECT | LangGraph 五节点图 + 三道安全防线 |
| 缓存降级 | 问答 / SQL 结果缓存，Redis 故障自动降级 | Redis TTL 600s → 进程内 TTL 缓存 |
| 内置数据 | 5 篇运维文档 + 504 条巡检记录 | 启动自动灌库、自动建表灌数，开箱可测 |

技术栈：Python 3.10+、FastAPI、LangGraph、OpenAI 兼容云端模型（默认通义千问 `qwen-plus`）、Qdrant、SQLite、Redis、loguru、Pydantic。模型统一走 `/v1/chat/completions` 与 `/v1/embeddings`，只需在 `.env` 填入 API Key；不做本地推理，不依赖 GPU，也不微调模型。

---

## 2. 目录结构

```
rag_platform/
├── main.py                # FastAPI 入口：生命周期、日志、异常处理、控制台页面 + 6 个 API 路由
├── static/                # 运维控制台：index.html + styles.css + app.js
├── config.py              # Pydantic Settings：模型 / Qdrant / Redis / SQLite 配置
├── llm_prompts.py         # 本项目全部提示词：RAG 与 Text2SQL 两套 System Prompt
├── llm_client.py          # 统一模型客户端：OpenAI 兼容接口，重试 + JSON 解析
├── cache.py               # Redis 缓存 + 进程内 TTL 缓存降级
├── document_loader.py     # 文档解析、切片、向量化、Qdrant 写入
├── retrieval.py           # RAG 检索问答（LangGraph 状态图）
├── sql_generator.py       # Text2SQL 生成、校验、只读执行（LangGraph 状态图）
├── db/
│   ├── init_db.py         # 建表 + 生成模拟巡检时序数据
│   └── robot_data.db      # 已生成的 SQLite 演示库（504 条）
├── knowledge_docs/        # 内置模拟故障 SOP / 运维手册（5 篇）
├── requirements.txt
├── docker-compose.yml     # 启动 Qdrant + Redis
├── .env.example           # 配置模板（复制为 .env）
└── README.md
```

---

## 3. 快速开始（Windows）

### 3.1 前置依赖

| 依赖 | 说明 |
| --- | --- |
| Python 3.10+ | 建议 3.10 / 3.11 |
| Docker Desktop | 用于启动 Qdrant 与 Redis；也可自行安装原生服务 |
| 模型 API Key | 通义千问 / DeepSeek / OpenAI / 硅基流动等，任一 OpenAI 兼容服务 |

```powershell
# 1) 启动 Qdrant 与 Redis
cd rag_platform
docker compose up -d

# 2) 安装依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3) 生成配置并填入 API Key
Copy-Item .env.example .env
# 编辑 .env：LLM_API_KEY=你的Key（Embedding 默认复用同一个 Key）

# 4) 启动服务
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/> 即可使用运维控制台：左侧切换「智能问答 / 数据查询 / 文档管理」，顶部实时显示 Qdrant、Redis、模型与数据库状态。需要调试原始接口时，访问 <http://127.0.0.1:8000/docs>。

**启动时自动完成的事**：创建 Qdrant 集合 → 集合为空则自动入库 `knowledge_docs/` → 检查并初始化 `db/robot_data.db`（首次自动灌 504 条模拟数据）→ 探测 Redis，不可用则自动降级。

> 若只想先验证接口形态、不启动 Qdrant 或尚未填写 API Key，服务同样能启动：`/health` 返回 `degraded`，问答与 SQL 接口返回 503 并附带真实错误原因，不会崩溃。

---

## 4. 接口说明

统一前缀 `/api/v1`，请求与响应均为 JSON。

### 4.1 健康检查

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
```

```json
{
  "status": "degraded",
  "version": "1.0.0",
  "qdrant": { "status": "down", "collection": "robot_ops_docs", "chunks": 0 },
  "redis": { "status": "degraded", "backend": "memory" },
  "llm": {
    "provider": "openai-compatible",
    "chat_model": "qwen-plus",
    "embed_model": "text-embedding-v3",
    "api_key_configured": false,
    "auth_mode": "missing_key",
    "status": "down",
    "detail": "HTTP 401"
  },
  "sqlite": { "status": "up", "rows": 504 }
}
```

四个依赖全部正常时 `status` 为 `ok`，否则为 `degraded`。`api_key_configured` 只暴露布尔值，不返回密钥内容。

### 4.2 文档上传

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/upload" `
  -F "file=@knowledge_docs/02_激光雷达故障排查.md"
```

```json
{ "ingested": true, "source": "02_激光雷达故障排查.md", "chunks": 3 }
```

仅支持 `.md` / `.txt`，单文件默认不超过 2 MB。同名文件重复上传会先删除旧切片，避免文档更新后残留旧版本内容。

### 4.3 RAG 问答

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/rag/ask" `
  -H "Content-Type: application/json" `
  -d '{\"question\":\"激光雷达无数据、导航起不来怎么排查？\",\"top_k\":4}'
```

```json
{
  "found": true,
  "answer": "依次检查雷达供电线缆和网口是否松动、雷达 IP 是否与工控机在同一网段，再重启雷达电源与雷达驱动节点；以上均无效时判定为雷达硬件故障，需要更换雷达。",
  "sources": [
    {
      "source": "02_激光雷达故障排查.md",
      "title": "故障1：雷达无数据，rqt查看topic无点云",
      "chunk_id": 0,
      "score": 0.8123
    }
  ],
  "confidence": 0.86,
  "degraded": false,
  "cached": false,
  "latency_ms": 2140
}
```

知识库无相关内容时：`found=false`，`answer` 固定为「暂无资料：当前知识库未收录相关内容。」，`sources=[]`。

### 4.4 Text2SQL 查询

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/sql/query" `
  -H "Content-Type: application/json" `
  -d '{\"question\":\"最近24小时电量低于20%的记录\"}'
```

```json
{
  "sql": "SELECT robot_id, time, battery FROM robot_run_record WHERE battery < 20 AND time >= datetime('now', 'localtime', '-1 day') ORDER BY time DESC LIMIT 50",
  "explain": "查询最近24小时内电量低于20%的记录，按时间倒序返回50条",
  "columns": ["robot_id", "time", "battery"],
  "rows": [
    { "robot_id": "RBT-02", "time": "2026-10-28 03:00:00", "battery": 17.4 }
  ],
  "row_count": 1,
  "truncated": false,
  "degraded": false,
  "cached": false,
  "latency_ms": 3210
}
```

### 4.5 重建知识库

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/reindex"
```

```json
{ "reindexed": true, "documents": 5, "chunks": 15, "failed": [] }
```

**切换 Embedding 模型后必须调用此接口**，否则新旧向量维度不一致会导致检索报错。

### 4.6 状态码约定

| 状态码 | 含义 |
| --- | --- |
| 200 | 成功 |
| 400 / 413 | 文件类型不支持 / 文件过大 |
| 409 | 向量维度与已有集合冲突 |
| 422 | SQL 未通过安全校验或执行失败（`detail` 中含 `sql` 与 `error`） |
| 503 | 模型 / 检索服务不可用（`detail` 为降级结构，含真实错误类型） |

---

## 5. 模型配置

模型调用统一走 OpenAI 兼容协议：只要服务支持 `/v1/chat/completions` 与 `/v1/embeddings` 即可，**无需改代码**。API Key 只写在 `.env`，不进入代码仓库。

默认预置通义千问（DashScope 兼容模式），对话与向量共用一个 Key，最省配置：

```ini
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=sk-你的Key
CHAT_MODEL=qwen-plus

EMBED_BASE_URL=        # 留空即复用 LLM_BASE_URL
EMBED_API_KEY=         # 留空即复用 LLM_API_KEY
EMBED_MODEL=text-embedding-v3
EMBED_DIM=1024
```

### 5.1 对话模型可选项

| 服务 | `LLM_BASE_URL` | `CHAT_MODEL` | 备注 |
| --- | --- | --- | --- |
| 通义千问（默认） | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` | 同一个 Key 可做 Embedding |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` | 无 Embedding，需单独配 |
| OpenAI | `https://api.openai.com` | `gpt-4o-mini` | 对话与向量同一 Key |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` | 同一个 Key 可做 Embedding |

### 5.2 Embedding 模型

`EMBED_DIM` 必须与所选模型的实际输出维度一致：

| 模型 | 维度 | 说明 |
| --- | --- | --- |
| `text-embedding-v3` | 1024 | 默认，通义千问 |
| `BAAI/bge-m3` | 1024 | 硅基流动，中文检索更强 |
| `text-embedding-3-small` | 1536 | OpenAI |
| `text-embedding-3-large` | 3072 | OpenAI |

`EMBED_BASE_URL` / `EMBED_API_KEY` 留空时自动复用 LLM 的地址与 Key；向量与对话用不同厂商时单独填写。切换模型或维度后：改 `.env` → 重启服务 → `POST /api/v1/documents/reindex` 重建集合。

---

## 6. 设计说明

### 6.1 为什么用 LangGraph

两条链路都有明确的分支与重试语义，状态图让流程可读、可观测、可单节点测试：

```
RAG：      START → retrieve ─┬─(有召回)→ generate → parse → finalize → END
                              └─(无召回/检索异常)→ finalize

Text2SQL： START → generate → validate ─┬─(通过)→ execute → finalize → END
                                        ├─(校验失败)→ regenerate → validate
                                        └─(重试耗尽/模型异常)→ finalize
```

- **召回为空直接短路**，不浪费一次模型调用，也不让模型凭空作答；
- SQL 校验失败时把错误原因回灌给模型重新生成，最多重试 1 次；
- 每个节点是纯函数式 `state -> dict`，便于面试时单独演示。

### 6.2 强制 JSON 的三层保障

1. **提示词层**：System Prompt 明确「只输出一个 JSON 对象」，禁止 Markdown、代码块、寒暄；
2. **模型层**：请求携带 `response_format: {"type": "json_object"}`；若网关不支持则自动去掉该字段重试一次；
3. **代码层**：`extract_json_object()` 依次尝试整体解析 → 代码围栏解析 → 首尾花括号截取，仍失败则附带「只能返回 JSON」的纠正提示重试，最终抛 `LLMJSONError` 走降级。

JSON 解析成功 ≠ 可信输出，因此 `parse` 节点还会做结构校验：

- `found=false` 时强制替换为标准「暂无资料」文案；
- `sources` 只保留本次召回中真实存在的 `chunk_id`，模型编造的来源会被丢弃；
- `confidence` 强制裁剪到 `[0, 1]`，答案长度截断，防止异常长输出。

### 6.3 Text2SQL 的三道安全防线

| 防线 | 措施 |
| --- | --- |
| 提示词 | 只允许单条 SELECT，禁止写操作、注释、`SELECT *`，自动 LIMIT 50 |
| 静态校验 `validate_sql()` | 单语句检查、注释检查、首关键字必须是 `SELECT`/`WITH`、21 个高危关键字黑名单、表名白名单（仅 `robot_run_record`）且支持 CTE、无 LIMIT 时自动追加 |
| 执行层 | SQLite `file:...?mode=ro` 只读 URI（物理拒绝写入）+ `set_progress_handler` 超时中断 + `fetchmany(max_rows+1)` 截断 |

字符串常量会先被掩码再检查，避免 `alarm_type = 'delete'` 这类字面量造成误判；CTE 名称会从 `WITH x AS (...)` 中提取并加入临时白名单，因此 `WITH low AS (...) SELECT ... FROM low` 可正常运行。

### 6.4 缓存与降级

- 缓存 key 由 `module + question + 参数` 做 SHA-256，问答与 SQL 分开命名空间；
- Redis 可用时写 Redis（TTL 600s），不可用时自动切到进程内 TTL 缓存，并对 Redis 做 30 秒重连冷却，避免每个请求都等连接超时；
- 降级结果（`degraded=true`）**不写缓存**，避免一次故障把错误答案缓存 10 分钟；
- `cached=true` 时响应带 `latency_ms`，便于演示缓存命中前后的耗时差异。

### 6.5 切片策略

先按 Markdown 一至四级标题切成语义段（保留小节标题），段内再做定长滑窗（默认 480 字符 / 80 重叠），并优先在句号、换行、分号处断开，避免把一句话切成两半。每个切片携带 `source`、`title`、`chunk_id`，问答时可直接回溯到原文位置。

---

## 7. 数据说明

### 7.1 知识库文档

| 文件 | 内容 |
| --- | --- |
| `01_机器人基础巡检 SOP.md` | 开机前检查、自动巡检流程、停机规范与禁止操作 |
| `02_激光雷达故障排查.md` | 雷达无数据、点云噪声与定位漂移、雷达超时断开 |
| `03_导航与避障故障.md` | 原地不动、频繁避障绕路、导航任务 aborted |
| `04_电池与充电故障.md` | 无法对接充电座、电量下降过快、低电量保护阈值 |
| `05_通信与网络故障.md` | 中控断连、rosbag 丢包延迟、gRPC 接口超时 |

文档中的 Nav2、rqt、rosbag 等属于现场运维术语，仅作为知识素材；本项目代码不依赖 ROS2 / rclpy / Gazebo。

### 7.2 巡检时序表

```sql
CREATE TABLE robot_run_record (
    robot_id     TEXT NOT NULL,   -- RBT-01 / RBT-02 / RBT-03
    time         TEXT NOT NULL,   -- 'YYYY-MM-DD HH:MM:SS'
    battery      REAL NOT NULL,   -- 0-100
    status       TEXT NOT NULL,   -- running / idle / charging / fault
    alarm_type   TEXT,            -- NULL 表示无报警
    patrol_point TEXT             -- A-01 ... B-05
);
```

3 台机器人 × 7 天 × 每小时 = 504 条，固定随机种子保证可复现；时间基准是**首次初始化当天**，因此「最近 24 小时」这类相对时间查询始终有数据。重新生成：

```powershell
python db/init_db.py --force
```

---

## 8. 常见问题

**Q：启动时报 `[Errno 10048]` 端口被占用？**
8000 已被其他程序占用，换一个端口启动即可，例如 `python -m uvicorn main:app --port 8010`，后续请求地址同步改成 8010。

**Q：启动日志报 Qdrant 初始化失败 / 502，服务还能用吗？**
能。服务会以 `degraded` 状态启动，`/health` 中 `qdrant.status=down`，问答接口返回 503。启动 `docker compose up -d` 后重启服务即可。

**Q：`/health` 里 `redis.backend` 是 `memory`？**
说明 Redis 未启动，缓存已自动降级到进程内，功能不受影响，只是缓存不跨进程共享。启动 Redis 后 30 秒内会自动恢复。

**Q：报「Embedding 维度不匹配」或 409？**
`EMBED_DIM` 与模型实际维度不一致。改 `.env` 中 `EMBED_DIM`，重启后调用 `POST /api/v1/documents/reindex` 重建集合。

**Q：模型返回不是合法 JSON？**
已内置三层解析 + 一次纠正重试；云端模型偶发格式抖动属于正常现象，重试后仍失败会返回 503 并记录 `JSONParseError`，不会把脏数据写进响应。

**Q：SQL 接口返回 422？**
表示模型生成的 SQL 未通过安全校验或执行失败。响应 `detail` 中同时给出 `sql` 与 `error`，可直接看出被拦截的原因（如命中高危关键字、非白名单表）。

**Q：为什么不用 `SELECT *`？**
白名单表虽只有 6 列，但强制显式列名能减少模型幻觉、稳定输出结构，也让返回的 `columns` 可预期。

---

## 9. 面试讲解要点

1. **链路完整**：文档 → 切片 → 向量 → 检索 → 生成 → 校验 → 缓存，一条 RAG 生产线全部自己实现，没有套壳框架。
2. **可观测**：LangGraph 节点划分让「召回为空」「JSON 解析失败」「SQL 校验失败」三类失败各有独立分支，日志里能直接定位耗时与失败点。
3. **安全边界清晰**：Text2SQL 不是「相信提示词」，而是提示词 + 静态校验 + 只读执行三层兜底，最坏情况也写不进数据库。
4. **降级不是口号**：Redis、Qdrant、模型服务任一不可用时服务仍能启动，`/health` 如实反映各依赖状态，错误信息保留原始异常类型。
5. **工程细节**：Windows 路径处理（`as_posix()`）、`httpx trust_env=False` 规避系统代理、UTF-8 读取、缓存键哈希、Embedding 维度前置校验。

---

## 10. 边界与后续演进

本项目刻意保持独立与轻量，**不包含**机器人本体控制、导航、ROS2 通信等能力。后续可演进方向：

- 接入机器人 Agent 作为工具（Tool）调用本服务；
- 增加 BM25 + 向量混合检索与 rerank；
- 文档解析扩展到 PDF / Word；
- 增加查询审计日志与按用户维度的限流。
