# RobotOps · 机器人运维 RAG & Text2SQL 平台

**Private RAG & Text2SQL service for inspection robot operations.**

面向巡检机器人运维场景的私有化知识问答与数据查询服务：把故障 SOP、运维手册沉淀为可检索的私有知识库，同时支持用自然语言查询巡检时序数据。项目以独立 HTTP 服务交付，可被机器人上层 Agent、运维后台或工单系统直接调用。

> 本项目是**独立的运维知识服务**，不包含、也不依赖 ROS2 / rclpy / Gazebo 或任何机器人导航代码，可脱离机器人本体单独部署与演示。

## 目录

- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [接口说明](#接口说明)
- [系统架构](#系统架构)
- [设计要点](#设计要点)
- [项目结构](#项目结构)
- [内置数据](#内置数据)

## 功能特性

| 模块 | 能力 | 关键实现 |
| --- | --- | --- |
| 运维控制台 | 问答、查询、文档管理三合一的中文 Web 界面 | 原生 HTML/CSS/JS，由 FastAPI 同源托管，无构建步骤、无 CDN 依赖 |
| 文档入库 | 加载 Markdown / TXT，切片、向量化写入 Qdrant | Markdown 标题分段 + 定长滑窗，同源旧切片覆盖更新 |
| RAG 问答 | 只依据召回文档作答，无资料返回「暂无资料」 | LangGraph 四节点图 + 严格 JSON + 引用真实性校验 |
| Text2SQL | 中文问题转 SQLite 只读 SELECT | LangGraph 五节点图 + 三道安全防线 |
| 缓存降级 | 问答 / SQL 结果缓存，Redis 故障自动降级 | Redis TTL 600s → 进程内 TTL 缓存 |
| 内置数据 | 5 篇运维文档 + 504 条巡检记录 | 启动自动灌库、自动建表灌数，开箱可测 |

## 技术栈

- **运行时**：Python 3.10+
- **Web 框架**：FastAPI + Uvicorn
- **编排**：LangGraph `StateGraph`
- **模型**：任意 OpenAI 兼容接口（`/v1/chat/completions` + `/v1/embeddings`），默认通义千问
- **向量库**：Qdrant（COSINE，UUID5 稳定 ID）
- **关系库**：SQLite（只读 URI + 超时中断）
- **缓存**：Redis + 进程内 TTL 降级
- **工程**：loguru、Pydantic Settings、httpx
- **前端**：原生 HTML / CSS / JavaScript

模型只做云端 API 调用，不在本地推理，不依赖 GPU，也不需要微调。

## 快速开始

### 1. 前置依赖

| 依赖 | 说明 |
| --- | --- |
| Python 3.10+ | 建议 3.10 / 3.11 |
| Docker Desktop | 用于启动 Qdrant 与 Redis；也可自行安装原生服务 |
| 模型 API Key | 通义千问 / DeepSeek / OpenAI / 硅基流动等任一 OpenAI 兼容服务 |

### 2. 启动 Qdrant 与 Redis

```powershell
cd rag_platform
docker compose up -d
```

若本机没有 Docker，也可以直接安装 Qdrant 与 Redis 原生服务，默认地址为 `127.0.0.1:6333`、`127.0.0.1:6379`。

### 3. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. 配置模型 Key

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少填写以下三项：

```ini
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的Key
CHAT_MODEL=qwen-plus
```

Embedding 默认复用同一个 Key：

```ini
EMBED_BASE_URL=          # 留空 = 复用 LLM_BASE_URL
EMBED_API_KEY=           # 留空 = 复用 LLM_API_KEY
EMBED_MODEL=text-embedding-v3
EMBED_DIM=1024           # 必须与模型实际输出维度一致
```

常用组合与维度说明见 [rag_platform/README.md](rag_platform/README.md#52-embedding-模型)。

### 5. 启动服务

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

打开 **<http://127.0.0.1:8000/>** 进入运维控制台；需要调试原始接口时访问 **<http://127.0.0.1:8000/docs>**。

> 若 8000 端口被占用，改用 `--port 8010`，后续请求地址同步替换即可。

**启动时自动完成**：创建 Qdrant 集合 → 集合为空则自动入库 `knowledge_docs/` → 检查并初始化 `db/robot_data.db`（首次自动灌 504 条模拟数据）→ 探测 Redis，不可用则自动降级。即使 Qdrant 或 Key 未就绪，服务也能以降级状态启动，不会崩溃。

## 接口说明

统一前缀 `/api/v1`，请求与响应均为 JSON。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 运维控制台页面 |
| GET | `/health` | 依赖健康检查（Qdrant / Redis / 模型 / SQLite） |
| POST | `/api/v1/documents/upload` | 上传 `.md` / `.txt` 文档并入库 |
| POST | `/api/v1/documents/reindex` | 清空并重建知识库 |
| POST | `/api/v1/rag/ask` | RAG 问答，返回答案、来源与置信度 |
| POST | `/api/v1/sql/query` | 自然语言转只读 SQL 并执行 |

示例：

```powershell
$body = @{ question = "激光雷达无数据、导航起不来怎么排查？"; top_k = 4 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/rag/ask -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) | ConvertTo-Json -Depth 5
```

```powershell
$body = @{ question = "最近24小时电量低于20%的记录" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/sql/query -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) | ConvertTo-Json -Depth 5
```

状态码约定：

| 状态码 | 含义 |
| --- | --- |
| 200 | 成功 |
| 400 / 413 | 文件类型不支持 / 文件过大 |
| 409 | 向量维度与已有集合冲突 |
| 422 | SQL 未通过安全校验或执行失败（`detail` 中含 `sql` 与 `error`） |
| 503 | 模型 / 检索服务不可用（`detail` 为降级结构，含真实错误类型） |

## 系统架构

```text
┌──────────────────────────────────────────────────────────────┐
│  运维控制台（/）  ·  机器人上层 Agent  ·  运维后台 / 工单系统  │
└───────────────────────────────┬──────────────────────────────┘
                                │ HTTP / JSON
┌───────────────────────────────▼──────────────────────────────┐
│                        FastAPI (main.py)                     │
│  /health   /documents/upload   /documents/reindex            │
│  /rag/ask                      /sql/query                    │
└───────────────┬───────────────────────────────┬──────────────┘
                │                               │
      ┌─────────▼──────────┐          ┌─────────▼──────────┐
      │ RAG LangGraph       │          │ Text2SQL LangGraph │
      │ retrieve→generate   │          │ generate→validate  │
      │ →parse→finalize     │          │ →execute→finalize  │
      └─────────┬──────────┘          └─────────┬──────────┘
                │                               │
      ┌─────────▼──────────┐          ┌─────────▼──────────┐
      │ Qdrant 向量库       │          │ SQLite（只读 URI）  │
      └─────────┬──────────┘          └─────────┬──────────┘
                │                               │
      ┌─────────▼───────────────────────────────▼──────────┐
      │         Redis 缓存（故障时降级到进程内 TTL）        │
      └─────────┬──────────────────────────────────────────┘
                │
      ┌─────────▼──────────────────────────────────────────┐
      │   OpenAI 兼容模型：/v1/chat/completions + embeddings │
      └────────────────────────────────────────────────────┘
```

## 设计要点

### 为什么用 LangGraph

两条链路都有明确的分支与重试语义，状态图让流程可读、可观测、可单节点测试：

```text
RAG：      START → retrieve ─┬─(有召回)→ generate → parse → finalize → END
                              └─(无召回/检索异常)→ finalize

Text2SQL： START → generate → validate ─┬─(通过)→ execute → finalize → END
                                        ├─(校验失败)→ regenerate → validate
                                        └─(重试耗尽/模型异常)→ finalize
```

- 召回为空直接短路，不浪费一次模型调用，也不让模型凭空作答；
- SQL 校验失败时把错误原因回灌给模型重新生成，最多重试 1 次；
- 每个节点是纯函数式 `state -> dict`，便于单独演示与测试。

### 强制 JSON 的三层保障

1. **提示词层**：System Prompt 明确只输出一个 JSON 对象，禁止 Markdown、代码块、寒暄；
2. **模型层**：请求携带 `response_format: {"type": "json_object"}`，网关不支持时自动去掉该字段重试；
3. **代码层**：整体解析 → 代码围栏解析 → 首尾花括号截取，仍失败则附带纠正提示重试，最终走降级。

JSON 解析成功不等于可信输出：`found=false` 时强制替换为标准「暂无资料」文案；`sources` 只保留本次召回中真实存在的切片；`confidence` 强制裁剪到 `[0, 1]`。

### Text2SQL 三道安全防线

| 防线 | 措施 |
| --- | --- |
| 提示词 | 只允许单条 SELECT，禁止写操作、注释、`SELECT *`，自动 LIMIT |
| 静态校验 | 单语句、禁注释、首关键字 `SELECT`/`WITH`、21 个高危关键字黑名单、表名白名单（仅 `robot_run_record`，支持 CTE）、无 LIMIT 自动追加 |
| 执行层 | SQLite `mode=ro` 只读 URI + `progress_handler` 超时中断 + `fetchmany` 截断 |

字符串常量先掩码再检查，避免 `alarm_type = 'delete'` 这类字面量误判；CTE 名称会加入临时表白名单。模型主动拒绝写操作时返回 422，而不是 503。

### 缓存与降级

- 缓存 key 由 `module + question + 参数` 做 SHA-256，问答与 SQL 分命名空间；
- Redis 可用时写 Redis（TTL 600s），不可用时切到进程内 TTL 缓存，并对 Redis 做 30 秒重连冷却；
- 降级结果不写缓存，避免一次故障把错误答案缓存 10 分钟；
- 命中缓存时响应带 `cached=true`，便于对比耗时。

## 项目结构

```text
.
├── rag_platform/
│   ├── main.py                # FastAPI 入口、生命周期、控制台页面 + 6 个 API 路由
│   ├── config.py              # Pydantic Settings：模型 / Qdrant / Redis / SQLite
│   ├── llm_prompts.py         # 全部提示词：RAG 与 Text2SQL 两套 System Prompt
│   ├── llm_client.py          # OpenAI 兼容客户端：重试 + 严格 JSON 解析
│   ├── cache.py               # Redis 缓存 + 进程内 TTL 降级
│   ├── document_loader.py     # 文档解析、切片、向量化、Qdrant 写入
│   ├── retrieval.py           # RAG 检索问答（LangGraph）
│   ├── sql_generator.py       # Text2SQL 生成、校验、只读执行（LangGraph）
│   ├── static/                # 运维控制台：index.html / styles.css / app.js
│   ├── db/
│   │   ├── init_db.py         # 建表 + 生成模拟巡检时序数据
│   │   └── robot_data.db      # 504 条演示数据
│   ├── knowledge_docs/        # 5 篇内置运维文档
│   ├── requirements.txt
│   ├── docker-compose.yml
│   ├── .env.example
│   └── README.md              # 详细配置、接口与设计说明
├── agent.md
└── .gitignore
```

## 内置数据

知识库包含 5 篇运维文档（共 15 个切片）：

| 文件 | 内容 |
| --- | --- |
| `01_机器人基础巡检 SOP.md` | 开机前检查、自动巡检流程、停机规范 |
| `02_激光雷达故障排查.md` | 雷达无数据、点云噪声、雷达超时断开 |
| `03_导航与避障故障.md` | 原地不动、频繁避障绕路、任务 aborted |
| `04_电池与充电故障.md` | 无法对接充电座、电量下降过快、低电量保护 |
| `05_通信与网络故障.md` | 中控断连、rosbag 丢包延迟、gRPC 接口超时 |

巡检表 `robot_run_record` 为 3 台机器人 × 7 天 × 每小时 = 504 条，固定随机种子可复现；时间基准为首次初始化当天，因此「最近 24 小时」这类相对时间查询始终有数据。

文档中的 Nav2、rqt、rosbag 等属于现场运维术语，仅作为知识素材，项目代码不依赖 ROS2 / rclpy / Gazebo。

---

更完整的配置项、FAQ 与接口示例请见 **[rag_platform/README.md](rag_platform/README.md)**。
