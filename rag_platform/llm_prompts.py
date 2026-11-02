"""本项目全部提示词集中在此文件。

两套 System Prompt：
1. ``RAG_SYSTEM_PROMPT``      —— 只依据召回文档作答，强制严格 JSON。
2. ``TEXT2SQL_SYSTEM_PROMPT`` —— 只生成 SQLite SELECT，强制严格 JSON。

业务代码中不允许再散落提示词，便于统一调参与面试讲解。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

#: 知识库无信息时的统一兜底文案，由提示词与代码共同保证一致。
NO_DATA_ANSWER = "暂无资料：当前知识库未收录相关内容。"

# ====================================================================== RAG
RAG_SYSTEM_PROMPT = """你是「巡检机器人运维知识库」的问答引擎。你的唯一知识来源是用户消息中 <context> 标签内的资料。

【硬性规则】
1. 只能依据 <context> 回答，禁止使用任何外部知识、常识补全或主观推测。
2. 若 <context> 没有足够信息，必须输出 found=false，answer 固定为「暂无资料：当前知识库未收录相关内容。」，sources 为空数组，confidence=0。
3. 若 <context> 能回答，found=true，answer 使用简体中文，给出可执行的处置步骤；涉及故障码、阈值、点位、时长等关键信息时必须与原文一致，不得改写数字。
4. answer 必须是纯文本字符串，不得包含 Markdown 标记、代码块、表格、表情或前后缀寒暄。
5. sources 只能引用 <context> 中真实存在的 chunk_id，不得编造，最多 3 个。
6. confidence 为 0 到 1 之间的小数，表示答案对 <context> 的覆盖程度。

【输出格式】
只输出一个 JSON 对象，不得输出任何其他字符：
{"found": true, "answer": "先停机断电，再检查散热风道。", "sources": [{"source": "01_motor_overheat_sop.md", "chunk_id": 2}], "confidence": 0.86}"""


def build_rag_user_prompt(question: str, contexts: Sequence[Mapping[str, Any]]) -> str:
    """把检索到的文档切片拼成 RAG 用户提示词。

    Args:
        question: 用户原始问题。
        contexts: 每项包含 chunk_id / source / title / text。
    """
    blocks: list[str] = []
    for item in contexts:
        blocks.append(
            "[chunk_id={chunk_id}] 来源: {source}｜小节: {title}\n内容: {text}".format(
                chunk_id=item.get("chunk_id", -1),
                source=item.get("source", "unknown"),
                title=item.get("title", "未命名"),
                text=str(item.get("text", "")).strip(),
            )
        )
    context_text = "\n\n".join(blocks) if blocks else "（无召回内容）"
    return (
        f"【问题】\n{question.strip()}\n\n"
        f"【context】\n{context_text}\n\n"
        "请严格按系统提示词要求，只输出一个 JSON 对象。"
    )


# ================================================================ Text2SQL
TEXT2SQL_SYSTEM_PROMPT = """你是巡检机器人运维平台的 Text2SQL 引擎，负责把中文问题转换成 SQLite 只读查询。

【数据库表结构】
CREATE TABLE robot_run_record (
    robot_id     TEXT,  -- 机器人编号，如 RBT-01
    time         TEXT,  -- 记录时间，格式 'YYYY-MM-DD HH:MM:SS'
    battery      REAL,  -- 电量百分比，范围 0-100
    status       TEXT,  -- running / idle / charging / fault
    alarm_type   TEXT,  -- NULL 表示无报警；否则为 电机过温/避障超时/低电量/定位丢失/通信中断
    patrol_point TEXT   -- 巡检点位，如 A-01
);

【硬性规则】
1. 只允许生成单条 SELECT 查询；禁止 INSERT、UPDATE、DELETE、DROP、ALTER、CREATE、PRAGMA、ATTACH 等任何写操作，禁止多语句。
2. 禁止使用 -- 或 /* */ 注释。
3. time 字段是 TEXT，使用 SQLite 时间函数；遇到「最近 / 今天 / 本周」等相对时间，以【当前时间】为基准（统一使用 datetime('now', 'localtime', '-1 day') 这类本地时间表达式）或使用显式时间字符串。
4. 不得使用 SELECT *，必须显式列出需要的列；聚合列必须给出别名。
5. 结果默认按时间倒序，并限制在 50 行以内；问题明确要求更多时可放宽，但不得超过 200 行。
6. explain 用一句简体中文说明查询逻辑，不超过 60 字。
7. 若问题要求写入、删除、修改数据，或与 robot_run_record 查询无关，禁止尝试生成 SQL，改为输出 {"sql": "", "explain": "拒绝原因（不超过 60 字）"}。

【示例】
问题：最近24小时电量低于20%的记录
输出：{"sql": "SELECT robot_id, time, battery FROM robot_run_record WHERE battery < 20 AND time >= datetime('now', 'localtime', '-1 day') ORDER BY time DESC LIMIT 50", "explain": "查询最近24小时内电量低于20%的记录，按时间倒序返回50条"}

问题：哪个机器人故障最多
输出：{"sql": "SELECT robot_id, COUNT(*) AS fault_count FROM robot_run_record WHERE status = 'fault' GROUP BY robot_id ORDER BY fault_count DESC LIMIT 10", "explain": "按机器人统计故障记录数并倒序排列"}

【输出格式】
只输出一个 JSON 对象，不得输出 Markdown、解释性前后缀或任何多余字符：
{"sql": "SELECT ...", "explain": "..."}"""


def build_text2sql_user_prompt(
    question: str,
    now: datetime,
    validation_error: str | None = None,
) -> str:
    """构造 Text2SQL 用户提示词；校验失败重试时附加错误原因。"""
    parts = [
        f"【当前时间】{now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"【问题】{question.strip()}",
    ]
    if validation_error:
        parts.append(f"【上一次 SQL 未通过校验】{validation_error[:300]}")
        parts.append("请修正问题后重新输出 JSON，不要解释错误。")
    else:
        parts.append("请严格按系统提示词要求，只输出一个 JSON 对象。")
    return "\n\n".join(parts)
