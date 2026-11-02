"""初始化模拟巡检时序数据库。

可直接运行：``python db/init_db.py``，也会在 FastAPI 启动时被自动调用。
数据使用固定随机种子生成，保证每次演示结果可复现（时间基准为运行当天）。
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "robot_data.db"
ROBOTS = ("RBT-01", "RBT-02", "RBT-03")
PATROL_POINTS = (
    "A-01",
    "A-02",
    "A-03",
    "A-04",
    "A-05",
    "B-01",
    "B-02",
    "B-03",
    "B-04",
    "B-05",
)
ALARM_TYPES = ("电机过温", "避障超时", "低电量", "定位丢失", "通信中断")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS robot_run_record (
    robot_id     TEXT NOT NULL,
    time         TEXT NOT NULL,
    battery      REAL NOT NULL,
    status       TEXT NOT NULL,
    alarm_type   TEXT,
    patrol_point TEXT
);

CREATE INDEX IF NOT EXISTS idx_robot_time ON robot_run_record (robot_id, time);
CREATE INDEX IF NOT EXISTS idx_status     ON robot_run_record (status);
CREATE INDEX IF NOT EXISTS idx_alarm      ON robot_run_record (alarm_type);
CREATE INDEX IF NOT EXISTS idx_point      ON robot_run_record (patrol_point);
"""


def _build_rows() -> list[tuple[str, str, float, str, str | None, str]]:
    """生成最近 7 天、每小时一条的模拟巡检记录（3 台机器人 × 168 小时）。"""
    rng = random.Random(42)
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=7)
    timestamps = [start + timedelta(hours=offset) for offset in range(7 * 24)]

    rows: list[tuple[str, str, float, str, str | None, str]] = []
    for robot_id in ROBOTS:
        battery = rng.uniform(60.0, 95.0)
        charging_hours = 0

        for moment in timestamps:
            status = "running"
            alarm: str | None = None

            if charging_hours > 0:
                status = "charging"
                battery = min(100.0, battery + rng.uniform(12.0, 20.0))
                charging_hours -= 1
            else:
                roll = rng.random()
                if roll < 0.03:
                    status = "fault"
                    alarm = rng.choice(ALARM_TYPES)
                    battery = max(0.0, battery - rng.uniform(2.0, 5.0))
                elif battery < 18.0:
                    status = "charging"
                    charging_hours = rng.randint(2, 3)
                    battery = min(100.0, battery + rng.uniform(12.0, 18.0))
                elif moment.hour < 6:
                    status = "idle"
                    battery = max(0.0, battery - 0.3)
                else:
                    status = "running"
                    battery = max(0.0, battery - rng.uniform(1.5, 4.0))
                    if battery < 25.0 and rng.random() < 0.35:
                        alarm = "低电量"

            rows.append(
                (
                    robot_id,
                    moment.strftime("%Y-%m-%d %H:%M:%S"),
                    round(battery, 1),
                    status,
                    alarm,
                    rng.choice(PATROL_POINTS),
                )
            )
    return rows


def ensure_database(db_path: Path = DEFAULT_DB_PATH, force: bool = False) -> dict[str, Any]:
    """建库建表并灌入模拟数据；已有数据时直接复用。"""
    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if force and db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(_SCHEMA)
        existing = connection.execute("SELECT COUNT(*) FROM robot_run_record").fetchone()[0]
        seeded = False
        if existing == 0:
            connection.executemany(
                """
                INSERT INTO robot_run_record
                    (robot_id, time, battery, status, alarm_type, patrol_point)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                _build_rows(),
            )
            connection.commit()
            seeded = True
            existing = connection.execute("SELECT COUNT(*) FROM robot_run_record").fetchone()[0]
    finally:
        connection.close()

    return {"path": str(db_path), "rows": int(existing), "seeded": seeded}


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化机器人巡检模拟数据库")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite 文件路径")
    parser.add_argument("--force", action="store_true", help="删除旧库并重新生成")
    args = parser.parse_args()

    result = ensure_database(args.db, force=args.force)
    action = "已重新生成" if result["seeded"] else "已存在，跳过灌数"
    print(f"[init_db] {action}：{result['path']}，共 {result['rows']} 条记录")


if __name__ == "__main__":
    main()
