from __future__ import annotations

"""LSQCA実行traceで共有する型。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScheduledOperation:
    """LSQCA資源モデル上で時刻と資源が確定した1命令。"""

    op: str
    qubits: tuple[int, ...]
    start: int
    end: int
    cr_id: int | None = None
    bank_id: int | None = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
