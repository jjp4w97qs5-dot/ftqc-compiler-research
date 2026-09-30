from __future__ import annotations

"""LSQCA実行中の物理状態、統計、scheduler bookkeepingを定義する。"""

from dataclasses import dataclass, field
from typing import Any

from .execution_trace import ScheduledOperation
from .magic_state import MagicStateRuntime


@dataclass
class ExecutionStatistics:
    """machine実行中に加算される評価counterと集計mapを保持する。"""

    resource_busy: dict[str, int] = field(default_factory=dict)
    wait_by_category: dict[str, int] = field(default_factory=dict)
    cache_origin: dict[int, str] = field(default_factory=dict)
    ld_count: int = 0
    st_count: int = 0
    cache_prefetch_count: int = 0
    cache_hit_count: int = 0
    inmemory_count: int = 0
    route_wait: int = 0
    compute_beats: int = 0
    inmemory_beats: int = 0
    transfer_beats: int = 0
    prefetch_hidden_hits: int = 0
    demand_cache_loads: int = 0
    local_transfers: int = 0
    nonlocal_transfers: int = 0
    cache_evict_count: int = 0
    cr_to_cr_transfer_count: int = 0
    max_cr_occupancy: int = 0
    cr_overflow_events: int = 0
    cr_slot_eviction_events: int = 0
    planned_load_count: int = 0
    planned_store_count: int = 0
    collective_rotation_count: int = 0
    collective_rotation_qubits: int = 0


@dataclass
class SchedulingState:
    """decision traceに必要なscheduler上の現在位置だけを保持する。"""

    decision_trace: list[dict[str, Any]] | None = None
    active_node_idx: int | None = None
    active_decision_id: int | None = None


@dataclass
class MachineState:
    """qubit配置、resident、資源可用時刻、実行記録を保持する。"""

    # SAM:   ("SAM", bank, row, col)
    # CACHE: ("CACHE", bank, slot, col_unused)
    # CR:    ("CR", cr, row, col)
    loc: dict[int, tuple[str, int, int, int]]
    home: dict[int, tuple[int, int, int]]
    sam_cells: dict[int, set[tuple[int, int]]]
    cr_res: dict[int, set[int]]
    cache_res: dict[int, list[int | None]]
    resource_until: dict[str, int] = field(default_factory=dict)
    q_ready: dict[int, int] = field(default_factory=dict)
    bank_head: dict[int, int] = field(default_factory=dict)
    trace: list[ScheduledOperation] = field(default_factory=list)
    statistics: ExecutionStatistics = field(default_factory=ExecutionStatistics)
    scheduling: SchedulingState = field(default_factory=SchedulingState)
    # 各CRに固定されたMSFとlocal bufferの実行状態。
    magic_state: MagicStateRuntime | None = None
