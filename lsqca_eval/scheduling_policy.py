from __future__ import annotations

"""LSQCA schedulerの最適化policyと合成実行設定を定義する。"""

from dataclasses import dataclass, field
from typing import Literal

from .architecture import ArchitectureConfig


# store、初期配置、prefetchで使用する既存policy名。
StorePolicy = Literal["home", "least_busy", "next_use_local", "spread_banks", "score"]
PlacementPolicy = Literal[
    "access_order",
    "sequential",
    "round_robin",
    "first_use_local",
    "plan_affinity",
]
PrefetchPolicy = Literal["none", "compute_window_demand_first"]


@dataclass(frozen=True)
class SchedulingPolicy:
    """配置、転送、prefetch、plan解釈に関する既存policyを保持する。"""

    store_policy: StorePolicy = "home"
    placement_policy: PlacementPolicy = "access_order"
    use_staging_prefetch: bool = False
    stage_demand_loads: bool = True
    prefetch_policy: PrefetchPolicy = "none"
    max_prefetch_per_step: int = 2
    use_execution_plan: bool = False
    execute_parallel_group_plan: bool = False
    enforce_planned_cr_assignment: bool = False
    overlap_group_transfers: bool = False
    retain_cr_residents: bool = True
    final_flush: bool = False
    parallel_group_assignment: str = "locality_distinct"
    record_plan_metadata: bool = False


@dataclass(frozen=True)
class ExecutionConfig:
    """machine architectureとscheduler policyを明示的に合成する。"""

    architecture: ArchitectureConfig
    policy: SchedulingPolicy = field(default_factory=SchedulingPolicy)
