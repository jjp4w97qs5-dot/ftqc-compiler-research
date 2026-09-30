from __future__ import annotations

"""LSQCAの構成、接続性、latency、gate実行条件を定義する。"""

from dataclasses import dataclass
import math
from typing import Any

from .magic_state import MagicStateConfig


# 既存machine modelのin-memory gateとgate latency。
SINGLE_INMEMORY_DEFAULT = frozenset({"H", "S", "X", "Z", "MEASURE", "RESET"})
BASE_LATENCY = {"CX": 1, "H": 3, "S": 2, "T": 3, "TDG": 3, "CCZ": 1}

# Point-SAMのaccess port座標。
POINT_PORT = (0, 0)


@dataclass(frozen=True)
class ArchitectureConfig:
    """LSQCA machineの容量、接続性、gate実行条件を保持する。"""

    # Core hardware parameters.
    cr_count: int
    cr_slots: int
    banks: int
    sam_type: str = "line-sam"
    rotation_epsilon: float = 1e-5

    # Small staging cache between SAM and CR.  It is for prefetch/temporary
    # transfer only, not a large normal memory.
    cache_slots_per_bank: int = 1

    # Connectivity.  If omitted, bank b is local to CR b mod cr_count.
    local_crs_by_bank: dict[int, tuple[int, ...]] | None = None
    nonlocal_hub_count: int = 1
    direct_cr_transfer: bool = True
    cr_to_cr_latency: int = 1

    allow_in_memory_single_qubit: bool = True
    in_memory_single_qubit_ops: frozenset[str] = SINGLE_INMEMORY_DEFAULT
    rz_requires_cr: bool = True
    t_requires_cr: bool = True
    # 各CRに固定するmagic-state供給設定。Noneは従来modelを厳密に維持する。
    magic_state: MagicStateConfig | None = None

    def __post_init__(self) -> None:
        """Local MSFと矛盾するin-memory magic gate設定を拒否する。"""

        if self.magic_state is not None and not self.t_requires_cr:
            raise ValueError("magic-state tracking requires t_requires_cr=True")
        if self.magic_state is not None and not self.rz_requires_cr:
            raise ValueError("magic-state tracking requires rz_requires_cr=True")

    def local_crs(self, bank: int) -> tuple[int, ...]:
        """指定bankからlocal接続される有効CRを返す。"""

        if self.local_crs_by_bank is None:
            return (bank % max(1, self.cr_count),)
        return tuple(c for c in self.local_crs_by_bank.get(bank, ()) if 0 <= c < self.cr_count)

    def is_local(self, bank: int, cr: int) -> bool:
        """bankとCRがlocal接続かを返す。"""

        return cr in self.local_crs(bank)

    def hub_for(self, bank: int, cr: int) -> str:
        """nonlocal transferが使用するhub資源名を返す。"""

        if self.nonlocal_hub_count <= 1:
            return "hub:global"
        return f"hub:{(bank + cr) % self.nonlocal_hub_count}"


def point_transport(row: int, col: int) -> int:
    """Point-SAM cellからportへのtransport costを既存実験式で返す。"""

    port_row, port_col = POINT_PORT
    vertical = abs(row - port_row)
    horizontal = abs(col - port_col)
    return 6 * min(horizontal, vertical) + 5 * abs(horizontal - vertical)


def point_port_distance(row: int, col: int) -> int:
    """scan距離とtransportを合わせたport access cost proxyを返す。"""

    port_row, port_col = POINT_PORT
    return abs(row - port_row) + abs(col - port_col) + point_transport(row, col)


def _magic_need(op: str, eps: float) -> int:
    """gate合成に必要な既存magic-state数を返す。"""

    name = op.upper()
    if name in {"T", "TDG"}:
        return 1
    if name == "RZ":
        return int(math.ceil(3.0 * math.log2(1.0 / eps)))
    return 0


def _gate_latency(op: str, eps: float) -> int:
    """既存latency式からgate beat数を返す。"""

    name = op.upper()
    if name == "RZ":
        return 3 * _magic_need(name, eps)
    return BASE_LATENCY.get(name, 1)


def _requires_cr(op: str, cfg: ArchitectureConfig) -> bool:
    """既存実行条件によりgateがCRを必要とするか判定する。"""

    name = op.upper()
    if name == "CX":
        return True
    if name == "CCZ":
        return True
    if name in {"T", "TDG"}:
        return cfg.t_requires_cr
    if name == "RZ":
        return cfg.rz_requires_cr
    if cfg.allow_in_memory_single_qubit and name in cfg.in_memory_single_qubit_ops:
        return False
    return True


def _route_resources(
    bank: int,
    cr: int,
    cfg: ArchitectureConfig,
    *,
    include_bank: bool,
    include_cr_port: bool,
) -> tuple[list[str], bool]:
    """bank–CR接続に対応する既存resource列とlocalityを返す。"""

    resources: list[str] = []
    if include_bank:
        resources.extend([f"bank:{bank}", f"mem_port:{bank}"])
    if include_cr_port:
        resources.append(f"cr_port:{cr}")
    if cfg.is_local(bank, cr):
        resources.append(f"local_route:{bank}:{cr}")
        return resources, True
    resources.append(cfg.hub_for(bank, cr))
    resources.append(f"nonlocal_route:{bank}:{cr}")
    return resources, False


def _sam_seek_latency(
    cfg: ArchitectureConfig,
    st: Any,
    bank: int,
    row: int,
    col: int,
    *,
    op: str,
) -> int:
    """既存SAM access式からLD/ST latencyを返す。"""

    op_const = 2 if op == "LD" else 1
    if cfg.sam_type == "point-sam":
        return point_port_distance(row, col) + op_const
    return abs(st.bank_head.get(bank, 0) - row) + op_const
