from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class Program:
    # 高水準構造を保持する前段IR全体。
    name: str
    qubits: list[str]
    body: list["Stmt"]


@dataclass
class Block:
    # subroutine/layer/scopeなどの構造境界を表す。
    kind: str
    name: str
    body: list["Stmt"]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Op:
    # 高水準IR上の量子ゲート。LSQCA命令ではない。
    gate: str
    qubits: tuple[str, ...]
    params: tuple[Any, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanDirective:
    # 高水準plannerがschedulerへ渡す構造・資源計画directive。
    kind: str
    qubits: tuple[str, ...] = ()
    groups: tuple[tuple[str, ...], ...] = ()
    policy: str = "affinity"
    meta: dict[str, Any] = field(default_factory=dict)


Stmt = Union[Block, Op, PlanDirective]
