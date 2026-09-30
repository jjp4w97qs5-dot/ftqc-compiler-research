from __future__ import annotations

"""構造付きProgram IRをschedulerが扱う整数qubit IDのevent列へloweringする。"""

from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .metadata import META_ANGLE, META_GROUPS, META_PARAMS, META_POLICY
from .program_ir import Block, PlanDirective, Op, Program, Stmt


@dataclass(frozen=True)
class LoweredOp:
    """整数qubit IDへ解決済みの論理演算。"""

    op: str
    qubits: tuple[int, ...]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoweredDirective:
    """整数qubit IDへ解決済みの高水準計画annotation。"""

    kind: str
    qubits: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


LoweredEvent: TypeAlias = LoweredOp | LoweredDirective


@dataclass(frozen=True)
class LoweredProgram:
    """qubit名解決後の線形event列。Block境界は現行実験と同様にflattenする。"""

    name: str
    qubit_names: tuple[str, ...]
    events: tuple[LoweredEvent, ...]


def lower_program(program: Program) -> LoweredProgram:
    """Programを整数IDのevent列へ変換する。演算順・plan directive内容は変更しない。"""

    name_to_id = _qubit_index(program)
    out: list[LoweredEvent] = []
    for stmt in program.body:
        _lower_stmt(stmt, out, name_to_id)
    return LoweredProgram(program.name, tuple(program.qubits), tuple(out))


def _qubit_index(program: Program) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for name in program.qubits:
        if name in mapping:
            raise ValueError(f"Duplicate qubit name: {name}")
        mapping[name] = len(mapping)
    return mapping


def _qid(mapping: dict[str, int], name: str) -> int:
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unknown qubit in Program: {name}") from exc


def _qids(mapping: dict[str, int], qubits: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(_qid(mapping, q) for q in qubits)


def _lower_stmt(stmt: Stmt, out: list[LoweredEvent], mapping: dict[str, int]) -> None:
    if isinstance(stmt, Op):
        meta = dict(stmt.meta)
        gate = stmt.gate.upper()
        if stmt.params:
            if gate == "RZ" and len(stmt.params) == 1:
                meta[META_ANGLE] = stmt.params[0]
            else:
                meta[META_PARAMS] = tuple(stmt.params)
        out.append(LoweredOp(gate, _qids(mapping, stmt.qubits), meta))
        return
    if isinstance(stmt, PlanDirective):
        groups = tuple(tuple(_qid(mapping, q) for q in group) for group in stmt.groups)
        qubits = _qids(mapping, stmt.qubits)
        if not qubits and groups:
            qubits = tuple(dict.fromkeys(q for group in groups for q in group))
        meta = dict(stmt.meta)
        meta[META_POLICY] = stmt.policy
        if groups:
            meta[META_GROUPS] = groups
        out.append(LoweredDirective(stmt.kind.upper(), qubits, meta))
        return
    if isinstance(stmt, Block):
        for child in stmt.body:
            _lower_stmt(child, out, mapping)
        return
    raise TypeError(f"Unsupported Program statement: {type(stmt)!r}")
