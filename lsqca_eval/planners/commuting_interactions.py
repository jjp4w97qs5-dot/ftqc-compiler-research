from __future__ import annotations

"""Generic high-level planner for commuting two-qubit interaction regions.

The planner is deliberately independent of the scheduler and of concrete
program names.  A caller provides an ordered qubit list and a set of commuting
pair interactions.  The planner then:

1. partitions the ordered qubit list into balanced CR ownership blocks;
2. assigns each interaction to one of its endpoint owners;
3. orders each CR stream to maximize consecutive endpoint reuse;
4. packs one interaction per active CR into explicit parallel waves.

The resulting waves can be encoded as ordinary PARALLEL_GROUPS directives.  The
runtime scheduler remains shared with all other workloads.
"""

from dataclasses import dataclass
from typing import Iterable, Iterator

from ..program_ir import Block, PlanDirective, Program, Stmt


@dataclass(frozen=True)
class PlannedPair:
    left: str
    right: str
    payload: tuple[object, ...] = ()

    @property
    def qubits(self) -> tuple[str, str]:
        return (self.left, self.right)


@dataclass(frozen=True)
class PlannedWave:
    groups: tuple[PlannedPair, ...]
    crs: tuple[int, ...]


@dataclass(frozen=True)
class PairInteractionPlan:
    cr_count: int
    cr_slots: int
    owner_by_qubit: dict[str, int]
    streams: dict[int, tuple[PlannedPair, ...]]
    waves: tuple[PlannedWave, ...]


@dataclass(frozen=True)
class InteractionArchitecture:
    """可換interaction実験で使用するCR・slot構成の推定結果。"""

    cr_count: int
    cr_slots: int
    banks: int
    cache_slots_per_bank: int
    max_parallel_groups: int
    max_group_size: int


def _walk(stmts: Iterable[Stmt]) -> Iterator[Stmt]:
    # Program内のBlockを再帰的にたどる。
    for stmt in stmts:
        yield stmt
        if isinstance(stmt, Block):
            yield from _walk(stmt.body)


def infer_interaction_architecture(program: Program, *, max_cr_count: int = 4) -> InteractionArchitecture:
    """宣言済みparallel group幅から既存規則で実験用資源数を決める。"""

    starts = [
        stmt for stmt in _walk(program.body)
        if isinstance(stmt, PlanDirective) and stmt.kind.upper() == "PARALLEL_GROUPS_BEGIN"
    ]
    if not starts:
        return InteractionArchitecture(1, 4, 4, 2, 1, 1)
    max_parallel = max(len(item.groups) for item in starts)
    max_group_size = max(len(group) for item in starts for group in item.groups)
    return InteractionArchitecture(
        cr_count=max(1, min(max_cr_count, max_parallel)),
        cr_slots=max(4, 2 * max_group_size),
        banks=4,
        cache_slots_per_bank=2,
        max_parallel_groups=max_parallel,
        max_group_size=max_group_size,
    )


def _balanced_owner(qubits: list[str], cr_count: int) -> dict[str, int]:
    if cr_count <= 0:
        raise ValueError("cr_count must be positive")
    n = max(1, len(qubits))
    return {
        q: min(cr_count - 1, index * cr_count // n)
        for index, q in enumerate(qubits)
    }


def _edge_key(pair: PlannedPair) -> tuple[str, str, tuple[object, ...]]:
    return (pair.left, pair.right, pair.payload)


def _order_stream(pairs: list[PlannedPair]) -> tuple[PlannedPair, ...]:
    if not pairs:
        return ()
    remaining = sorted(pairs, key=_edge_key)
    ordered = [remaining.pop(0)]
    while remaining:
        current = set(ordered[-1].qubits)
        def key(pair: PlannedPair) -> tuple[int, int, tuple[str, str, tuple[object, ...]]]:
            shared = len(current & set(pair.qubits))
            # A weak ordered-qubit tie-break: nearby lexical names tend to
            # correspond to nearby indices in the current builders.
            lexical_gap = min(
                abs(_suffix_int(a) - _suffix_int(b))
                for a in current
                for b in pair.qubits
            )
            return (-shared, lexical_gap, _edge_key(pair))
        next_pair = min(remaining, key=key)
        remaining.remove(next_pair)
        ordered.append(next_pair)
    return tuple(ordered)


def _suffix_int(name: str) -> int:
    digits = ""
    for char in reversed(name):
        if not char.isdigit():
            break
        digits = char + digits
    return int(digits) if digits else 0


def plan_commuting_pairs(
    qubits: Iterable[str],
    pairs: Iterable[PlannedPair],
    *,
    cr_count: int,
) -> PairInteractionPlan:
    ordered_qubits = list(qubits)
    owner = _balanced_owner(ordered_qubits, cr_count)
    loads = {cr: 0 for cr in range(cr_count)}
    assigned: dict[int, list[PlannedPair]] = {cr: [] for cr in range(cr_count)}

    internal: list[PlannedPair] = []
    boundary: list[PlannedPair] = []
    for pair in pairs:
        left_owner = owner[pair.left]
        right_owner = owner[pair.right]
        (internal if left_owner == right_owner else boundary).append(pair)

    # Internal interactions stay on their owner CR.  Boundary interactions are
    # assigned to the less-loaded endpoint owner, preserving locality at one end
    # without introducing an unrelated third CR.
    for pair in sorted(internal, key=_edge_key):
        cr = owner[pair.left]
        assigned[cr].append(pair)
        loads[cr] += 1
    for pair in sorted(boundary, key=_edge_key):
        candidates = (owner[pair.left], owner[pair.right])
        cr = min(candidates, key=lambda c: (loads[c], c))
        assigned[cr].append(pair)
        loads[cr] += 1

    streams = {cr: _order_stream(items) for cr, items in assigned.items()}
    waves: list[PlannedWave] = []
    positions = {cr: 0 for cr in range(cr_count)}
    while any(positions[cr] < len(streams[cr]) for cr in range(cr_count)):
        groups: list[PlannedPair] = []
        crs: list[int] = []
        used_qubits: set[str] = set()
        for cr in range(cr_count):
            stream = streams[cr]
            position = positions[cr]
            if position >= len(stream):
                continue
            pair = stream[position]
            if used_qubits & set(pair.qubits):
                continue
            groups.append(pair)
            crs.append(cr)
            used_qubits.update(pair.qubits)
            positions[cr] += 1
        if not groups:
            # This can only occur when all active stream fronts conflict.  Pick
            # the lowest-CR front; subsequent waves will make progress while
            # preserving every per-CR stream order.
            cr = min(cr for cr in range(cr_count) if positions[cr] < len(streams[cr]))
            pair = streams[cr][positions[cr]]
            groups.append(pair)
            crs.append(cr)
            positions[cr] += 1
        waves.append(PlannedWave(tuple(groups), tuple(crs)))

    max_group_size = 2
    return PairInteractionPlan(
        cr_count=cr_count,
        cr_slots=max(4, 2 * max_group_size),
        owner_by_qubit=owner,
        streams={cr: tuple(stream) for cr, stream in streams.items()},
        waves=tuple(waves),
    )
