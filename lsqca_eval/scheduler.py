from __future__ import annotations

"""LSQCAのSAM/CRメモリ階層と資源競合を扱う実行scheduler。

Program IRをloweringした論理演算列に対し、qubit位置、CR容量、bank/port/route、
staging cache、gate latencyを追跡してScheduledOperation列と集計値を生成する。
高水準plannerから渡されたparallel-group/CR移動計画も同じ資源モデルで実行する。
"""

from collections import defaultdict
from dataclasses import dataclass, field
import heapq
import math
from collections import deque
from typing import Any

from .architecture import (
    _gate_latency,
    _requires_cr,
    _sam_seek_latency,
    point_port_distance,
)
from .scheduling_policy import ExecutionConfig
from .execution_trace import ScheduledOperation
from .machine import (
    append_trace,
    execute_cr_gate,
    execute_in_memory_gate,
    initialize_machine_state,
    load_sam_to_cache,
    load_sam_to_cr,
    store_cache_to_sam,
    store_cr_to_sam,
    transfer_cache_to_cr,
    transfer_cr_to_cache,
    transfer_cr_to_cr,
    update_qubit_ready,
)
from .runtime_state import MachineState
from .lowering import LoweredDirective, LoweredOp, lower_program
from .sam_layout import sam_coord
from .metadata import META_ANGLE, META_CR_IDS, META_DST_CRS, META_GROUPS


# scheduler内部のnext-use未到達値。
NEXT_USE_INF = 10**12


@dataclass(frozen=True)
class OperationNode:
    """lowering結果とplan metadataを束ねるscheduler内部型。"""

    idx: int
    op: str
    qubits: tuple[int, ...]
    angle: float | None = None
    preferred_cr: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _free_cache_slot(st: MachineState, bank: int) -> int | None:
    """指定bankの先頭空きcache slotを既存候補順で選ぶ。"""

    for idx, q in enumerate(st.cache_res.get(bank, [])):
        if q is None:
            return idx
    return None


def _occupied_sam_cells(st: MachineState, bank: int) -> set[tuple[int, int]]:
    """指定bankで現在qubitが占有するSAM cell集合を返す。"""

    return {
        (int(loc[2]), int(loc[3]))
        for loc in st.loc.values()
        if loc[0] == "SAM" and int(loc[1]) == bank
    }


def _free_sam_cells(st: MachineState, bank: int) -> set[tuple[int, int]]:
    """指定bankの固定layout内にある空きSAM cell集合を返す。"""

    return set(st.sam_cells.get(bank, set())) - _occupied_sam_cells(st, bank)


def _sam_occupancy(st: MachineState, bank: int) -> int:
    """指定bankの現在SAM resident数を返す。"""

    return sum(1 for loc in st.loc.values() if loc[0] == "SAM" and loc[1] == bank)


def _cache_occupancy(st: MachineState, bank: int) -> int:
    """指定bankの現在cache resident数を返す。"""

    return sum(1 for q in st.cache_res.get(bank, []) if q is not None)


def _cr_store_banks_with_capacity(st: MachineState, cfg: ExecutionConfig) -> list[int]:
    """CR residentを受け入れ可能なstore bank候補を既存順で返す。"""

    return [
        bank
        for bank in range(cfg.architecture.banks)
        if len(_free_sam_cells(st, bank)) > _cache_occupancy(st, bank)
    ]


def _allocate_store_cell(
    q: int,
    bank: int,
    st: MachineState,
    cfg: ExecutionConfig,
) -> tuple[int, int]:
    """既存geometry規則で選択済みbank内のstore cellを選ぶ。"""

    free = _free_sam_cells(st, bank)
    if not free:
        raise RuntimeError(f"No free SAM cell for q={q}, bank={bank}")
    home_bank, home_row, home_col = st.home[q]
    if home_bank == bank and (home_row, home_col) in free:
        return home_row, home_col
    if cfg.architecture.sam_type == "point-sam":
        return min(free, key=lambda cell: (point_port_distance(*cell), cell[0], cell[1]))
    head = st.bank_head.get(bank, 0)
    return min(free, key=lambda cell: (abs(head - cell[0]), cell[0], cell[1]))


def _store_bank(
    q: int,
    next_cr: int | None,
    st: MachineState,
    cfg: ExecutionConfig,
) -> int:
    """既存store policyとtie-breakで書戻しbankを選ぶ。"""

    home_bank, _, _ = st.home[q]
    feasible = _cr_store_banks_with_capacity(st, cfg)
    if not feasible:
        raise RuntimeError(f"No SAM bank can accept q={q}; free-cell invariant violated")
    if cfg.policy.store_policy == "home" or next_cr is None:
        return (
            home_bank
            if home_bank in feasible
            else min(feasible, key=lambda b: (-len(_free_sam_cells(st, b)), b))
        )
    if cfg.policy.store_policy == "next_use_local":
        candidates = [b for b in feasible if cfg.architecture.is_local(b, next_cr)]
        if candidates:
            return min(
                candidates,
                key=lambda b: (
                    st.resource_until.get(f"bank:{b}", 0),
                    _cache_occupancy(st, b),
                    b,
                ),
            )
    if cfg.policy.store_policy == "least_busy":
        return min(
            feasible,
            key=lambda b: (
                st.resource_until.get(f"bank:{b}", 0),
                _cache_occupancy(st, b),
                b,
            ),
        )
    if cfg.policy.store_policy == "spread_banks":
        return min(
            feasible,
            key=lambda b: (
                _sam_occupancy(st, b),
                st.resource_until.get(f"bank:{b}", 0),
                b,
            ),
        )
    if cfg.policy.store_policy == "score":
        best_bank = feasible[0]
        best_score = 10**18
        for bank in feasible:
            score = 10 * _sam_occupancy(st, bank)
            score += st.resource_until.get(f"bank:{bank}", 0) // 100
            score += 5 * _cache_occupancy(st, bank)
            if next_cr is not None and not cfg.architecture.is_local(bank, next_cr):
                score += 30
            if bank == home_bank:
                score -= 2
            if score < best_score:
                best_bank, best_score = bank, score
        return best_bank
    return home_bank if home_bank in feasible else feasible[0]


def _flush_cache_to_sam(
    q: int,
    t: int,
    st: MachineState,
    cfg: ExecutionConfig,
    *,
    reason: str = "final_cache_flush",
) -> int:
    """cache residentのstore cellを選びmachine primitiveへ渡す。"""

    where, bank, _, _ = st.loc[q]
    if where != "CACHE":
        return max(t, st.q_ready.get(q, 0))
    row, col = _allocate_store_cell(q, bank, st, cfg)
    return store_cache_to_sam(
        q,
        t,
        st,
        cfg.architecture,
        row=row,
        col=col,
        reason=reason,
    )


def _ensure_cache_slot(
    bank: int,
    t: int,
    st: MachineState,
    cfg: ExecutionConfig,
    next_use_after: dict[int, int],
) -> tuple[int, int]:
    """既存next-use規則でcache slotまたはvictimを選ぶ。"""

    slot = _free_cache_slot(st, bank)
    if slot is not None:
        return t, slot
    entries = [
        (idx, q)
        for idx, q in enumerate(st.cache_res.get(bank, []))
        if q is not None
    ]
    if not entries:
        raise RuntimeError(f"No cache slot data for full bank cache: bank={bank}")
    idx, victim = max(
        entries,
        key=lambda item: (
            next_use_after.get(int(item[1]), NEXT_USE_INF),
            int(item[1]),
        ),
    )
    assert victim is not None
    row, col = _allocate_store_cell(int(victim), bank, st, cfg)
    end = store_cache_to_sam(
        int(victim),
        t,
        st,
        cfg.architecture,
        row=row,
        col=col,
        trace_op="CACHE_EVICT_ST",
        reason="cache_evict",
        cache_eviction=True,
    )
    return end, idx


def _sam_to_cache(
    q: int,
    t: int,
    st: MachineState,
    cfg: ExecutionConfig,
    next_use_after: dict[int, int],
    next_cr_by_node: dict[int, int | None],
    *,
    reason: str,
) -> int:
    """cache slotを選択してSAM→cache primitiveを実行する。"""

    where, bank, _, _ = st.loc[q]
    if where in {"CACHE", "CR"}:
        return max(t, st.q_ready.get(q, 0))
    t, slot = _ensure_cache_slot(bank, t, st, cfg, next_use_after)
    return load_sam_to_cache(
        q,
        t,
        st,
        cfg.architecture,
        slot=slot,
        reason=reason,
    )


def _store_from_cr(
    q: int,
    t: int,
    st: MachineState,
    cfg: ExecutionConfig,
    *,
    next_use_cr: int | None,
) -> int:
    """store bank/cellを選択してCR→SAM primitiveを実行する。"""

    if st.loc[q][0] != "CR":
        return max(t, st.q_ready.get(q, 0))
    bank = _store_bank(q, next_use_cr, st, cfg)
    row, col = _allocate_store_cell(q, bank, st, cfg)
    return store_cr_to_sam(
        q,
        t,
        st,
        cfg.architecture,
        bank=bank,
        row=row,
        col=col,
    )


def _load_to_cr(
    q: int,
    cr: int,
    t: int,
    st: MachineState,
    cfg: ExecutionConfig,
    next_use_after: dict[int, int],
    next_cr_by_node: dict[int, int | None],
) -> int:
    """現在locationとpolicyからload pathを選びmachine primitiveを実行する。"""

    where = st.loc[q][0]
    if where == "CR" and st.loc[q][1] == cr:
        return max(t, st.q_ready.get(q, 0))
    if where == "CR" and st.loc[q][1] != cr:
        if cfg.architecture.direct_cr_transfer:
            return transfer_cr_to_cr(q, cr, t, st, cfg.architecture)
        t = _store_from_cr(q, t, st, cfg, next_use_cr=cr)
        where = st.loc[q][0]
    if where == "SAM":
        if (
            cfg.architecture.cache_slots_per_bank <= 0
            or not cfg.policy.stage_demand_loads
        ):
            return load_sam_to_cr(
                q,
                cr,
                t,
                st,
                cfg.architecture,
                reason="direct_demand",
            )
        t = _sam_to_cache(
            q,
            t,
            st,
            cfg,
            next_use_after,
            next_cr_by_node,
            reason="demand",
        )
    if st.loc[q][0] == "CACHE":
        return transfer_cache_to_cr(q, cr, t, st, cfg.architecture)
    raise RuntimeError(f"Cannot load q={q} from loc={st.loc[q]}")


def _final_flush_all(
    t: int,
    st: MachineState,
    cfg: ExecutionConfig,
) -> int:
    """全cache/CR residentのstore選択と実行を従来順で行う。"""

    for _, slots in list(st.cache_res.items()):
        for q in list(slots):
            if q is not None:
                t = _flush_cache_to_sam(int(q), t, st, cfg)
    for cr in range(cfg.architecture.cr_count):
        for q in list(st.cr_res.get(cr, set())):
            t = _store_from_cr(int(q), t, st, cfg, next_use_cr=None)
    return t

def _evict_for_space(cr: int, need_slots: int, t: int, st: MachineState, cfg: ExecutionConfig, next_use_after: dict[int, int], next_cr_by_node: dict[int, int | None], protected: set[int] | None = None) -> int:
    """指定CRに必要slot数を空けるまで既存next-use規則でevictする。"""

    protected = protected or set()
    while len(st.cr_res.get(cr, set())) + need_slots > cfg.architecture.cr_slots:
        victims = [q for q in st.cr_res[cr] if q not in protected]
        if not victims:
            # This indicates an infeasible gate for the configured CR slot count.
            # We record it explicitly instead of silently overflowing.
            st.statistics.cr_overflow_events += 1
            break
        victim = max(victims, key=lambda q: (next_use_after.get(q, NEXT_USE_INF), q))
        st.statistics.cr_slot_eviction_events += 1
        next_node = next_use_after.get(victim, -1)
        t = _store_from_cr(victim, t, st, cfg, next_use_cr=next_cr_by_node.get(next_node))
    return t


def _schedule_prefetches(
    compute_start: int,
    compute_end: int,
    current: OperationNode,
    next_use_after: dict[int, int],
    st: MachineState,
    cfg: ExecutionConfig,
    *,
    demand_banks: set[int] | None = None,
    staging_reserve: int = 0,
) -> None:
    """compute window内へ収まる既存候補順のprefetchを発行する。"""

    if not cfg.policy.use_staging_prefetch or cfg.architecture.cache_slots_per_bank <= 0:
        return
    if cfg.policy.prefetch_policy == "none":
        return
    candidates: list[tuple[int, int]] = []
    current_qs = set(map(int, current.qubits))
    blocked = demand_banks or set()
    for q, node_idx in next_use_after.items():
        q = int(q)
        if q in current_qs:
            continue
        loc = st.loc.get(q, ("", 0, 0, 0))
        if loc[0] != "SAM":
            continue
        if int(loc[1]) in blocked:
            continue
        candidates.append((int(node_idx), q))
    candidates.sort()
    issued = 0
    free_slots = sum(item is None for slots in st.cache_res.values() for item in slots)
    prefetch_limit = min(cfg.policy.max_prefetch_per_step, max(0, free_slots - staging_reserve))
    dummy_next_cr: dict[int, int | None] = {}
    for _, q in candidates:
        if issued >= prefetch_limit:
            break
        _, bank, row, col = st.loc[q]
        slot = _free_cache_slot(st, bank)
        if slot is None:
            continue
        resources = [f"bank:{bank}", f"mem_port:{bank}", f"cache_slot:{bank}:{slot}"]
        earliest = max(compute_start, st.q_ready.get(q, 0))
        start = max([earliest, *(st.resource_until.get(r, 0) for r in resources)])
        dur = _sam_seek_latency(cfg.architecture, st, bank, row, col, op="LD")
        if start + dur > compute_end:
            continue
        actual_end = _sam_to_cache(q, compute_start, st, cfg, next_use_after, dummy_next_cr, reason="prefetch")
        if actual_end > compute_end:
            raise RuntimeError("compute-window prefetch escaped its reserved compute interval")
        issued += 1



def _planned_qubits(node: OperationNode) -> tuple[int, ...]:
    """plan metadataが指定する対象qubit列を既存優先順で返す。"""

    groups = node.meta.get(META_GROUPS) or ()
    if groups:
        return tuple(int(q) for q in groups[0])
    return tuple(int(q) for q in node.qubits)


def _rotation_staging_reserve(nodes: list[OperationNode]) -> dict[int, int]:
    """plan-marked gateごとに次のcollective rotation必要slot数を返す。"""

    reserve = 0
    by_node: dict[int, int] = {}
    for node in reversed(nodes):
        if node.op in {"CR_LOAD", "CR_STORE", "PHASE_BARRIER"}:
            reserve = 0
        elif node.op == "CR_ROTATE":
            reserve = len(_planned_qubits(node))
        elif node.meta.get("_plan_no_prefetch") and reserve > 0:
            by_node[node.idx] = reserve
    return by_node


def _schedule_collective_cr_rotation(
    node: OperationNode,
    t_floor: int,
    st: MachineState,
    cfg: ExecutionConfig,
) -> int:
    """CR resident qubit群をstaging cache経由で一斉に別CRへ移す。"""
    dst_crs = tuple(int(c) for c in node.meta.get(META_DST_CRS, ()))
    move_qubits = _planned_qubits(node)
    if len(dst_crs) != len(move_qubits):
        raise ValueError("CR_ROTATE requires one destination CR per moved qubit")
    earliest = max([t_floor, *(st.q_ready.get(q, 0) for q in node.qubits)])
    staged: list[tuple[int, int, int, int]] = []
    claimed: set[tuple[int, int]] = set()
    for q, dst in zip(move_qubits, dst_crs):
        where, src, _, _ = st.loc[q]
        if where != "CR":
            raise RuntimeError(f"CR_ROTATE requires CR resident q={q}, loc={st.loc[q]}")
        candidates: list[tuple[int, int, int, int, int]] = []
        for bank in range(cfg.architecture.banks):
            for slot, item in enumerate(st.cache_res.get(bank, [])):
                if item is not None or (bank, slot) in claimed:
                    continue
                candidates.append((
                    0 if cfg.architecture.is_local(bank, dst) else 1,
                    0 if cfg.architecture.is_local(bank, src) else 1,
                    st.resource_until.get(f"cache_slot:{bank}:{slot}", 0),
                    bank,
                    slot,
                ))
        if not candidates:
            raise RuntimeError("No free staging slot for collective CR rotation")
        _, _, _, bank, slot = min(candidates)
        claimed.add((bank, slot))
        staged.append((q, dst, bank, slot))

    phase1_end = earliest
    for q, dst, bank, slot in staged:
        end = transfer_cr_to_cache(
            q,
            bank,
            slot,
            earliest,
            st,
            cfg.architecture,
            trace_op="CR_ROTATE_OUT",
            reason=f"collective_cr_rotation_out:{node.meta.get('name', '')}",
            include_transfer_metadata=True,
        )
        phase1_end = max(phase1_end, end)

    finish = phase1_end
    for q, dst, bank, slot in staged:
        end = transfer_cache_to_cr(
            q,
            dst,
            max(phase1_end, st.q_ready[q]),
            st,
            cfg.architecture,
            trace_op="CR_ROTATE_IN",
            reason=f"collective_cr_rotation_in:{node.meta.get('name', '')}",
            count_as_load=False,
            include_transfer_metadata=True,
        )
        finish = max(finish, end)
    update_qubit_ready(st, list(node.qubits), finish)
    st.statistics.collective_rotation_count += 1
    st.statistics.collective_rotation_qubits += len(staged)
    return finish


def _schedule_explicit_cr_load(
    node: OperationNode,
    t_floor: int,
    st: MachineState,
    cfg: ExecutionConfig,
    next_use_after: dict[int, int],
    next_cr_by_node: dict[int, int | None],
) -> int:
    """高水準planで指定されたqubitを指定CRへload/migrateする。"""
    move_qubits = _planned_qubits(node)
    dst_crs = tuple(int(c) for c in node.meta.get(META_DST_CRS, ()))
    if len(move_qubits) != len(dst_crs):
        raise ValueError("CR_LOAD requires one destination CR per moved qubit")
    earliest = max([t_floor, *(st.q_ready.get(q, 0) for q in node.qubits)])
    finish = earliest
    for q, dst in zip(move_qubits, dst_crs):
        local_t = earliest
        if not (st.loc[q][0] == "CR" and int(st.loc[q][1]) == dst):
            local_t = _evict_for_space(dst, 1, earliest, st, cfg, next_use_after, next_cr_by_node)
        finish = max(finish, _load_to_cr(q, dst, local_t, st, cfg, next_use_after, next_cr_by_node))
    update_qubit_ready(st, list(node.qubits), finish)
    append_trace(st, ScheduledOperation(
        "CR_LOAD_BARRIER", move_qubits, earliest, finish,
        reason=f"explicit_cr_load:{node.meta.get('name', '')}",
    ))
    return finish


def _schedule_explicit_cr_store(
    node: OperationNode,
    t_floor: int,
    st: MachineState,
    cfg: ExecutionConfig,
) -> int:
    """高水準planで指定されたqubitをSAMへstoreする。"""
    move_qubits = _planned_qubits(node)
    earliest = max([t_floor, *(st.q_ready.get(q, 0) for q in node.qubits)])
    finish = earliest
    for q in move_qubits:
        where = st.loc[q][0]
        if where == "CR":
            end = _store_from_cr(q, earliest, st, cfg, next_use_cr=None)
        elif where == "CACHE":
            end = _flush_cache_to_sam(q, earliest, st, cfg, reason="explicit_cr_store")
        else:
            end = max(earliest, st.q_ready.get(q, 0))
        finish = max(finish, end)
    update_qubit_ready(st, list(node.qubits), finish)
    append_trace(st, ScheduledOperation(
        "CR_STORE_BARRIER", move_qubits, earliest, finish,
        reason=f"explicit_cr_store:{node.meta.get('name', '')}",
    ))
    return finish


def _schedule_cr_gate(
    node: OperationNode,
    cr: int,
    t_floor: int,
    st: MachineState,
    cfg: ExecutionConfig,
    next_use_after: dict[int, int],
    next_cr_by_node: dict[int, int | None],
    *,
    demand_banks: set[int] | None = None,
    staging_reserve: int = 0,
) -> int:
    """operand loadからcompute・overlap・prefetch・storeまでを既存順で実行する。"""

    t = max([t_floor, *(st.q_ready.get(q, 0) for q in node.qubits)]) if node.qubits else t_floor
    protected: set[int] = set(q for q in node.qubits if st.loc[q][0] == "CR" and st.loc[q][1] == cr)
    for q in node.qubits:
        if not (st.loc[q][0] == "CR" and st.loc[q][1] == cr):
            t = _evict_for_space(cr, 1, t, st, cfg, next_use_after, next_cr_by_node, protected=protected)
        t = _load_to_cr(q, cr, t, st, cfg, next_use_after, next_cr_by_node)
        protected.add(q)
    t0, end = execute_cr_gate(
        node.op,
        node.qubits,
        node.idx,
        node.preferred_cr,
        node.meta,
        cr,
        t,
        st,
        cfg.architecture,
    )
    _schedule_group_transfer_overlap(
        node,
        cr,
        t0,
        end,
        st,
        cfg,
        next_use_after,
        next_cr_by_node,
    )
    _schedule_prefetches(
        t0,
        end,
        node,
        next_use_after,
        st,
        cfg,
        demand_banks=demand_banks,
        staging_reserve=staging_reserve,
    )
    if not cfg.policy.retain_cr_residents:
        for q in node.qubits:
            next_node = next_use_after.get(q, -1)
            end = _store_from_cr(q, end, st, cfg, next_use_cr=next_cr_by_node.get(next_node))
    return end


def _annotate_group_transfer_plan(nodes: list[OperationNode], cfg: ExecutionConfig) -> None:
    """Derive one reusable streaming plan from declared parallel groups.

    The planner is intentionally program-agnostic.  It only sees a sequence of
    PARALLEL_GROUPS regions and their fixed CR assignments.  For each CR it
    orders the assigned groups by source position, then derives the same three
    sets used by the hand-planned adder schedule:

    * previous-group qubits no longer needed by the current group -> store;
    * current-group qubits shared with the next group -> keep;
    * next-group qubits not present in the current group -> load.

    The longest gate in each group is selected as the overlap window.  No gate
    or latency is changed by this analysis.
    """
    groups: dict[str, list[OperationNode]] = defaultdict(list)
    for node in nodes:
        key = node.meta.get("planned_group_key")
        if key is not None:
            groups[str(key)].append(node)
    if not groups:
        return

    occurrences: list[dict[str, Any]] = []
    for key, members in groups.items():
        members.sort(key=lambda n: n.idx)
        first = members[0]
        qubits = tuple(map(int, first.meta.get("planned_group_qubits", first.qubits)))
        cr = int(first.meta["planned_cr"])
        occurrences.append({
            "key": key,
            "cr": cr,
            "qubits": qubits,
            "members": members,
            "first_idx": members[0].idx,
            "last_idx": members[-1].idx,
        })
    occurrences.sort(key=lambda item: item["first_idx"])

    by_cr: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for occurrence in occurrences:
        by_cr[int(occurrence["cr"])].append(occurrence)

    for cr, sequence in by_cr.items():
        for position, occurrence in enumerate(sequence):
            previous = set(sequence[position - 1]["qubits"]) if position > 0 else set()
            current = set(occurrence["qubits"])
            following = set(sequence[position + 1]["qubits"]) if position + 1 < len(sequence) else set()
            store_qs = tuple(sorted(previous - current))
            load_qs = tuple(sorted(following - current))
            keep_qs = tuple(sorted(current & following))
            anchor = max(
                occurrence["members"],
                key=lambda n: (_gate_latency(n.op, cfg.architecture.rotation_epsilon), -n.idx),
            )
            for member in occurrence["members"]:
                member.meta.update({
                    "planned_sequence_cr": cr,
                    "planned_sequence_position": position,
                    "planned_group_first_idx": occurrence["first_idx"],
                    "planned_group_last_idx": occurrence["last_idx"],
                })
            anchor.meta.update({
                "planned_pipeline_anchor": True,
                "planned_pipeline_store": store_qs,
                "planned_pipeline_keep": keep_qs,
                "planned_pipeline_load": load_qs,
                "planned_pipeline_next_group_first_idx": (
                    sequence[position + 1]["first_idx"] if position + 1 < len(sequence) else None
                ),
            })


def _add_parallel_group_dependencies(
    nodes: list[OperationNode],
    deps: dict[int, set[int]],
    succ: dict[int, list[int]],
) -> None:
    """Keep each planned CR stream in the source-derived group order."""
    by_cr_position: dict[tuple[int, int], tuple[int, int]] = {}
    for node in nodes:
        if "planned_sequence_cr" not in node.meta:
            continue
        key = (int(node.meta["planned_sequence_cr"]), int(node.meta["planned_sequence_position"]))
        first = int(node.meta["planned_group_first_idx"])
        last = int(node.meta["planned_group_last_idx"])
        by_cr_position[key] = (first, last)
    by_cr: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for (cr, position), (first, last) in by_cr_position.items():
        by_cr[cr].append((position, first, last))
    for stream in by_cr.values():
        stream.sort()
        for (_, _, previous_last), (_, current_first, _) in zip(stream, stream[1:]):
            if previous_last not in deps[current_first]:
                deps[current_first].add(previous_last)
                succ[previous_last].append(current_first)


def _schedule_group_transfer_overlap(
    node: OperationNode,
    cr: int,
    compute_start: int,
    compute_end: int,
    st: MachineState,
    cfg: ExecutionConfig,
    next_use_after: dict[int, int],
    next_cr_by_node: dict[int, int | None],
) -> None:
    """Overlap plan-derived stores and direct CR loads with current compute."""
    if not cfg.policy.execute_parallel_group_plan or not cfg.policy.overlap_group_transfers:
        return
    if not node.meta.get("planned_pipeline_anchor"):
        return

    current = set(map(int, node.meta.get("planned_group_qubits", node.qubits)))
    protected = set(current)
    store_qs = tuple(map(int, node.meta.get("planned_pipeline_store", ())))
    load_qs = tuple(map(int, node.meta.get("planned_pipeline_load", ())))
    next_first = node.meta.get("planned_pipeline_next_group_first_idx")

    # Previous-group values are no longer required by the current group.  Store
    # them first so their computational slots are available to the warm loads.
    for q in store_qs:
        if st.loc.get(q, ("", 0, 0, 0))[0] != "CR" or int(st.loc[q][1]) != cr:
            continue
        before = len(st.trace)
        future_node = next_use_after.get(q, -1)
        _store_from_cr(q, compute_start, st, cfg, next_use_cr=next_cr_by_node.get(future_node))
        for op in st.trace[before:]:
            op.meta["planned_pipeline"] = True
            op.meta["planned_transfer_kind"] = "store"
            op.meta["planned_anchor_node"] = node.idx
        st.statistics.planned_store_count += 1

    for q in load_qs:
        if st.loc.get(q, ("", 0, 0, 0))[0] == "CR" and int(st.loc[q][1]) == cr:
            protected.add(q)
            continue
        extended_next_use = dict(next_use_after)
        if next_first is not None:
            extended_next_use[q] = int(next_first)
        _evict_for_space(
            cr,
            1,
            compute_start,
            st,
            cfg,
            extended_next_use,
            next_cr_by_node,
            protected=protected,
        )
        before = len(st.trace)
        _load_to_cr(q, cr, compute_start, st, cfg, extended_next_use, next_cr_by_node)
        for op in st.trace[before:]:
            op.meta["planned_pipeline"] = True
            op.meta["planned_transfer_kind"] = "load"
            op.meta["planned_anchor_node"] = node.idx
        protected.add(q)
        st.statistics.planned_load_count += 1


def _parse_nodes(events: list[Any], cfg: ExecutionConfig) -> list[OperationNode]:
    """Lowered event列からscheduler用operation node列を作る。

    高水準PlanDirectiveは演算列を直接変更せず、CR affinity、並列group計画、
    明示的CR load/rotate/store、phase barrierだけをnode metadataへ反映する。
    """
    all_qubits = [int(q) for event in events for q in getattr(event, "qubits", ())]
    qubit_count = max(all_qubits, default=-1) + 1

    def block_owner(q: int) -> int:
        if qubit_count <= 0:
            return int(q) % max(1, cfg.architecture.cr_count)
        return min(
            cfg.architecture.cr_count - 1,
            int(q) * cfg.architecture.cr_count // qubit_count,
        )

    nodes: list[OperationNode] = []
    scope_pref: dict[int, int] = {}
    group_pref: dict[int, int] = {}
    group_strict = False
    group_origin: dict[int, dict[str, Any]] = {}
    group_plan: dict[int, dict[str, Any]] = {}
    last_group_cr: dict[int, int] = {}
    parallel_wave = -1

    for event in events:
        if isinstance(event, LoweredDirective):
            kind = event.kind.upper()

            # These are executable high-level plan operations, not mere affinity annotations.
            if kind in {"CR_ROTATE", "CR_LOAD", "CR_STORE", "PHASE_BARRIER"}:
                nodes.append(OperationNode(
                    len(nodes), kind, tuple(map(int, event.qubits)), None, None, dict(event.meta)
                ))
                continue

            if not cfg.policy.use_execution_plan:
                continue

            if kind == "PARALLEL_GROUPS_BEGIN":
                parallel_wave += 1
                groups = event.meta.get(META_GROUPS) or tuple((q,) for q in event.qubits)
                group_pref.clear()
                group_origin.clear()
                group_plan.clear()
                group_strict = META_CR_IDS in event.meta
                explicit_crs = tuple(event.meta.get(META_CR_IDS, ())) or tuple(event.meta.get("planned_crs", ()))
                if explicit_crs and len(explicit_crs) != len(groups):
                    raise RuntimeError("Explicit CR assignment length does not match parallel groups")
                used: set[int] = set()
                assigned_load = {c: 0 for c in range(cfg.architecture.cr_count)}
                policy = str(cfg.policy.parallel_group_assignment).lower()

                for group_index, group in enumerate(groups):
                    group = tuple(map(int, group))
                    counts: dict[int, int] = defaultdict(int)
                    for q in group:
                        bank = int(q) % cfg.architecture.banks
                        for cr in cfg.architecture.local_crs(bank):
                            counts[cr] += 1

                    if explicit_crs:
                        chosen: int | None = int(explicit_crs[group_index])
                        if not 0 <= chosen < cfg.architecture.cr_count:
                            raise RuntimeError(f"Invalid explicit CR: {chosen}")
                    elif policy == "ignore":
                        chosen = None
                    elif policy == "round_robin":
                        chosen = group_index % cfg.architecture.cr_count
                    elif policy == "reuse_balanced":
                        history = {
                            c: sum(1 for q in group if last_group_cr.get(q) == c)
                            for c in range(cfg.architecture.cr_count)
                        }
                        chosen = min(
                            range(cfg.architecture.cr_count),
                            key=lambda c: (assigned_load[c], -history[c], -counts.get(c, 0), c),
                        )
                    elif policy == "block_partition":
                        owner_counts = {
                            c: sum(1 for q in group if block_owner(q) == c)
                            for c in range(cfg.architecture.cr_count)
                        }
                        best_owner = max(owner_counts.values(), default=0)
                        candidates = [
                            c for c in range(cfg.architecture.cr_count)
                            if owner_counts[c] == best_owner
                        ]
                        history = {
                            c: sum(1 for q in group if last_group_cr.get(q) == c)
                            for c in range(cfg.architecture.cr_count)
                        }
                        chosen = min(candidates, key=lambda c: (-history[c], assigned_load[c], c))
                    elif policy in {"locality_distinct", "locality_balanced"}:
                        chosen = min(
                            range(cfg.architecture.cr_count),
                            key=lambda c: (-counts.get(c, 0), c in used, assigned_load[c], c),
                        )
                        if chosen in used and len(used) < cfg.architecture.cr_count:
                            remaining = [
                                c for c in range(cfg.architecture.cr_count) if c not in used
                            ]
                            chosen = min(remaining, key=lambda c: (-counts.get(c, 0), assigned_load[c], c))
                    else:
                        raise ValueError(
                            "Unknown parallel_group_assignment: "
                            f"{cfg.policy.parallel_group_assignment}"
                        )

                    if chosen is not None:
                        used.add(chosen)
                        assigned_load[chosen] += max(1, len(group))
                        for q in group:
                            group_pref[q] = chosen
                            last_group_cr[q] = chosen
                            if cfg.policy.execute_parallel_group_plan:
                                group_plan[q] = {
                                    "planned_wave": parallel_wave,
                                    "planned_group": group_index,
                                    "planned_group_key": f"wave{parallel_wave}:group{group_index}",
                                    "planned_group_qubits": group,
                                    "planned_cr": chosen,
                                    "planned_region": event.meta.get("name", f"parallel_wave_{parallel_wave}"),
                                }
                    if cfg.policy.record_plan_metadata:
                        region = event.meta.get("name", f"parallel_region_{len(nodes)}")
                        for q in group:
                            group_origin[q] = {
                                "declared_parallel_region": region,
                                "declared_parallel_group": group_index,
                                "declared_parallel_group_size": len(group),
                                "declared_parallel_policy": event.meta.get("policy"),
                            }
                continue

            if kind == "PARALLEL_GROUPS_END":
                group_pref.clear()
                group_origin.clear()
                group_plan.clear()
                group_strict = False
                continue

            if kind == "SCOPE_BEGIN" and event.qubits:
                counts: dict[int, int] = defaultdict(int)
                for q in event.qubits:
                    bank = int(q) % cfg.architecture.banks
                    for cr in cfg.architecture.local_crs(bank):
                        counts[cr] += 1
                chosen = min(
                    range(cfg.architecture.cr_count),
                    key=lambda c: (-counts.get(c, 0), c),
                )
                for q in event.qubits:
                    scope_pref[int(q)] = chosen
                continue

            if kind == "SCOPE_END":
                for q in event.qubits:
                    scope_pref.pop(int(q), None)
                continue

            # Historical generic memory annotations are not supported in the current execution path.
            continue

        if isinstance(event, LoweredOp):
            qs = tuple(map(int, event.qubits))
            preferred = next((group_pref[q] for q in qs if q in group_pref), None)
            if preferred is None:
                preferred = next((scope_pref[q] for q in qs if q in scope_pref), None)
            meta = dict(event.meta)
            if preferred is not None and group_strict:
                meta["strict_cr_assignment"] = True
            if cfg.policy.execute_parallel_group_plan:
                plans = [group_plan[q] for q in qs if q in group_plan]
                if plans:
                    plan = plans[0]
                    if any(item["planned_group_key"] != plan["planned_group_key"] for item in plans[1:]):
                        raise RuntimeError(f"Gate spans multiple planned groups: qubits={qs}")
                    meta.update(plan)
                    preferred = int(plan["planned_cr"])
            if cfg.policy.record_plan_metadata:
                origins = [group_origin[q] for q in qs if q in group_origin]
                if origins:
                    meta.update(origins[0])
            nodes.append(OperationNode(
                len(nodes), event.op.upper(), qs, event.meta.get(META_ANGLE), preferred, meta
            ))

    if cfg.policy.execute_parallel_group_plan:
        _annotate_group_transfer_plan(nodes, cfg)
    return nodes

def _build_deps(nodes: list[OperationNode]) -> tuple[dict[int, set[int]], dict[int, list[int]], list[int]]:
    """同一qubit上の元順からdependency、successor、ready集合を構築する。"""

    deps = {n.idx: set() for n in nodes}
    succ = {n.idx: [] for n in nodes}
    last: dict[int, int] = {}
    for n in nodes:
        for q in n.qubits:
            if q in last:
                deps[n.idx].add(last[q])
                succ[last[q]].append(n.idx)
            last[q] = n.idx
    return deps, succ, [i for i, dep in deps.items() if not dep]



def _prepare_access_queues(nodes: list[OperationNode]) -> dict[int, deque[int]]:
    """qubitごとの将来access node列を元順で構築する。"""

    queues: dict[int, deque[int]] = defaultdict(deque)
    for n in nodes:
        for q in n.qubits:
            queues[int(q)].append(n.idx)
    return queues


def _next_use_dict_for_node(
    node: OperationNode,
    st: MachineState,
    cfg: ExecutionConfig,
    queues: dict[int, deque[int]],
    access_heap: list[tuple[int, int]],
) -> dict[int, int]:
    """現在nodeのCR/cache residentとprefetch候補に必要なnext-use mapを作る。"""

    # Build a small next-use map instead of copying the full future-use map at
    # every instruction.  This keeps large O(n)-gate workloads close to linear.
    out: dict[int, int] = {}
    def add(q: int) -> None:
        out[q] = queues.get(q, deque())[0] if queues.get(q) else NEXT_USE_INF
    for q in node.qubits:
        add(int(q))
    for qs in st.cr_res.values():
        for q in qs:
            add(int(q))
    for slots in st.cache_res.values():
        for q in slots:
            if q is not None:
                add(int(q))
    if cfg.policy.use_staging_prefetch and cfg.policy.max_prefetch_per_step > 0:
        current_qs = set(map(int, node.qubits))
        temp: list[tuple[int, int]] = []
        added = 0
        budget = max(8, cfg.policy.max_prefetch_per_step * 8)
        while access_heap and added < cfg.policy.max_prefetch_per_step and budget > 0:
            budget -= 1
            idx, q = heapq.heappop(access_heap)
            qqueue = queues.get(q)
            if not qqueue or qqueue[0] != idx:
                continue  # stale entry
            temp.append((idx, q))
            if q in current_qs:
                continue
            if st.loc.get(q, ("", 0, 0, 0))[0] != "SAM":
                continue
            out[q] = idx
            added += 1
        for item in temp:
            heapq.heappush(access_heap, item)
    return out

def _choose_cr(node: OperationNode, st: MachineState, cfg: ExecutionConfig, details: dict[str, Any] | None = None) -> int:
    """明示planまたは既存score/tie-breakで実行CRを選ぶ。"""

    if node.preferred_cr is not None and node.meta.get("strict_cr_assignment"):
        return int(node.preferred_cr)
    if (
        cfg.policy.execute_parallel_group_plan
        and cfg.policy.enforce_planned_cr_assignment
        and "planned_cr" in node.meta
    ):
        planned = int(node.meta["planned_cr"])
        if not (0 <= planned < cfg.architecture.cr_count):
            raise RuntimeError(f"Invalid planned CR {planned} for node {node.idx}")
        if len(set(map(int, node.qubits))) > cfg.architecture.cr_slots:
            raise RuntimeError(
                f"Planned group gate exceeds CR capacity: node={node.idx}, "
                f"qubits={node.qubits}, slots={cfg.architecture.cr_slots}"
            )
        if details is not None:
            details.update({
                "candidates": (planned,),
                "scores": (),
                "chosen_cr": planned,
                "chosen_score": ("planned", planned),
            })
        return planned
    score_rows: list[dict[str, Any]] = []
    best_cr = 0
    best_key: tuple[int, ...] | None = None
    unique_qubits = tuple(dict.fromkeys(map(int, node.qubits)))
    for cr in range(cfg.architecture.cr_count):
        residents = st.cr_res.get(cr, set())
        resident_operands = sum(1 for q in unique_qubits if q in residents)
        required_transfers = len(unique_qubits) - resident_operands
        cr_to_cr_transfers = sum(
            1 for q in unique_qubits
            if st.loc[q][0] == "CR" and int(st.loc[q][1]) != cr
        )
        nonlocal_transfers = sum(
            1 for q in unique_qubits
            if st.loc[q][0] in {"SAM", "CACHE"}
            and not cfg.architecture.is_local(int(st.loc[q][1]), cr)
        )
        protected_residents = {q for q in unique_qubits if q in residents}
        evictable = len(residents - protected_residents)
        evictions_needed = max(
            0,
            len(residents) + required_transfers - cfg.architecture.cr_slots,
        )
        infeasible = int(
            len(unique_qubits) > cfg.architecture.cr_slots or evictions_needed > evictable
        )
        total_transfer_ops = required_transfers + evictions_needed
        available_time = max(
            st.resource_until.get(f"cr_compute:{cr}", 0),
            st.resource_until.get(f"cr_port:{cr}", 0),
        )
        preferred_mismatch = int(node.preferred_cr is not None and node.preferred_cr != cr)
        key = (
            infeasible,
            total_transfer_ops,
            cr_to_cr_transfers,
            nonlocal_transfers,
            available_time,
            preferred_mismatch,
            cr,
        )
        score_rows.append({
            "cr": cr,
            "key": key,
            "infeasible": infeasible,
            "resident_operands": resident_operands,
            "required_transfers": required_transfers,
            "estimated_evictions": evictions_needed,
            "cr_to_cr_transfers": cr_to_cr_transfers,
            "nonlocal_transfers": nonlocal_transfers,
            "available_time": available_time,
            "preferred_mismatch": preferred_mismatch,
            "occupancy": len(residents),
        })
        if best_key is None or key < best_key:
            best_key = key
            best_cr = cr
    if details is not None:
        details.update({
            "candidates": tuple(range(cfg.architecture.cr_count)),
            "scores": tuple(score_rows),
            "chosen_cr": best_cr,
            "chosen_score": best_key,
        })
    return best_cr


def _apply_initial_placement(loc: dict[int, tuple[str, int, int, int]], nodes: list[OperationNode], cfg: ExecutionConfig) -> dict[int, tuple[str, int, int, int]]:
    """初期SAM locationへ既存placement policyを適用する。"""

    if cfg.policy.placement_policy == "access_order":
        return loc
    qubits = sorted(loc)
    first_node: dict[int, OperationNode] = {}
    plan_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    plan_first: dict[int, int] = {}
    for n in nodes:
        for q in n.qubits:
            first_node.setdefault(q, n)
            if "planned_cr" in n.meta:
                cr = int(n.meta["planned_cr"])
                plan_counts[int(q)][cr] += 1
                plan_first.setdefault(int(q), cr)
    bank_counts = {b: 0 for b in range(cfg.architecture.banks)}
    new_loc: dict[int, tuple[str, int, int, int]] = {}
    block = max(1, math.ceil(len(qubits) / max(1, cfg.architecture.banks)))
    for pos, q in enumerate(qubits):
        if cfg.policy.placement_policy == "sequential":
            bank = min(cfg.architecture.banks - 1, pos // block)
        elif cfg.policy.placement_policy == "round_robin":
            bank = pos % cfg.architecture.banks
        elif cfg.policy.placement_policy == "first_use_local":
            node = first_node.get(q)
            pref = node.preferred_cr if node is not None else None
            if pref is not None:
                candidates = [
                    b for b in range(cfg.architecture.banks)
                    if cfg.architecture.is_local(b, pref)
                ]
                bank = (
                    min(candidates, key=lambda b: (bank_counts[b], b))
                    if candidates else pos % cfg.architecture.banks
                )
            else:
                # Use the first co-access group as a weak signal: distribute its
                # operands across local banks instead of packing them.
                bank = pos % cfg.architecture.banks
        elif cfg.policy.placement_policy == "plan_affinity":
            counts = plan_counts.get(q, {})
            if counts:
                first = plan_first[q]
                pref = min(counts, key=lambda cr: (-counts[cr], cr != first, cr))
                candidates = [
                    b for b in range(cfg.architecture.banks)
                    if cfg.architecture.is_local(b, pref)
                ]
                bank = (
                    min(candidates, key=lambda b: (bank_counts[b], b))
                    if candidates else pos % cfg.architecture.banks
                )
            else:
                bank = pos % cfg.architecture.banks
        else:
            bank = int(loc[q][1]) % cfg.architecture.banks
        row, col = sam_coord(cfg.architecture.sam_type, bank_counts[bank])
        bank_counts[bank] += 1
        new_loc[q] = ("SAM", bank, row, col)
    return new_loc

def _initial_access_order_layout(lowered: Any, cfg: ExecutionConfig) -> dict[int, tuple[str, int, int, int]]:
    """最初に参照される順でqubitを並べ、bankへround-robin配置する。"""
    seen: set[int] = set()
    order: list[int] = []
    for event in lowered.events:
        for q in event.qubits:
            q = int(q)
            if q not in seen:
                seen.add(q)
                order.append(q)
    order.extend(q for q in range(len(lowered.qubit_names)) if q not in seen)
    per_bank = {bank: 0 for bank in range(cfg.architecture.banks)}
    loc: dict[int, tuple[str, int, int, int]] = {}
    for position, q in enumerate(order):
        bank = position % cfg.architecture.banks
        row, col = sam_coord(cfg.architecture.sam_type, per_bank[bank])
        per_bank[bank] += 1
        loc[q] = ("SAM", bank, row, col)
    return loc


def _build_initial_state(program: Any, cfg: ExecutionConfig) -> tuple[list[OperationNode], MachineState]:
    """Program IRをloweringし、初期SAM配置とscheduler状態を構築する。"""
    lowered = lower_program(program)
    nodes = _parse_nodes(list(lowered.events), cfg)
    loc = _initial_access_order_layout(lowered, cfg)
    loc = _apply_initial_placement(loc, nodes, cfg)
    return nodes, initialize_machine_state(loc, cfg.architecture)

def schedule_program(
    program: Any,
    cfg: ExecutionConfig,
    *,
    decision_trace: list[dict[str, Any]] | None = None,
) -> tuple[list[ScheduledOperation], dict[str, Any]]:
    """Programをdependency順に実行しtraceと従来schemaのmetricsを返す。"""

    nodes, st = _build_initial_state(program, cfg)
    st.scheduling.decision_trace = decision_trace
    staging_reserve_by_node = _rotation_staging_reserve(nodes)
    deps, succ, ready = _build_deps(nodes)
    if cfg.policy.execute_parallel_group_plan:
        _add_parallel_group_dependencies(nodes, deps, succ)
        ready = [idx for idx, predecessors in deps.items() if not predecessors]
    access_queues = _prepare_access_queues(nodes)
    access_heap: list[tuple[int, int]] = []
    for q, queue in access_queues.items():
        if queue:
            heapq.heappush(access_heap, (queue[0], q))
    next_cr_by_node = {n.idx: n.preferred_cr for n in nodes}
    ready_heap: list[tuple[int, int]] = []
    for idx in ready:
        heapq.heappush(ready_heap, (0, idx))
    done: set[int] = set()
    decision_id = 0
    while ready_heap:
        scheduler_priority, idx = heapq.heappop(ready_heap)
        if idx in done:
            continue
        node = nodes[idx]
        # Unique dependency frontier.  The historical heap can contain a
        # duplicate when one predecessor is shared through two operands; this
        # observation deliberately does not normalize or mutate that heap.
        frontier_priority: dict[int, int] = {idx: scheduler_priority}
        for priority, frontier_idx in ready_heap:
            if frontier_idx not in done:
                frontier_priority[frontier_idx] = min(priority, frontier_priority.get(frontier_idx, priority))
        same_priority = sum(1 for priority in frontier_priority.values() if priority == scheduler_priority)
        later_priority = sum(1 for priority in frontier_priority.values() if priority > scheduler_priority)
        loc_before = {int(q): tuple(st.loc[int(q)]) for q in node.qubits}
        occ_before = {
            cr: len(st.cr_res.get(cr, set()))
            for cr in range(cfg.architecture.cr_count)
        }
        waits_before = dict(st.statistics.wait_by_category)
        counters_before = {
            "ld": st.statistics.ld_count,
            "st": st.statistics.st_count,
            "prefetch": st.statistics.cache_prefetch_count,
            "local": st.statistics.local_transfers,
            "nonlocal": st.statistics.nonlocal_transfers,
            "cr_to_cr": st.statistics.cr_to_cr_transfer_count,
            "slot_evict": st.statistics.cr_slot_eviction_events,
        }
        trace_start = len(st.trace)
        st.scheduling.active_node_idx = idx
        st.scheduling.active_decision_id = decision_id
        # Retire this node from per-qubit future-use queues.  Qubit dependencies
        # guarantee that per-qubit uses are processed in program order.
        for q in node.qubits:
            q = int(q)
            queue = access_queues.get(q)
            if queue and queue[0] == idx:
                queue.popleft()
                if queue:
                    heapq.heappush(access_heap, (queue[0], q))
        next_use_after = _next_use_dict_for_node(node, st, cfg, access_queues, access_heap)
        earliest = max((st.q_ready.get(q, 0) for q in node.qubits), default=0)
        cr_details: dict[str, Any] = {}
        if node.op == "CR_ROTATE":
            finish = _schedule_collective_cr_rotation(node, earliest, st, cfg)
        elif node.op == "CR_LOAD":
            finish = _schedule_explicit_cr_load(node, earliest, st, cfg, next_use_after, next_cr_by_node)
        elif node.op == "CR_STORE":
            finish = _schedule_explicit_cr_store(node, earliest, st, cfg)
        elif node.op == "PHASE_BARRIER":
            finish = earliest
            update_qubit_ready(st, list(node.qubits), finish)
        elif not _requires_cr(node.op, cfg.architecture) and len(node.qubits) == 1:
            finish = execute_in_memory_gate(
                node.op,
                node.qubits,
                node.idx,
                node.preferred_cr,
                node.meta,
                earliest,
                st,
                cfg.architecture,
            )
        else:
            cr = _choose_cr(node, st, cfg, cr_details if decision_trace is not None else None)
            demand_banks: set[int] = set()
            # Protect demand from the next few scheduler candidates.
            protected_ready = heapq.nsmallest(8, ready_heap)
            for _, frontier_idx in protected_ready:
                if frontier_idx in done or frontier_idx == idx:
                    continue
                for q in nodes[frontier_idx].qubits:
                    loc = st.loc.get(int(q), ("", 0, 0, 0))
                    if loc[0] == "SAM":
                        demand_banks.add(int(loc[1]))
            finish = _schedule_cr_gate(
                node, cr, earliest, st, cfg, next_use_after, next_cr_by_node,
                demand_banks=demand_banks,
                staging_reserve=staging_reserve_by_node.get(idx, 0),
            )
        if decision_trace is not None:
            emitted = st.trace[trace_start:]
            gate_ops = [op for op in emitted if op.meta.get("node_idx") == idx]
            gate_op = gate_ops[-1] if gate_ops else None
            wait_delta = {
                cat: st.statistics.wait_by_category.get(cat, 0) - waits_before.get(cat, 0)
                for cat in set(st.statistics.wait_by_category) | set(waits_before)
            }
            decision_trace.append({
                "record_type": "decision",
                "decision_id": decision_id,
                "node_idx": idx,
                "op": node.op,
                "qubits": tuple(node.qubits),
                "scheduler_priority": scheduler_priority,
                "dependency_ready": earliest,
                "frontier_width": len(frontier_priority),
                "same_priority_width": same_priority,
                "later_priority_width": later_priority,
                "tie_break_deferred": max(0, same_priority - 1),
                "later_data_ready_deferred": later_priority,
                "preferred_cr": node.preferred_cr,
                "chosen_cr": cr_details.get("chosen_cr", gate_op.cr_id if gate_op else None),
                "candidate_cr_count": len(cr_details.get("candidates", ())),
                "candidate_crs": cr_details.get("candidates", ()),
                "candidate_scores": cr_details.get("scores", ()),
                "gate_start": gate_op.start if gate_op else None,
                "gate_end": gate_op.end if gate_op else finish,
                "start_gap": (gate_op.start - earliest) if gate_op else None,
                "finish": finish,
                "trace_start_index": trace_start,
                "trace_end_index": len(st.trace),
                "emitted_ops": tuple(op.op for op in emitted),
                "emitted_transfer_count": sum(1 for op in emitted if op.op in {"LD", "ST", "SAM_TO_CACHE", "CACHE_TO_CR", "CACHE_EVICT_ST", "CR_TO_CR"}),
                "loc_before": loc_before,
                "loc_after": {int(q): tuple(st.loc[int(q)]) for q in node.qubits},
                "cr_occupancy_before": occ_before,
                "cr_occupancy_after": {
                    cr: len(st.cr_res.get(cr, set()))
                    for cr in range(cfg.architecture.cr_count)
                },
                "cr_residents_after": {
                    cr: tuple(sorted(st.cr_res.get(cr, set())))
                    for cr in range(cfg.architecture.cr_count)
                },
                "cache_residents_after": {
                    bank: tuple(q for q in st.cache_res.get(bank, ()) if q is not None)
                    for bank in range(cfg.architecture.banks)
                },
                "ld_delta": st.statistics.ld_count - counters_before["ld"],
                "st_delta": st.statistics.st_count - counters_before["st"],
                "prefetch_delta": st.statistics.cache_prefetch_count - counters_before["prefetch"],
                "local_transfer_delta": st.statistics.local_transfers - counters_before["local"],
                "nonlocal_transfer_delta": st.statistics.nonlocal_transfers - counters_before["nonlocal"],
                "cr_to_cr_transfer_delta": st.statistics.cr_to_cr_transfer_count - counters_before["cr_to_cr"],
                "cr_slot_eviction_delta": st.statistics.cr_slot_eviction_events - counters_before["slot_evict"],
                "wait_delta": wait_delta,
                "phase": node.meta.get("phase"),
                "source_loop": node.meta.get("source_loop"),
                "loop_iter": node.meta.get("loop_iter"),
                "module_id": node.meta.get("module_id"),
                "declared_parallel_region": node.meta.get("declared_parallel_region"),
                "declared_parallel_group": node.meta.get("declared_parallel_group"),
                "target": node.meta.get("target"),
                "control": node.meta.get("control"),
            })
        st.scheduling.active_node_idx = None
        st.scheduling.active_decision_id = None
        decision_id += 1
        done.add(idx)
        for nxt in succ[idx]:
            deps[nxt].discard(idx)
            if not deps[nxt]:
                e = max((st.q_ready.get(q, 0) for q in nodes[nxt].qubits), default=finish)
                heapq.heappush(ready_heap, (e, nxt))
    if cfg.policy.final_flush:
        _final_flush_all(max((op.end for op in st.trace), default=0), st, cfg)
    total = max((op.end for op in st.trace), default=0)
    def busy(prefix: str) -> int:
        return sum(v for k, v in st.statistics.resource_busy.items() if k.startswith(prefix))
    bank_busy = busy("bank:")
    mem_port_busy = busy("mem_port:")
    cr_port_busy = busy("cr_port:")
    cr_compute_busy = busy("cr_compute:")
    hub_busy = busy("hub:")
    local_route_busy = busy("local_route:")
    nonlocal_route_busy = busy("nonlocal_route:")
    cr_to_cr_route_busy = busy("cr_to_cr_route:")
    cache_slot_busy = busy("cache_slot:")
    denom = max(1, total)
    transfer_ops = st.statistics.ld_count + st.statistics.st_count + st.statistics.cache_prefetch_count + st.statistics.cr_to_cr_transfer_count
    prefetch_hide_rate = st.statistics.prefetch_hidden_hits / st.statistics.cache_prefetch_count if st.statistics.cache_prefetch_count else 0.0
    cache_hit_rate = st.statistics.cache_hit_count / max(1, st.statistics.cache_hit_count + st.statistics.demand_cache_loads)
    final_sam_locations = [
        (int(loc[1]), int(loc[2]), int(loc[3]))
        for loc in st.loc.values()
        if loc[0] == "SAM"
    ]
    final_sam_collision_count = len(final_sam_locations) - len(set(final_sam_locations))
    final_sam_out_of_layout_count = sum(
        1
        for bank, row, col in final_sam_locations
        if (row, col) not in st.sam_cells.get(bank, set())
    )
    metrics = {
        "total_beats": total,
        "gate_count": sum(1 for n in nodes if n.op not in {"CR_ROTATE", "CR_LOAD", "CR_STORE", "PHASE_BARRIER"}),
        "scheduler_node_count": len(nodes),
        "ld_count": st.statistics.ld_count,
        "st_count": st.statistics.st_count,
        "transfer_ops": transfer_ops,
        "cache_prefetch_count": st.statistics.cache_prefetch_count,
        "cache_hit_count": st.statistics.cache_hit_count,
        "prefetch_hidden_hits": st.statistics.prefetch_hidden_hits,
        "prefetch_hide_rate": round(prefetch_hide_rate, 6),
        "cache_hit_rate": round(cache_hit_rate, 6),
        "cache_evict_count": st.statistics.cache_evict_count,
        "cr_to_cr_transfer_count": st.statistics.cr_to_cr_transfer_count,
        "inmemory_count": st.statistics.inmemory_count,
        "route_wait": st.statistics.route_wait,
        "bank_wait": st.statistics.wait_by_category.get("bank", 0),
        "mem_port_wait": st.statistics.wait_by_category.get("mem_port", 0),
        "cr_port_wait": st.statistics.wait_by_category.get("cr_port", 0),
        "cr_compute_wait": st.statistics.wait_by_category.get("cr_compute", 0),
        "hub_wait": st.statistics.wait_by_category.get("hub", 0),
        "local_route_wait": st.statistics.wait_by_category.get("local_route", 0),
        "nonlocal_route_wait": st.statistics.wait_by_category.get("nonlocal_route", 0),
        "cr_to_cr_route_wait": st.statistics.wait_by_category.get("cr_to_cr_route", 0),
        "cache_slot_wait": st.statistics.wait_by_category.get("cache_slot", 0),
        "compute_beats": st.statistics.compute_beats,
        "inmemory_beats": st.statistics.inmemory_beats,
        "transfer_beats": st.statistics.transfer_beats,
        "overhead_beats": max(0, total - st.statistics.compute_beats - st.statistics.inmemory_beats),
        "bank_busy_beats": bank_busy,
        "mem_port_busy_beats": mem_port_busy,
        "cr_port_busy_beats": cr_port_busy,
        "cr_compute_busy_beats": cr_compute_busy,
        "hub_busy_beats": hub_busy,
        "local_route_busy_beats": local_route_busy,
        "nonlocal_route_busy_beats": nonlocal_route_busy,
        "cr_to_cr_route_busy_beats": cr_to_cr_route_busy,
        "cache_slot_busy_beats": cache_slot_busy,
        "bank_utilization": round(
            bank_busy / (denom * max(1, cfg.architecture.banks)), 6
        ),
        "mem_port_utilization": round(
            mem_port_busy / (denom * max(1, cfg.architecture.banks)), 6
        ),
        "cr_port_utilization": round(
            cr_port_busy / (denom * max(1, cfg.architecture.cr_count)), 6
        ),
        "cr_compute_utilization": round(
            cr_compute_busy / (denom * max(1, cfg.architecture.cr_count)), 6
        ),
        "hub_utilization": round(
            hub_busy / (denom * max(1, cfg.architecture.nonlocal_hub_count)), 6
        ),
        "cache_slot_utilization": round(
            cache_slot_busy
            / (
                denom
                * max(
                    1,
                    cfg.architecture.banks * cfg.architecture.cache_slots_per_bank,
                )
            ),
            6,
        ) if cfg.architecture.cache_slots_per_bank else 0.0,
        "local_transfers": st.statistics.local_transfers,
        "nonlocal_transfers": st.statistics.nonlocal_transfers,
        "local_transfer_ratio": round(st.statistics.local_transfers / max(1, st.statistics.local_transfers + st.statistics.nonlocal_transfers), 6),
        "trace_len": len(st.trace),
        "max_cr_occupancy": st.statistics.max_cr_occupancy,
        "cr_overflow_events": st.statistics.cr_overflow_events,
        "cr_slot_eviction_events": st.statistics.cr_slot_eviction_events,
        "planned_load_count": st.statistics.planned_load_count,
        "planned_store_count": st.statistics.planned_store_count,
        "collective_rotation_count": st.statistics.collective_rotation_count,
        "collective_rotation_qubits": st.statistics.collective_rotation_qubits,
        "final_cr_resident_count": sum(len(v) for v in st.cr_res.values()),
        "final_cache_resident_count": sum(1 for slots in st.cache_res.values() for q in slots if q is not None),
        "sam_cell_count": sum(len(cells) for cells in st.sam_cells.values()),
        "initial_distinct_sam_columns": len({col for _, _, col in st.home.values()}),
        "final_sam_cell_collision_count": final_sam_collision_count,
        "final_sam_out_of_layout_count": final_sam_out_of_layout_count,
    }
    if st.magic_state is not None:
        # 明示的に有効化された場合だけ従来schemaへmagic-state指標を追加する。
        metrics.update(st.magic_state.metrics(total))
    return st.trace, metrics
