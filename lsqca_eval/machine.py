from __future__ import annotations

"""LSQCA machine state、資源占有、転送、gate実行primitive。"""

from typing import Any

from .architecture import (
    ArchitectureConfig,
    _gate_latency,
    _magic_need,
    _route_resources,
    _sam_seek_latency,
)
from .execution_trace import ScheduledOperation
from .magic_state import MagicStateRuntime
from . import runtime_state as _runtime


def apply_machine_layout(
    st: _runtime.MachineState,
    loc: dict[int, tuple[str, int, int, int]],
    architecture: ArchitectureConfig,
) -> None:
    """選択済み初期layoutをmachine stateへ適用する。"""

    copied = {int(q): tuple(value) for q, value in loc.items()}
    home = {
        q: (int(value[1]), int(value[2]), int(value[3]))
        for q, value in copied.items()
    }
    sam_cells = {bank: set() for bank in range(architecture.banks)}
    for _, bank, row, col in copied.values():
        cell = (int(row), int(col))
        if cell in sam_cells[int(bank)]:
            raise RuntimeError(f"Initial SAM cell collision at bank={bank}, cell={cell}")
        sam_cells[int(bank)].add(cell)
    st.loc = copied
    st.home = home
    st.sam_cells = sam_cells
    st.cr_res = {cr: set() for cr in range(architecture.cr_count)}
    st.cache_res = {
        bank: [None for _ in range(architecture.cache_slots_per_bank)]
        for bank in range(architecture.banks)
    }
    st.q_ready = {q: 0 for q in copied}
    st.bank_head = {bank: 0 for bank in range(architecture.banks)}


def initialize_machine_state(
    loc: dict[int, tuple[str, int, int, int]],
    architecture: ArchitectureConfig,
) -> _runtime.MachineState:
    """選択済み初期layoutから空のmachine stateを構築する。"""

    # Architectureで明示された場合だけ各CRのlocal MSFを有効化する。
    magic_state = (
        MagicStateRuntime(architecture.magic_state, architecture.cr_count)
        if architecture.magic_state is not None
        else None
    )
    st = _runtime.MachineState(
        loc={},
        home={},
        sam_cells={},
        cr_res={},
        cache_res={},
        magic_state=magic_state,
    )
    apply_machine_layout(st, loc, architecture)
    return st


def update_qubit_ready(
    st: _runtime.MachineState,
    qubits: tuple[int, ...] | list[int],
    ready: int,
) -> None:
    """指定qubitのready時刻を既存のmax規則で更新する。"""

    for q in qubits:
        qid = int(q)
        st.q_ready[qid] = max(st.q_ready.get(qid, 0), ready)


def append_trace(st: _runtime.MachineState, operation: ScheduledOperation) -> None:
    """選択済み実行記録をmachine traceへ追加する。"""

    st.trace.append(operation)


def _res_category(resource: str) -> str:
    """資源名を既存metricsのwait categoryへ分類する。"""

    if resource.startswith("bank:"):
        return "bank"
    if resource.startswith("mem_port:"):
        return "mem_port"
    if resource.startswith("cr_port:"):
        return "cr_port"
    if resource.startswith("cr_compute:"):
        return "cr_compute"
    if resource.startswith("cache_slot:"):
        return "cache_slot"
    if resource.startswith("hub:"):
        return "hub"
    if resource.startswith("local_route:"):
        return "local_route"
    if resource.startswith("nonlocal_route:"):
        return "nonlocal_route"
    if resource.startswith("cr_to_cr_route:"):
        return "cr_to_cr_route"
    return "other"


def wait_for_resources(
    st: _runtime.MachineState,
    resources: list[str],
    earliest: int,
) -> tuple[int, int]:
    """指定資源が全て空く時刻と待ち時間を既存順で計算する。"""

    ready = max((st.resource_until.get(r, 0) for r in resources), default=0)
    wait = max(0, ready - earliest)
    if wait:
        blockers = [r for r in resources if st.resource_until.get(r, 0) == ready]
        cat = _res_category(blockers[0]) if blockers else "other"
        st.statistics.wait_by_category[cat] = st.statistics.wait_by_category.get(cat, 0) + wait
        if st.scheduling.decision_trace is not None:
            st.scheduling.decision_trace.append({
                "record_type": "resource_wait",
                "decision_id": st.scheduling.active_decision_id,
                "node_idx": st.scheduling.active_node_idx,
                "earliest": earliest,
                "resource_ready": ready,
                "wait": wait,
                "category": cat,
                "resources": tuple(resources),
                "blockers": tuple(blockers),
            })
    return ready if ready > earliest else earliest, wait


def reserve_resources(
    st: _runtime.MachineState,
    resources: list[str],
    end: int,
    start: int | None = None,
) -> None:
    """指定資源をendまで占有しbusy beatを加算する。"""

    if start is None:
        start = min((st.resource_until.get(r, end) for r in resources), default=end)
    dur = max(0, end - start)
    for r in resources:
        st.resource_until[r] = end
        st.statistics.resource_busy[r] = st.statistics.resource_busy.get(r, 0) + dur


def transfer_metadata(
    *,
    source: tuple[str, int, int, int],
    target: tuple[str, int, int, int],
    resources: list[str],
    architecture: ArchitectureConfig,
    is_local: bool | None = None,
) -> dict[str, Any]:
    """転送endpointと予約資源を従来schemaのtrace metadataへ変換する。"""

    meta: dict[str, Any] = {
        "sam_type": architecture.sam_type,
        "source_kind": source[0],
        "source_index": int(source[1]),
        "source_row_or_slot": int(source[2]),
        "source_col": int(source[3]),
        "target_kind": target[0],
        "target_index": int(target[1]),
        "target_row_or_slot": int(target[2]),
        "target_col": int(target[3]),
        "route_resources": tuple(resources),
    }
    if source[0] == "SAM":
        meta.update({"source_bank": source[1], "source_row": source[2], "source_sam_col": source[3]})
    if target[0] == "SAM":
        meta.update({"target_bank": target[1], "target_row": target[2], "target_sam_col": target[3]})
    if source[0] == "CR":
        meta["source_cr"] = source[1]
    if target[0] == "CR":
        meta["target_cr"] = target[1]
    if is_local is not None:
        meta["is_local_transfer"] = bool(is_local)
    return meta


def store_cache_to_sam(
    q: int,
    t: int,
    st: _runtime.MachineState,
    architecture: ArchitectureConfig,
    *,
    row: int,
    col: int,
    trace_op: str = "CACHE_FINAL_ST",
    reason: str = "final_cache_flush",
    cache_eviction: bool = False,
) -> int:
    """cache residentを選択済みSAM cellへstoreする。"""

    where, bank, slot, _ = st.loc[q]
    if where != "CACHE":
        return max(t, st.q_ready.get(q, 0))
    source = tuple(st.loc[q])
    target = ("SAM", bank, row, col)
    if (row, col) not in st.sam_cells.get(bank, set()):
        raise RuntimeError(f"Store target is outside the fixed SAM layout: {(bank, row, col)}")
    for other, location in st.loc.items():
        if other != q and location == target:
            raise RuntimeError(f"Store target is occupied: q={q}, target={target}, owner={other}")
    resources = [f"bank:{bank}", f"mem_port:{bank}", f"cache_slot:{bank}:{slot}"]
    t0, wait = wait_for_resources(st, resources, max(t, st.q_ready.get(q, 0)))
    st.statistics.route_wait += wait
    dur = _sam_seek_latency(architecture, st, bank, row, col, op="ST")
    end = t0 + dur
    reserve_resources(st, resources, end, t0)
    st.statistics.transfer_beats += dur
    st.bank_head[bank] = row
    st.cache_res[bank][slot] = None
    st.loc[q] = ("SAM", bank, row, col)
    st.q_ready[q] = end
    st.statistics.st_count += 1
    if cache_eviction:
        st.statistics.cache_evict_count += 1
    st.trace.append(ScheduledOperation(
        trace_op, (q,), t0, end, bank_id=bank, reason=reason,
        meta=transfer_metadata(
            source=source,
            target=target,
            resources=resources,
            architecture=architecture,
        ),
    ))
    return end


def load_sam_to_cache(
    q: int,
    t: int,
    st: _runtime.MachineState,
    architecture: ArchitectureConfig,
    *,
    slot: int,
    reason: str,
) -> int:
    """qubitを選択済みcache slotへSAMからloadする。"""

    where, bank, row, col = st.loc[q]
    if where == "CACHE":
        return max(t, st.q_ready.get(q, 0))
    if where != "SAM":
        return max(t, st.q_ready.get(q, 0))
    source = tuple(st.loc[q])
    if not 0 <= slot < len(st.cache_res.get(bank, [])):
        raise RuntimeError(f"Invalid cache slot: bank={bank}, slot={slot}")
    if st.cache_res[bank][slot] is not None:
        raise RuntimeError(f"Cache slot is occupied: bank={bank}, slot={slot}")
    target = ("CACHE", bank, slot, 0)
    resources = [f"bank:{bank}", f"mem_port:{bank}", f"cache_slot:{bank}:{slot}"]
    t0, wait = wait_for_resources(st, resources, max(t, st.q_ready.get(q, 0)))
    st.statistics.route_wait += wait
    dur = _sam_seek_latency(architecture, st, bank, row, col, op="LD")
    end = t0 + dur
    reserve_resources(st, resources, end, t0)
    st.statistics.transfer_beats += dur
    st.bank_head[bank] = row
    st.loc[q] = ("CACHE", bank, slot, 0)
    st.cache_res[bank][slot] = q
    st.statistics.cache_origin[q] = reason
    st.q_ready[q] = end
    if reason == "prefetch":
        st.statistics.cache_prefetch_count += 1
    else:
        st.statistics.ld_count += 1
        st.statistics.demand_cache_loads += 1
    st.trace.append(ScheduledOperation(
        "PREFETCH_CACHE" if reason == "prefetch" else "SAM_TO_CACHE",
        (q,), t0, end, bank_id=bank, reason=reason,
        meta=transfer_metadata(
            source=source,
            target=target,
            resources=resources,
            architecture=architecture,
        ),
    ))
    return end


def transfer_cache_to_cr(
    q: int,
    cr: int,
    t: int,
    st: _runtime.MachineState,
    architecture: ArchitectureConfig,
    *,
    trace_op: str = "CACHE_TO_CR",
    reason: str = "cache_to_cr",
    count_as_load: bool = True,
    include_transfer_metadata: bool = True,
) -> int:
    """qubitをcacheから選択済みCRへ転送する。"""

    where, bank, slot, _ = st.loc[q]
    if where != "CACHE":
        raise RuntimeError(f"CACHE->CR requires CACHE location: q={q}, loc={st.loc[q]}")
    source = tuple(st.loc[q])
    target = ("CR", cr, 0, 0)
    resources, is_local = _route_resources(
        bank,
        cr,
        architecture,
        include_bank=False,
        include_cr_port=True,
    )
    resources.append(f"cache_slot:{bank}:{slot}")
    t0, wait = wait_for_resources(st, resources, max(t, st.q_ready.get(q, 0)))
    st.statistics.route_wait += wait
    end = t0 + 1
    reserve_resources(st, resources, end, t0)
    st.statistics.transfer_beats += 1
    origin = st.statistics.cache_origin.pop(q, "")
    if count_as_load and origin == "prefetch":
        st.statistics.prefetch_hidden_hits += 1
    st.cache_res[bank][slot] = None
    st.loc[q] = ("CR", cr, 0, 0)
    st.cr_res.setdefault(cr, set()).add(q)
    occ = len(st.cr_res.setdefault(cr, set()))
    st.statistics.max_cr_occupancy = max(st.statistics.max_cr_occupancy, occ)
    if occ > architecture.cr_slots:
        st.statistics.cr_overflow_events += 1
    st.q_ready[q] = end
    if count_as_load:
        st.statistics.ld_count += 1
        st.statistics.cache_hit_count += 1
    if is_local:
        st.statistics.local_transfers += 1
    else:
        st.statistics.nonlocal_transfers += 1
    trace_meta = (
        transfer_metadata(
            source=source,
            target=target,
            resources=resources,
            architecture=architecture,
            is_local=is_local,
        )
        if include_transfer_metadata else {}
    )
    st.trace.append(ScheduledOperation(
        trace_op, (q,), t0, end, cr_id=cr, bank_id=bank, reason=reason,
        meta=trace_meta,
    ))
    return end


def transfer_cr_to_cache(
    q: int,
    bank: int,
    slot: int,
    t: int,
    st: _runtime.MachineState,
    architecture: ArchitectureConfig,
    *,
    trace_op: str = "CR_TO_CACHE",
    reason: str = "cr_to_cache",
    include_transfer_metadata: bool = True,
) -> int:
    """qubitを現在CRから選択済みcache slotへ転送する。"""

    where, source_cr, _, _ = st.loc[q]
    if where != "CR":
        raise RuntimeError(f"CR->CACHE requires CR location: q={q}, loc={st.loc[q]}")
    if not 0 <= slot < len(st.cache_res.get(bank, [])):
        raise RuntimeError(f"Invalid cache slot: bank={bank}, slot={slot}")
    if st.cache_res[bank][slot] is not None:
        raise RuntimeError(f"Cache slot is occupied: bank={bank}, slot={slot}")
    source = tuple(st.loc[q])
    target = ("CACHE", bank, slot, 0)
    resources, is_local = _route_resources(
        bank,
        source_cr,
        architecture,
        include_bank=False,
        include_cr_port=True,
    )
    resources.append(f"cache_slot:{bank}:{slot}")
    start, wait = wait_for_resources(st, resources, max(t, st.q_ready.get(q, 0)))
    st.statistics.route_wait += wait
    end = start + 1
    reserve_resources(st, resources, end, start)
    st.statistics.transfer_beats += 1
    st.cr_res[source_cr].discard(q)
    st.cache_res[bank][slot] = q
    st.loc[q] = target
    st.q_ready[q] = end
    if is_local:
        st.statistics.local_transfers += 1
    else:
        st.statistics.nonlocal_transfers += 1
    trace_meta = (
        transfer_metadata(
            source=source,
            target=target,
            resources=resources,
            architecture=architecture,
            is_local=is_local,
        )
        if include_transfer_metadata else {}
    )
    st.trace.append(ScheduledOperation(
        trace_op,
        (q,),
        start,
        end,
        cr_id=source_cr,
        bank_id=bank,
        reason=reason,
        meta=trace_meta,
    ))
    return end


def load_sam_to_cr(
    q: int,
    cr: int,
    t: int,
    st: _runtime.MachineState,
    architecture: ArchitectureConfig,
    *,
    reason: str = "direct_ld",
    trace_op: str = "LD",
    resources: list[str] | None = None,
    is_local: bool | None = None,
) -> int:
    """qubitをSAMから選択済みCRへloadする。"""

    where, bank, row, col = st.loc[q]
    if where != "SAM":
        return max(t, st.q_ready.get(q, 0))
    source = tuple(st.loc[q])
    target = ("CR", cr, 0, 0)
    if resources is None:
        resources, route_local = _route_resources(
            bank,
            cr,
            architecture,
            include_bank=True,
            include_cr_port=True,
        )
        is_local = route_local
    elif is_local is None:
        is_local = architecture.is_local(bank, cr)
    t0, wait = wait_for_resources(st, resources, max(t, st.q_ready.get(q, 0)))
    st.statistics.route_wait += wait
    dur = _sam_seek_latency(architecture, st, bank, row, col, op="LD")
    end = t0 + dur
    reserve_resources(st, resources, end, t0)
    st.statistics.transfer_beats += dur
    st.bank_head[bank] = row
    st.loc[q] = ("CR", cr, 0, 0)
    st.cr_res.setdefault(cr, set()).add(q)
    occ = len(st.cr_res.setdefault(cr, set()))
    st.statistics.max_cr_occupancy = max(st.statistics.max_cr_occupancy, occ)
    if occ > architecture.cr_slots:
        st.statistics.cr_overflow_events += 1
    st.q_ready[q] = end
    st.statistics.ld_count += 1
    if is_local:
        st.statistics.local_transfers += 1
    else:
        st.statistics.nonlocal_transfers += 1
    st.trace.append(ScheduledOperation(
        trace_op, (q,), t0, end, cr_id=cr, bank_id=bank, reason=reason,
        meta=transfer_metadata(
            source=source,
            target=target,
            resources=resources,
            architecture=architecture,
            is_local=is_local,
        ),
    ))
    return end


def transfer_cr_to_cr(
    q: int,
    target_cr: int,
    t: int,
    st: _runtime.MachineState,
    architecture: ArchitectureConfig,
) -> int:
    """qubitを現在CRから選択済みCRへ直接転送する。"""

    where, source_cr, _, _ = st.loc[q]
    if where != "CR":
        raise RuntimeError(f"CR->CR requires CR location: q={q}, loc={st.loc[q]}")
    if source_cr == target_cr:
        return max(t, st.q_ready.get(q, 0))
    source = tuple(st.loc[q])
    target = ("CR", target_cr, 0, 0)
    resources = [
        f"cr_port:{source_cr}",
        f"cr_port:{target_cr}",
        "hub:global",
        f"cr_to_cr_route:{source_cr}:{target_cr}",
    ]
    t0, wait = wait_for_resources(st, resources, max(t, st.q_ready.get(q, 0)))
    st.statistics.route_wait += wait
    dur = max(1, int(architecture.cr_to_cr_latency))
    end = t0 + dur
    reserve_resources(st, resources, end, t0)
    st.statistics.transfer_beats += dur
    st.cr_res.setdefault(source_cr, set()).discard(q)
    st.cr_res.setdefault(target_cr, set()).add(q)
    st.loc[q] = target
    st.q_ready[q] = end
    st.statistics.cr_to_cr_transfer_count += 1
    st.statistics.nonlocal_transfers += 1
    occ = len(st.cr_res.setdefault(target_cr, set()))
    st.statistics.max_cr_occupancy = max(st.statistics.max_cr_occupancy, occ)
    if occ > architecture.cr_slots:
        st.statistics.cr_overflow_events += 1
    st.trace.append(ScheduledOperation(
        "CR_TO_CR", (q,), t0, end, cr_id=target_cr, reason="direct_cr_migration",
        meta=transfer_metadata(
            source=source,
            target=target,
            resources=resources,
            architecture=architecture,
            is_local=False,
        ),
    ))
    return end


def store_cr_to_sam(
    q: int,
    t: int,
    st: _runtime.MachineState,
    architecture: ArchitectureConfig,
    *,
    bank: int,
    row: int,
    col: int,
    reason: str = "store",
    trace_op: str = "ST",
    resources: list[str] | None = None,
    is_local: bool | None = None,
) -> int:
    """CR residentを選択済みSAM cellへstoreする。"""

    where, cr, _, _ = st.loc[q]
    if where == "SAM":
        return max(t, st.q_ready.get(q, 0))
    if where == "CACHE":
        # Cache-resident qubit can be left there unless forced.  For CR eviction,
        # this branch is not normally used.
        return max(t, st.q_ready.get(q, 0))
    if where != "CR":
        return max(t, st.q_ready.get(q, 0))
    source = tuple(st.loc[q])
    target = ("SAM", bank, row, col)
    if (row, col) not in st.sam_cells.get(bank, set()):
        raise RuntimeError(f"Store target is outside the fixed SAM layout: {(bank, row, col)}")
    for other, location in st.loc.items():
        if other != q and location == target:
            raise RuntimeError(f"Store target is occupied: q={q}, target={target}, owner={other}")
    if resources is None:
        resources, route_local = _route_resources(
            bank,
            cr,
            architecture,
            include_bank=True,
            include_cr_port=True,
        )
        is_local = route_local
    elif is_local is None:
        is_local = architecture.is_local(bank, cr)
    t0, wait = wait_for_resources(st, resources, max(t, st.q_ready.get(q, 0)))
    st.statistics.route_wait += wait
    dur = _sam_seek_latency(architecture, st, bank, row, col, op="ST")
    end = t0 + dur
    reserve_resources(st, resources, end, t0)
    st.statistics.transfer_beats += dur
    st.bank_head[bank] = row
    st.loc[q] = ("SAM", bank, row, col)
    st.cr_res.setdefault(cr, set()).discard(q)
    st.q_ready[q] = end
    st.statistics.st_count += 1
    if is_local:
        st.statistics.local_transfers += 1
    else:
        st.statistics.nonlocal_transfers += 1
    st.trace.append(ScheduledOperation(
        trace_op, (q,), t0, end, cr_id=cr, bank_id=bank, reason=reason,
        meta=transfer_metadata(
            source=source,
            target=target,
            resources=resources,
            architecture=architecture,
            is_local=is_local,
        ),
    ))
    return end


def execute_in_memory_gate(
    op: str,
    qubits: tuple[int, ...],
    node_idx: int,
    preferred_cr: int | None,
    meta: dict[str, Any],
    t_floor: int,
    st: _runtime.MachineState,
    architecture: ArchitectureConfig,
) -> int:
    """単一qubit gateを現在のSAM/cache/CR location上で実行する。"""

    # In-memory single-qubit gates use SAM/cache bank resources but not CR.
    # CACHE-resident qubits are treated as bank-local and cheap.
    q = int(qubits[0])
    loc = st.loc[q]
    t = max(t_floor, st.q_ready.get(q, 0))
    if loc[0] == "CR":
        cr = loc[1]
        resources = [f"cr_compute:{cr}"]
        t0, wait = wait_for_resources(st, resources, t)
        dur = _gate_latency(op, architecture.rotation_epsilon)
        end = t0 + dur
        reserve_resources(st, resources, end, t0)
        st.statistics.compute_beats += dur
        st.q_ready[q] = end
        st.trace.append(ScheduledOperation(
            op, qubits, t0, end, cr_id=cr, reason="single_in_cr",
            meta={**meta, "node_idx": node_idx, "preferred_cr": preferred_cr},
        ))
        return end
    if loc[0] == "SAM":
        _, bank, row, col = loc
        resources = [f"bank:{bank}"]
        t0, wait = wait_for_resources(st, resources, t)
        st.statistics.route_wait += wait
        seek = max(0, _sam_seek_latency(architecture, st, bank, row, col, op="ST") - 1)
        dur = seek + _gate_latency(op, architecture.rotation_epsilon)
        end = t0 + dur
        reserve_resources(st, resources, end, t0)
        st.bank_head[bank] = row
        st.statistics.inmemory_beats += dur
        st.q_ready[q] = end
        st.statistics.inmemory_count += 1
        st.trace.append(ScheduledOperation(
            op, qubits, t0, end, bank_id=bank, reason="inmemory_single",
            meta={**meta, "node_idx": node_idx, "preferred_cr": preferred_cr},
        ))
        return end
    if loc[0] == "CACHE":
        _, bank, slot, _ = loc
        resources = [f"cache_slot:{bank}:{slot}"]
        t0, wait = wait_for_resources(st, resources, t)
        dur = max(1, _gate_latency(op, architecture.rotation_epsilon))
        end = t0 + dur
        reserve_resources(st, resources, end, t0)
        st.statistics.inmemory_beats += dur
        st.q_ready[q] = end
        st.statistics.inmemory_count += 1
        st.trace.append(ScheduledOperation(
            op, qubits, t0, end, bank_id=bank, reason="cache_single",
            meta={**meta, "node_idx": node_idx, "preferred_cr": preferred_cr},
        ))
        return end
    raise RuntimeError(f"Unsupported location for in-memory gate: q={q}, loc={loc}")


def execute_cr_gate(
    op: str,
    qubits: tuple[int, ...],
    node_idx: int,
    preferred_cr: int | None,
    meta: dict[str, Any],
    cr: int,
    t: int,
    st: _runtime.MachineState,
    architecture: ArchitectureConfig,
    *,
    include_preferred_cr: bool = True,
    metadata_suffix: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """operand配置後のCR compute予約とtrace追加だけを実行する。"""

    resources = [f"cr_compute:{cr}"]
    t0, wait = wait_for_resources(st, resources, t)
    dur = _gate_latency(op, architecture.rotation_epsilon)
    # CR選択後にlocal bufferを消費し、不足時は選択済みCRを占有したまま待つ。
    reservation = None
    if st.magic_state is not None:
        required = _magic_need(op, architecture.rotation_epsilon)
        if required:
            reservation = st.magic_state.consume_for_gate(
                cr,
                t0,
                required,
                injection_spacing=3,
            )
    magic_wait = reservation.wait_beats if reservation is not None else 0
    end = t0 + dur + magic_wait
    reserve_resources(st, resources, end, t0)
    st.statistics.compute_beats += dur
    if magic_wait:
        st.statistics.wait_by_category["magic_state"] = (
            st.statistics.wait_by_category.get("magic_state", 0) + magic_wait
        )
        if st.scheduling.decision_trace is not None:
            st.scheduling.decision_trace.append({
                "record_type": "magic_state_wait",
                "decision_id": st.scheduling.active_decision_id,
                "node_idx": st.scheduling.active_node_idx,
                "cr_id": cr,
                "earliest": t0,
                "wait": magic_wait,
                "required": reservation.state_count,
            })
    for q in qubits:
        st.q_ready[q] = end
    trace_meta = {**meta, "node_idx": node_idx}
    if include_preferred_cr:
        trace_meta["preferred_cr"] = preferred_cr
    if metadata_suffix:
        trace_meta.update(metadata_suffix)
    if (
        reservation is not None
        and st.magic_state is not None
        and st.magic_state.config.mode == "finite"
    ):
        trace_meta.update({
            "magic_state_required": reservation.state_count,
            "magic_state_wait_beats": reservation.wait_beats,
            "magic_state_first_consumption_time": reservation.first_consumption_time,
            "magic_state_last_consumption_time": reservation.last_consumption_time,
            "magic_state_buffer_before": reservation.buffer_before,
            "magic_state_buffer_after": reservation.buffer_after,
            "magic_state_buffer_empty_at_request": reservation.buffer_empty_at_request,
        })
    st.trace.append(ScheduledOperation(
        op, qubits, t0, end, cr_id=cr, reason="cr_gate",
        meta=trace_meta,
    ))
    return t0, end
