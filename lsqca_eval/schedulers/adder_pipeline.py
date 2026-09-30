from __future__ import annotations

"""既存Cuccaro five-slot pipelineを共通machine上で実行する専用scheduler。"""

from typing import Any, Iterable, Literal

from lsqca_eval.execution_trace import ScheduledOperation
from lsqca_eval.sam_layout import sam_coord
from lsqca_eval.architecture import (
    _requires_cr,
)
from lsqca_eval.scheduling_policy import ExecutionConfig
from lsqca_eval.machine import (
    execute_cr_gate,
    execute_in_memory_gate,
    initialize_machine_state,
    load_sam_to_cr,
    store_cr_to_sam,
)
from lsqca_eval.runtime_state import MachineState
from lsqca_eval.program_ir import Op, Program
from lsqca_eval.programs.cdkm_adder import emit_maj, emit_uma

LayoutPolicy = Literal["sequential", "parity"]
# Adder専用metricsで転送時間を集計する命令集合。
TRANSFER_OPS = frozenset({"PIPE_LD", "PIPE_ST"})


def _adder_names(program: Program) -> tuple[int, dict[str, int], list[str], list[str], str, str]:
    """固定pipelineが扱うAdder register名とqubit IDを検証して返す。"""

    name_to_id = {name: idx for idx, name in enumerate(program.qubits)}
    a_names = sorted(
        (name for name in program.qubits if name.startswith("a") and name[1:].isdigit()),
        key=lambda name: int(name[1:]),
    )
    b_names = sorted(
        (name for name in program.qubits if name.startswith("b") and name[1:].isdigit()),
        key=lambda name: int(name[1:]),
    )
    if not a_names or len(a_names) != len(b_names):
        raise ValueError("The fixed pipeline requires equal non-empty a/b registers")
    expected = {"cin", "cout", *a_names, *b_names}
    if set(program.qubits) != expected:
        raise ValueError("The fixed pipeline only supports the CDKM adder qubit set")
    return len(a_names), name_to_id, a_names, b_names, "cin", "cout"


def _allocate_layout(
    program: Program,
    *,
    banks: int,
    sam_type: str,
    policy: LayoutPolicy,
) -> dict[int, tuple[str, int, int, int]]:
    """既存のsequentialまたはparity規則で初期SAM配置を構築する。"""

    if banks != 4:
        raise ValueError("The fixed pipeline requires four banks")
    n, name_to_id, a_names, b_names, cin, cout = _adder_names(program)
    used = {bank: 0 for bank in range(banks)}
    layout: dict[int, tuple[str, int, int, int]] = {}

    def allocate(q: int, bank: int) -> None:
        row, col = sam_coord(sam_type, used[bank])
        used[bank] += 1
        layout[q] = ("SAM", bank, row, col)

    if policy == "parity":
        for i in range(n):
            allocate(name_to_id[a_names[i]], 0 if i % 2 else 1)
            allocate(name_to_id[b_names[i]], 2 if i % 2 else 3)
        allocate(name_to_id[cin], 0)
        allocate(name_to_id[cout], 2)
        return layout

    # Exact equivalent of corrected placement_policy="sequential" for the
    # program's qubit-id order.
    qubits = list(range(len(program.qubits)))
    block = max(1, (len(qubits) + banks - 1) // banks)
    for pos, q in enumerate(qubits):
        allocate(q, min(banks - 1, pos // block))
    return layout


def _io_resources(
    bank: int,
    cr: int,
    cfg: ExecutionConfig,
    *,
    direction: str,
    split_io: bool,
) -> tuple[list[str], bool]:
    """Adder LD/STが予約する既存I/O資源とlocal判定を返す。"""

    if direction not in {"load", "store"}:
        raise ValueError(direction)
    port = f"cr_port:{direction}:{cr}" if split_io else f"cr_port:{cr}"
    resources = [f"bank:{bank}", f"mem_port:{bank}", port]
    if cfg.architecture.is_local(bank, cr):
        resources.append(f"local_route:{bank}:{cr}")
        return resources, True
    resources.extend([
        cfg.architecture.hub_for(bank, cr),
        f"nonlocal_route:{bank}:{cr}",
    ])
    return resources, False


def _pipe_load(
    q: int,
    cr: int,
    issue_time: int,
    st: MachineState,
    cfg: ExecutionConfig,
    *,
    split_io: bool,
    reason: str,
) -> int:
    """Adder qubitをSAMからCRへ既存PIPE_LD順で転送する。"""

    where, bank, row, col = st.loc[q]
    if where == "CR":
        if bank != cr:
            raise RuntimeError(f"Qubit is resident in another CR: q={q}, loc={st.loc[q]}")
        return max(issue_time, st.q_ready.get(q, 0))
    if where != "SAM":
        raise RuntimeError(f"Pipeline load requires SAM or CR: q={q}, loc={st.loc[q]}")
    if len(st.cr_res.get(cr, set())) >= cfg.architecture.cr_slots:
        st.statistics.cr_overflow_events += 1
        raise RuntimeError(f"Five-slot pipeline overflow before loading q={q}")
    resources, is_local = _io_resources(bank, cr, cfg, direction="load", split_io=split_io)
    return load_sam_to_cr(
        q,
        cr,
        issue_time,
        st,
        cfg.architecture,
        reason=reason,
        trace_op="PIPE_LD",
        resources=resources,
        is_local=is_local,
    )


def _pipe_store(
    q: int,
    cr: int,
    issue_time: int,
    target_cell: tuple[int, int, int],
    st: MachineState,
    cfg: ExecutionConfig,
    *,
    split_io: bool,
    reason: str,
) -> int:
    """Adder qubitをCRから指定SAM cellへ既存PIPE_ST順で転送する。"""

    where, resident_cr, _, _ = st.loc[q]
    if where != "CR" or resident_cr != cr:
        raise RuntimeError(f"Pipeline store requires CR residency: q={q}, loc={st.loc[q]}")
    bank, row, col = target_cell
    if (row, col) not in st.sam_cells.get(bank, set()):
        raise RuntimeError(f"Pipeline ST target is outside the fixed SAM layout: {target_cell}")
    resources, is_local = _io_resources(bank, cr, cfg, direction="store", split_io=split_io)
    return store_cr_to_sam(
        q,
        issue_time,
        st,
        cfg.architecture,
        bank=bank,
        row=row,
        col=col,
        reason=reason,
        trace_op="PIPE_ST",
        resources=resources,
        is_local=is_local,
    )


def _interval_union(intervals: Iterable[tuple[int, int]]) -> int:
    """既存Adder metrics用に区間集合のunion長を計算する。"""

    ordered = sorted((s, e) for s, e in intervals if e > s)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _interval_intersection(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> int:
    """compute区間とtransfer区間が重なる総時間を計算する。"""

    events: list[tuple[int, int, int]] = []
    for start, end in left:
        if end > start:
            events.extend([(start, 1, 0), (end, -1, 0)])
    for start, end in right:
        if end > start:
            events.extend([(start, 0, 1), (end, 0, -1)])
    events.sort(key=lambda item: item[0])
    active_left = active_right = 0
    previous: int | None = None
    overlap = 0
    for time, delta_left, delta_right in events:
        if previous is not None and active_left > 0 and active_right > 0:
            overlap += time - previous
        active_left += delta_left
        active_right += delta_right
        previous = time
    return overlap


def _metrics(st: MachineState, cfg: ExecutionConfig, gate_count: int, *, split_io: bool) -> dict[str, Any]:
    """Adder専用MachineStateから従来schemaのmetricsを組み立てる。"""

    total = max((op.end for op in st.trace), default=0)
    transfer_intervals = [(op.start, op.end) for op in st.trace if op.op in TRANSFER_OPS]
    compute_intervals = [(op.start, op.end) for op in st.trace if op.op not in TRANSFER_OPS]
    transfer_wall = _interval_union(transfer_intervals)
    compute_wall = _interval_union(compute_intervals)
    overlap = _interval_intersection(compute_intervals, transfer_intervals)
    final_home_mismatches = sum(
        1 for q, cell in st.home.items() if st.loc.get(q) != ("SAM", *cell)
    )
    final_sam = [loc[1:] for loc in st.loc.values() if loc[0] == "SAM"]
    final_outside = sum(
        1 for bank, row, col in final_sam if (row, col) not in st.sam_cells.get(bank, set())
    )
    def busy(prefix: str) -> int:
        return sum(v for k, v in st.statistics.resource_busy.items() if k.startswith(prefix))
    denom = max(1, total)
    lanes = 2 if split_io else 1
    metrics = {
        "total_beats": total,
        "gate_count": gate_count,
        "ld_count": st.statistics.ld_count,
        "st_count": st.statistics.st_count,
        "transfer_ops": st.statistics.ld_count + st.statistics.st_count,
        "cache_prefetch_count": 0,
        "cache_hit_count": 0,
        "prefetch_hidden_hits": 0,
        "prefetch_hide_rate": 0.0,
        "cache_hit_rate": 0.0,
        "cache_evict_count": 0,
        "inmemory_count": st.statistics.inmemory_count,
        "route_wait": st.statistics.route_wait,
        "bank_wait": st.statistics.wait_by_category.get("bank", 0),
        "mem_port_wait": st.statistics.wait_by_category.get("mem_port", 0),
        "cr_port_wait": st.statistics.wait_by_category.get("cr_port", 0),
        "cr_compute_wait": st.statistics.wait_by_category.get("cr_compute", 0),
        "hub_wait": st.statistics.wait_by_category.get("hub", 0),
        "local_route_wait": st.statistics.wait_by_category.get("local_route", 0),
        "nonlocal_route_wait": st.statistics.wait_by_category.get("nonlocal_route", 0),
        "cache_slot_wait": 0,
        "compute_beats": st.statistics.compute_beats,
        "inmemory_beats": st.statistics.inmemory_beats,
        "transfer_beats": st.statistics.transfer_beats,
        "overhead_beats": max(0, total - st.statistics.compute_beats - st.statistics.inmemory_beats),
        "compute_wall_beats": compute_wall,
        "transfer_wall_beats": transfer_wall,
        "compute_transfer_overlap_beats": overlap,
        "exposed_transfer_beats": transfer_wall - overlap,
        "transfer_hiding_rate": round(overlap / transfer_wall, 6) if transfer_wall else 0.0,
        "bank_busy_beats": busy("bank:"),
        "mem_port_busy_beats": busy("mem_port:"),
        "cr_port_busy_beats": busy("cr_port:"),
        "cr_compute_busy_beats": busy("cr_compute:"),
        "bank_utilization": round(busy("bank:") / (denom * cfg.architecture.banks), 6),
        "mem_port_utilization": round(
            busy("mem_port:") / (denom * cfg.architecture.banks), 6
        ),
        "cr_port_utilization": round(
            busy("cr_port:") / (denom * cfg.architecture.cr_count * lanes), 6
        ),
        "cr_compute_utilization": round(
            busy("cr_compute:") / (denom * cfg.architecture.cr_count), 6
        ),
        "local_transfers": st.statistics.local_transfers,
        "nonlocal_transfers": st.statistics.nonlocal_transfers,
        "local_transfer_ratio": round(st.statistics.local_transfers / max(1, st.statistics.ld_count + st.statistics.st_count), 6),
        "trace_len": len(st.trace),
        "max_cr_occupancy": st.statistics.max_cr_occupancy,
        "cr_overflow_events": st.statistics.cr_overflow_events,
        "cr_slot_eviction_events": st.statistics.cr_slot_eviction_events,
        "final_cr_resident_count": sum(len(v) for v in st.cr_res.values()),
        "final_cache_resident_count": 0,
        "sam_cell_count": sum(len(v) for v in st.sam_cells.values()),
        "final_sam_cell_collision_count": len(final_sam) - len(set(final_sam)),
        "final_sam_out_of_layout_count": final_outside,
        "final_home_mismatches": final_home_mismatches,
    }
    if st.magic_state is not None:
        # 共通machineで追跡したlocal MSF指標を専用schedulerにも追加する。
        metrics.update(st.magic_state.metrics(total))
    return metrics


def schedule_adder_pipeline(
    program: Program,
    cfg: ExecutionConfig,
    *,
    split_io: bool,
    layout_policy: LayoutPolicy,
) -> tuple[list[ScheduledOperation], dict[str, Any]]:
    """5-slot Adder pipelineを実行しtraceとmetricsを返す。"""

    if (
        cfg.architecture.cr_count != 1
        or cfg.architecture.cr_slots != 5
        or cfg.architecture.banks != 4
    ):
        raise ValueError("Pipeline requires one CR, five slots, and four banks")
    if cfg.policy.use_staging_prefetch or cfg.architecture.cache_slots_per_bank:
        raise ValueError("Generic staging cache must be disabled")

    n, name_to_id, a_names, b_names, cin_name, cout_name = _adder_names(program)
    initial = _allocate_layout(
        program,
        banks=cfg.architecture.banks,
        sam_type=cfg.architecture.sam_type,
        policy=layout_policy,
    )
    st = initialize_machine_state(initial, cfg.architecture)
    a = [name_to_id[name] for name in a_names]
    b = [name_to_id[name] for name in b_names]
    cin = name_to_id[cin_name]
    cout = name_to_id[cout_name]
    gate_idx = 0

    def qids(op: Op) -> tuple[int, ...]:
        return tuple(name_to_id[name] for name in op.qubits)

    def schedule_op(op: Op, floor: int) -> int:
        nonlocal gate_idx
        op_name = op.gate.upper()
        qubits = qids(op)
        node_idx = gate_idx
        gate_idx += 1
        if not _requires_cr(op_name, cfg.architecture) and len(qubits) == 1:
            return execute_in_memory_gate(
                op_name, qubits, node_idx, None, {}, floor, st, cfg.architecture
            )
        earliest = max([floor, *(st.q_ready.get(q, 0) for q in qubits)])
        _, end = execute_cr_gate(
            op_name, qubits, node_idx, None, {}, 0, earliest, st, cfg.architecture
        )
        return end

    def load_many(items: Iterable[int], issue_time: int, reason: str) -> None:
        for q in items:
            _pipe_load(q, 0, issue_time, st, cfg, split_io=split_io, reason=reason)

    def store_many(items: Iterable[tuple[int, tuple[int, int, int], str]], issue_time: int) -> None:
        for q, target, reason in items:
            _pipe_store(q, 0, issue_time, target, st, cfg, split_io=split_io, reason=reason)

    floor = 0
    for q in (cin, a[0], b[0]):
        floor = _pipe_load(q, 0, floor, st, cfg, split_io=split_io, reason="prologue_load")

    for i in range(n):
        current = (cin if i == 0 else a[i - 1], a[i], b[i])
        if not all(st.loc[q][0] == "CR" for q in current):
            raise RuntimeError(f"Forward operands not resident at i={i}")
        if i < n - 1:
            load_many((a[i + 1], b[i + 1]), floor, "forward_load_next")
        names = (cin_name if i == 0 else a_names[i - 1], a_names[i], b_names[i])
        for op in emit_maj(*names):
            floor = schedule_op(op, floor)
        if i < n - 1:
            previous = cin if i == 0 else a[i - 1]
            store_many((
                (previous, st.home[previous], "forward_store_a_home"),
                (b[i], st.home[b[i + 1]], "forward_store_b_shift"),
            ), floor)

    _pipe_load(cout, 0, floor, st, cfg, split_io=split_io, reason="pivot_load_cout")
    floor = schedule_op(Op("CX", (a_names[-1], cout_name)), floor)
    _pipe_store(cout, 0, floor, st.home[cout], st, cfg, split_io=split_io, reason="pivot_store_cout")

    for i in reversed(range(n)):
        current = (cin if i == 0 else a[i - 1], a[i], b[i])
        if not all(st.loc[q][0] == "CR" for q in current):
            raise RuntimeError(f"Backward operands not resident at i={i}")
        if i > 0:
            next_previous = cin if i == 1 else a[i - 2]
            load_many((next_previous, b[i - 1]), floor, "backward_load_next")
        names = (cin_name if i == 0 else a_names[i - 1], a_names[i], b_names[i])
        for op in emit_uma(*names):
            floor = schedule_op(op, floor)
        if i > 0:
            store_many((
                (a[i], st.home[a[i]], "backward_store_a_home"),
                (b[i], st.home[b[i]], "backward_store_b_home"),
            ), floor)
        else:
            store_many((
                (a[0], st.home[a[0]], "epilogue_store_a_home"),
                (b[0], st.home[b[0]], "epilogue_store_b_home"),
                (cin, st.home[cin], "epilogue_store_cin_home"),
            ), floor)

    metrics = _metrics(st, cfg, gate_idx, split_io=split_io)
    required_zero = (
        "cr_overflow_events", "final_cr_resident_count", "final_cache_resident_count",
        "final_sam_cell_collision_count", "final_sam_out_of_layout_count", "final_home_mismatches",
    )
    for key in required_zero:
        if metrics[key] != 0:
            raise RuntimeError(f"{key}={metrics[key]}")
    return st.trace, metrics
