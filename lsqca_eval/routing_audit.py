from __future__ import annotations

"""Post-hoc path occupancy audit for LSQCA scheduler ScheduledOperation traces.

The scheduler reserves coarse resources online.  This module never changes a
schedule.  It checks those declared reservations and, separately, embeds bank
and CR ports in a deterministic candidate grid to expose physical-edge sharing
that the coarse model may have omitted.  Candidate-grid conflicts are therefore
diagnostics under an explicit topology assumption, not claims about a finalized
hardware floorplan.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .execution_trace import ScheduledOperation


TRANSFER_OPS = frozenset({
    "LD", "ST", "SAM_TO_CACHE", "PREFETCH_CACHE", "CACHE_TO_CR",
    "CR_TO_CACHE", "CACHE_EVICT_ST", "CACHE_FINAL_ST", "CR_TO_CR",
    "CR_ROTATE_OUT", "CR_ROTATE_IN", "PIPE_LD", "PIPE_ST",
})


def _edge(a: str, b: str) -> str:
    left, right = sorted((a, b))
    return f"{left}<->{right}"


def _grid_edge(prefix: str, x0: int, y0: int, x1: int, y1: int) -> str:
    return _edge(f"{prefix}:{x0}:{y0}", f"{prefix}:{x1}:{y1}")


def _sam_internal_edges(kind: str, bank: int, row: int, col: int) -> list[str]:
    """Line-SAMだけにcandidate cell-to-port経路を構築する。"""
    if kind == "line-sam":
        out = [_edge(f"sam_cell:{bank}:{row}:{col}", f"line_row:{bank}:{row}")]
        if row == 0:
            out.append(_edge(f"line_row:{bank}:0", f"bank_port:{bank}"))
        else:
            for current in range(row, 0, -1):
                out.append(_edge(f"line_row:{bank}:{current}", f"line_row:{bank}:{current - 1}"))
            out.append(_edge(f"line_row:{bank}:0", f"bank_port:{bank}"))
        return out
    if kind == "point-sam":
        # Point-SAM内部は正規modelが宣言したresourceだけを監査する。
        return []
    raise ValueError(f"Unknown SAM type: {kind}")


def _interconnect_edges(bank: int, cr: int) -> list[str]:
    """Candidate Manhattan embedding through a shared vertical trunk.

    Bank b is placed at (0, 2b), CR c at (4, 2c), and the shared trunk is x=2.
    This compact embedding is intentionally simple and fully reproducible.
    """
    bank_y, cr_y = 2 * bank, 2 * cr
    out = [
        _grid_edge("fabric", 0, bank_y, 1, bank_y),
        _grid_edge("fabric", 1, bank_y, 2, bank_y),
    ]
    y = bank_y
    step = 1 if cr_y >= bank_y else -1
    while y != cr_y:
        out.append(_grid_edge("fabric", 2, y, 2, y + step))
        y += step
    out.extend([
        _grid_edge("fabric", 2, cr_y, 3, cr_y),
        _grid_edge("fabric", 3, cr_y, 4, cr_y),
    ])
    return out


@dataclass(frozen=True)
class PathTransfer:
    transfer_id: int
    trace_index: int
    op: str
    qubits: tuple[int, ...]
    start: int
    end: int
    bank_id: int | None
    cr_id: int | None
    source_kind: str | None
    target_kind: str | None
    source_row: int | None
    source_col: int | None
    target_row: int | None
    target_col: int | None
    sam_type: str
    exact_sam_endpoint: bool
    declared_resources: tuple[str, ...]
    candidate_edges: tuple[str, ...]


@dataclass(frozen=True)
class PathConflict:
    model: str
    edge: str
    first_transfer_id: int
    second_transfer_id: int
    overlap_start: int
    overlap_end: int
    overlap_beats: int


def _as_int(value: Any) -> int | None:
    return None if value is None else int(value)


def reconstruct_transfer_paths(trace: Iterable[ScheduledOperation]) -> list[PathTransfer]:
    transfers: list[PathTransfer] = []
    for trace_index, op in enumerate(trace):
        if op.op not in TRANSFER_OPS:
            continue
        meta = op.meta or {}
        source_kind = meta.get("source_kind")
        target_kind = meta.get("target_kind")
        bank = _as_int(op.bank_id)
        cr = _as_int(op.cr_id)
        if bank is None:
            bank = _as_int(meta.get("source_bank", meta.get("target_bank")))
        if cr is None:
            cr = _as_int(meta.get("source_cr", meta.get("target_cr")))

        source_row = _as_int(meta.get("source_row"))
        source_col = _as_int(meta.get("source_sam_col"))
        target_row = _as_int(meta.get("target_row"))
        target_col = _as_int(meta.get("target_sam_col"))
        sam_type = str(meta.get("sam_type", "line-sam"))
        candidate: list[str] = []
        exact = False
        if source_kind == "SAM" and bank is not None and source_row is not None and source_col is not None:
            candidate.extend(_sam_internal_edges(sam_type, bank, source_row, source_col))
            exact = True
        if target_kind == "SAM" and bank is not None and target_row is not None and target_col is not None:
            candidate.extend(_sam_internal_edges(sam_type, bank, target_row, target_col))
            exact = True
        if bank is not None and cr is not None and ({source_kind, target_kind} & {"CR"}):
            candidate.extend(_interconnect_edges(bank, cr))

        # Preserve order for path readability but remove duplicate edges.
        candidate = list(dict.fromkeys(candidate))
        declared = tuple(str(r) for r in meta.get("route_resources", ()))
        transfers.append(PathTransfer(
            transfer_id=len(transfers), trace_index=trace_index, op=op.op,
            qubits=tuple(map(int, op.qubits)), start=int(op.start), end=int(op.end),
            bank_id=bank, cr_id=cr, source_kind=source_kind, target_kind=target_kind,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            sam_type=sam_type, exact_sam_endpoint=exact, declared_resources=declared,
            candidate_edges=tuple(candidate),
        ))
    return transfers


def _intervals_by_edge(transfers: list[PathTransfer], model: str) -> dict[str, list[tuple[int, int, int]]]:
    out: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for transfer in transfers:
        edges = (
            tuple(f"resource:{resource}" for resource in transfer.declared_resources)
            if model == "declared_resource" else transfer.candidate_edges
        )
        for edge in dict.fromkeys(edges):
            out[edge].append((transfer.start, transfer.end, transfer.transfer_id))
    return out


def find_path_conflicts(transfers: list[PathTransfer], model: str) -> list[PathConflict]:
    if model not in {"declared_resource", "candidate_grid"}:
        raise ValueError(model)
    conflicts: list[PathConflict] = []
    for edge, intervals in _intervals_by_edge(transfers, model).items():
        active: list[tuple[int, int, int]] = []
        for start, end, transfer_id in sorted(intervals):
            active = [item for item in active if item[1] > start]
            for other_start, other_end, other_id in active:
                overlap_end = min(end, other_end)
                if overlap_end > start:
                    conflicts.append(PathConflict(
                        model=model, edge=edge,
                        first_transfer_id=min(other_id, transfer_id),
                        second_transfer_id=max(other_id, transfer_id),
                        overlap_start=start, overlap_end=overlap_end,
                        overlap_beats=overlap_end - start,
                    ))
            active.append((start, end, transfer_id))
    return conflicts


def _edge_rows(transfers: list[PathTransfer], conflicts: list[PathConflict], model: str) -> list[dict[str, Any]]:
    conflict_count: dict[str, int] = defaultdict(int)
    for conflict in conflicts:
        conflict_count[conflict.edge] += 1
    rows: list[dict[str, Any]] = []
    for edge, intervals in sorted(_intervals_by_edge(transfers, model).items()):
        events: list[tuple[int, int]] = []
        for start, end, _ in intervals:
            events.extend(((start, 1), (end, -1)))
        concurrency = current = 0
        previous: int | None = None
        union = 0
        for time, delta in sorted(events, key=lambda item: (item[0], item[1])):
            if previous is not None and current > 0:
                union += time - previous
            current += delta
            concurrency = max(concurrency, current)
            previous = time
        rows.append({
            "model": model, "edge": edge, "transfer_count": len(intervals),
            "sum_busy_beats": sum(end - start for start, end, _ in intervals),
            "union_busy_beats": union, "max_concurrency": concurrency,
            "conflict_count": conflict_count.get(edge, 0),
        })
    return rows


def audit_path_occupancy(trace: Iterable[ScheduledOperation]) -> dict[str, Any]:
    transfers = reconstruct_transfer_paths(trace)
    declared = find_path_conflicts(transfers, "declared_resource")
    candidate = find_path_conflicts(transfers, "candidate_grid")
    candidate_pairs = {(c.first_transfer_id, c.second_transfer_id) for c in candidate}
    declared_rows = _edge_rows(transfers, declared, "declared_resource")
    candidate_rows = _edge_rows(transfers, candidate, "candidate_grid")
    candidate_pair_intervals = {
        (c.first_transfer_id, c.second_transfer_id, c.overlap_start, c.overlap_end)
        for c in candidate
    }
    summary = {
        "transfer_count": len(transfers),
        "sam_endpoint_transfer_count": sum(
            t.source_kind == "SAM" or t.target_kind == "SAM" for t in transfers
        ),
        "exact_sam_endpoint_count": sum(t.exact_sam_endpoint for t in transfers),
        "declared_resource_conflict_count": len(declared),
        "candidate_grid_conflict_count": len(candidate),
        "candidate_grid_conflicting_transfer_pair_count": len(candidate_pairs),
        "candidate_grid_unique_pair_overlap_beats": sum(
            end - start for _, _, start, end in candidate_pair_intervals
        ),
        "candidate_grid_conflicting_edge_count": sum(
            int(row["conflict_count"]) > 0 for row in candidate_rows
        ),
        "candidate_grid_max_edge_concurrency": max(
            (int(row["max_concurrency"]) for row in candidate_rows), default=0
        ),
        "candidate_grid_is_assumption": True,
        "candidate_grid_layout": "bank b=(0,2b), shared trunk x=2, CR c=(4,2c)",
        "line_sam_column_semantics": "stored and audited as endpoint identity; excluded from seek latency",
    }
    return {
        "transfers": transfers,
        "conflicts": [*declared, *candidate],
        "edge_rows": [*declared_rows, *candidate_rows],
        "summary": summary,
    }
