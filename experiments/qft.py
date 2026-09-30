from __future__ import annotations

"""QFTの高水準実行計画比較。"""

import argparse
import json
import time
from collections import Counter
from functools import partial
from pathlib import Path

from experiments.common import ExperimentCase, ProgramMethod, parallel_stats, run_experiment_cases, write_csv
from lsqca_eval.architecture import ArchitectureConfig
from lsqca_eval.scheduling_policy import ExecutionConfig, SchedulingPolicy
from lsqca_eval.scheduler import schedule_program
from lsqca_eval.lowering import LoweredDirective, LoweredOp, lower_program
from lsqca_eval.program_ir import Program
from lsqca_eval.planners.qft_plan import (
    build_qft_blocked_cr_plan,
    build_qft_diagonal_waves,
    build_qft_explicit_cr_waves,
)
from lsqca_eval.programs.qft import build_qft_original_order


METHODS = (
    ProgramMethod("base_order", build_qft_original_order),
    ProgramMethod("wave_order", lambda n: build_qft_diagonal_waves(n, annotate_parallel=False)),
    ProgramMethod("wave_groups", lambda n: build_qft_diagonal_waves(n, annotate_parallel=True)),
    ProgramMethod("wave_explicit", lambda n: build_qft_explicit_cr_waves(n, cr_count=4)),
    ProgramMethod("blocked_cr_plan", lambda n: build_qft_blocked_cr_plan(n, cr_count=4)),
)


def execution_config(sam_type: str, cr_slots: int = 4) -> ExecutionConfig:
    return ExecutionConfig(
        architecture=ArchitectureConfig(
            cr_count=4,
            cr_slots=cr_slots,
            banks=4,
            sam_type=sam_type,
            cache_slots_per_bank=2,
            local_crs_by_bank={bank: (bank,) for bank in range(4)},
            nonlocal_hub_count=1,
            allow_in_memory_single_qubit=True,
            direct_cr_transfer=True,
            cr_to_cr_latency=1,
        ),
        policy=SchedulingPolicy(
            store_policy="score",
            placement_policy="round_robin",
            use_staging_prefetch=True,
            stage_demand_loads=True,
            prefetch_policy="compute_window_demand_first",
            max_prefetch_per_step=2,
            use_execution_plan=True,
            retain_cr_residents=True,
            final_flush=True,
            record_plan_metadata=True,
        ),
    )


def program_counts(program: Program) -> dict[str, int]:
    events = lower_program(program).events
    ops = [event for event in events if isinstance(event, LoweredOp)]
    directives = [event for event in events if isinstance(event, LoweredDirective)]
    counts = Counter(event.op for event in ops)
    return {
        "program_gate_count": len(ops),
        "program_directive_count": len(directives),
        "H_count": counts["H"],
        "CX_count": counts["CX"],
        "RZ_count": counts["RZ"],
    }


def audit_blocked_plan(trace: list) -> dict[str, object]:
    intervals: list[tuple[str, int, int]] = []
    opened: dict[str, int] = {}
    for op in trace:
        if op.op == "CR_LOAD_BARRIER" and "qft_load_source_" in op.reason:
            key = op.reason.split(":", 1)[-1].replace("qft_load_source_", "")
            opened[key] = op.end
        elif op.op == "CR_STORE_BARRIER" and "qft_store_source_" in op.reason:
            key = op.reason.split(":", 1)[-1].replace("qft_store_source_", "")
            start = opened.pop(key, None)
            if start is not None:
                intervals.append((key, start, op.start))
    bad: list[tuple[str, int]] = []
    sam_ops = {"LD", "ST", "SAM_TO_CACHE", "CACHE_FINAL_ST", "CACHE_EVICT_ST"}
    for key, start, end in intervals:
        inside = [op for op in trace if op.start >= start and op.end <= end and op.op in sam_ops]
        if inside:
            bad.append((key, len(inside)))
    return {
        "cross_tile_count_audited": len(intervals),
        "cross_tile_sam_ops": sum(count for _, count in bad),
        "cross_tile_bad_intervals": json.dumps(bad),
    }


def run_case(n: int, sam_type: str, cr_slots: int, method: ProgramMethod) -> dict[str, object]:
    program = method.build(n)
    started = time.perf_counter()
    trace, metrics = schedule_program(program, execution_config(sam_type, cr_slots))
    elapsed = time.perf_counter() - started
    row: dict[str, object] = {
        "n": n,
        "sam_type": sam_type,
        "cr_slots": cr_slots,
        "method": method.name,
        "seconds": round(elapsed, 6),
        **program_counts(program),
        **metrics,
        **parallel_stats(trace, int(metrics["total_beats"])),
    }
    if method.name == "blocked_cr_plan":
        row.update(audit_blocked_plan(trace))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, action="append", help="Qubit数。複数指定可。")
    parser.add_argument("--sam-type", choices=("line-sam", "point-sam"), action="append")
    parser.add_argument("--cr-slots", type=int, action="append")
    parser.add_argument("--method", choices=tuple(method.name for method in METHODS), action="append")
    parser.add_argument("--output", type=Path, default=Path("results/qft/all_results.csv"))
    args = parser.parse_args()
    sizes = args.n or [16, 32, 64, 128]
    sam_types = args.sam_type or ["line-sam", "point-sam"]
    slots = args.cr_slots or [4]
    methods = tuple(method for method in METHODS if not args.method or method.name in args.method)
    cases: list[ExperimentCase] = []
    for n in sizes:
        for sam_type in sam_types:
            for cr_slots in slots:
                for method in methods:
                    cases.append(
                        ExperimentCase(
                            labels={"n": n, "sam_type": sam_type, "cr_slots": cr_slots, "method": method.name},
                            display=(n, sam_type, cr_slots, method.name),
                            run=partial(run_case, n, sam_type, cr_slots, method),
                        )
                    )
    rows = run_experiment_cases(cases)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
