from __future__ import annotations

"""QASMBench multiplierのmacro順・wave順比較。"""

import argparse
import time
from collections import Counter
from functools import partial
from pathlib import Path

from experiments.common import ExperimentCase, ProgramMethod, parallel_stats, run_experiment_cases, write_csv
from lsqca_eval.architecture import ArchitectureConfig
from lsqca_eval.scheduling_policy import ExecutionConfig, SchedulingPolicy
from lsqca_eval.scheduler import schedule_program
from lsqca_eval.lowering import LoweredDirective, LoweredOp, lower_program
from lsqca_eval.planners.qasmbench_multiplier_plan import build_qasmbench_multiplier_wave, wave_order_macros
from lsqca_eval.program_ir import Program
from lsqca_eval.programs.qasmbench_multiplier import (
    build_qasmbench_multiplier,
    decomposed_gate_counts,
)


METHODS = (
    ProgramMethod("qasm_order", lambda n: build_qasmbench_multiplier(n, annotate_scopes=False)),
    ProgramMethod("qasm_order_scope", lambda n: build_qasmbench_multiplier(n, annotate_scopes=True)),
    ProgramMethod("wave_order", lambda n: build_qasmbench_multiplier_wave(n, annotate_parallel=False, explicit_crs=False, cr_count=4)),
    ProgramMethod("wave_groups", lambda n: build_qasmbench_multiplier_wave(n, annotate_parallel=True, explicit_crs=False, cr_count=4)),
    ProgramMethod("wave_explicit_cr", lambda n: build_qasmbench_multiplier_wave(n, annotate_parallel=True, explicit_crs=True, cr_count=4)),
)


def execution_config(sam_type: str) -> ExecutionConfig:
    return ExecutionConfig(
        architecture=ArchitectureConfig(
            cr_count=4,
            cr_slots=4,
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


def program_counts(program: Program) -> dict[str, object]:
    events = lower_program(program).events
    gates = Counter(event.op for event in events if isinstance(event, LoweredOp))
    directives = Counter(event.kind for event in events if isinstance(event, LoweredDirective))
    width = len(program.qubits) // 5
    formula = decomposed_gate_counts(width)
    return {
        "width": width,
        "program_gate_count": sum(gates.values()),
        "program_directive_count": sum(directives.values()),
        "H_count": gates["H"],
        "CX_count": gates["CX"],
        "T_count": gates["T"],
        "TDG_count": gates["TDG"],
        "gate_formula_match": sum(gates.values()) == formula["total"],
        "derived_wave_count": len(wave_order_macros(width)),
    }


def run_case(total_qubits: int, sam_type: str, method: ProgramMethod) -> dict[str, object]:
    program = method.build(total_qubits)
    started = time.perf_counter()
    trace, metrics = schedule_program(program, execution_config(sam_type))
    seconds = time.perf_counter() - started
    return {
        "n": total_qubits,
        "sam_type": sam_type,
        "method": method.name,
        "seconds": round(seconds, 6),
        **program_counts(program),
        **metrics,
        **parallel_stats(trace, int(metrics["total_beats"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, action="append", help="総logical qubit数。5の倍数。")
    parser.add_argument("--sam-type", choices=("line-sam", "point-sam"), action="append")
    parser.add_argument("--method", choices=tuple(method.name for method in METHODS), action="append")
    parser.add_argument("--output", type=Path, default=Path("results/multiplier/all_results.csv"))
    args = parser.parse_args()
    sizes = args.n or [15, 25, 45, 75, 100, 150]
    sam_types = args.sam_type or ["line-sam", "point-sam"]
    methods = tuple(method for method in METHODS if not args.method or method.name in args.method)
    cases: list[ExperimentCase] = []
    for n in sizes:
        for sam_type in sam_types:
            for method in methods:
                cases.append(
                    ExperimentCase(
                        labels={"n": n, "sam_type": sam_type, "method": method.name},
                        display=(n, sam_type, method.name),
                        run=partial(run_case, n, sam_type, method),
                    )
                )
    rows = run_experiment_cases(cases)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
