from __future__ import annotations

"""MaxCut QAOA kernelと1D Ising kernelの高水準計画比較。"""

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from experiments.common import ExperimentCase, run_experiment_cases, write_csv
from lsqca_eval.architecture import ArchitectureConfig
from lsqca_eval.scheduling_policy import ExecutionConfig, SchedulingPolicy
from lsqca_eval.scheduler import schedule_program
from lsqca_eval.lowering import LoweredOp, lower_program
from lsqca_eval.planners.commuting_interactions import InteractionArchitecture, infer_interaction_architecture
from lsqca_eval.planners.ising1d_plan import build_ising1d_plan, ising1d_placement_policy
from lsqca_eval.planners.maxcut_qaoa_plan import build_maxcut_qaoa_plan, maxcut_qaoa_placement_policy
from lsqca_eval.program_ir import Program
from lsqca_eval.programs.ising1d import build_ising1d_kernel
from lsqca_eval.programs.maxcut_qaoa import build_maxcut_qaoa


@dataclass(frozen=True)
class ProgramPlan:
    """各programの参照builder・planner・配置policyをまとめる。"""

    build_reference: Callable[[int, bool], Program]
    build_planned: Callable[[int, InteractionArchitecture], Program]
    placement_policy: Callable[[str], str]


# program名から参照builder・専用planner・配置policyを選ぶ登録表。
PROGRAM_PLANS = {
    "maxcut_qaoa": ProgramPlan(
        build_reference=lambda n, annotate_parallel: build_maxcut_qaoa(n, p=1, annotate_parallel=annotate_parallel),
        build_planned=build_maxcut_qaoa_plan,
        placement_policy=maxcut_qaoa_placement_policy,
    ),
    "ising1d": ProgramPlan(
        build_reference=lambda n, annotate_parallel: build_ising1d_kernel(n, layers=1, annotate_parallel=annotate_parallel),
        build_planned=build_ising1d_plan,
        placement_policy=ising1d_placement_policy,
    ),
}
# CLIで受け付けるprogram名を登録順のまま保持する。
PROGRAMS = tuple(PROGRAM_PLANS)


@dataclass(frozen=True)
class Method:
    name: str
    cr_count: int
    low_level: bool
    use_plan: bool = False
    planned_placement: bool = False
    overlap_transfers: bool = False


def methods(architecture: InteractionArchitecture) -> tuple[Method, ...]:
    return (
        Method("base_1cr", 1, False),
        Method("four_cr_only", architecture.cr_count, False),
        Method("low_level_4cr", architecture.cr_count, True),
        Method("hl_order_cr_4cr", architecture.cr_count, False, use_plan=True),
        Method("hl_order_cr_placement_4cr", architecture.cr_count, False, use_plan=True, planned_placement=True),
        Method("hl_full_plan_4cr", architecture.cr_count, False, use_plan=True, planned_placement=True, overlap_transfers=True),
    )


def build_reference_program(program_name: str, n: int, *, annotate_parallel: bool) -> Program:
    # 実験で選択されたprogramの参照builderを呼び出す。
    return PROGRAM_PLANS[program_name].build_reference(n, annotate_parallel)


def execution_config(method: Method, sam_type: str, architecture: InteractionArchitecture, program_name: str) -> ExecutionConfig:
    planned = method.use_plan
    low = method.low_level
    if method.name in {"base_1cr", "four_cr_only"}:
        placement = "sequential"
    elif method.name == "low_level_4cr":
        placement = "round_robin"
    elif method.planned_placement:
        placement = PROGRAM_PLANS[program_name].placement_policy(sam_type)
    else:
        placement = "round_robin"
    return ExecutionConfig(
        architecture=ArchitectureConfig(
            cr_count=method.cr_count,
            cr_slots=architecture.cr_slots if method.cr_count > 1 else 4,
            banks=architecture.banks,
            sam_type=sam_type,
            cache_slots_per_bank=architecture.cache_slots_per_bank if (low or planned) else 0,
            allow_in_memory_single_qubit=True,
            direct_cr_transfer=True,
            cr_to_cr_latency=1,
        ),
        policy=SchedulingPolicy(
            retain_cr_residents=low or planned,
            use_staging_prefetch=low,
            stage_demand_loads=low,
            prefetch_policy="compute_window_demand_first" if low else "none",
            placement_policy=placement,
            store_policy="score" if (low or planned) else "home",
            use_execution_plan=planned,
            execute_parallel_group_plan=planned,
            enforce_planned_cr_assignment=planned,
            overlap_group_transfers=method.overlap_transfers,
            parallel_group_assignment="reuse_balanced" if planned else "ignore",
            max_prefetch_per_step=2,
            final_flush=True,
            record_plan_metadata=True,
        ),
    )


def gate_multiset(program: Program) -> Counter[tuple[str, tuple[int, ...], tuple[tuple[str, str], ...]]]:
    result: Counter[tuple[str, tuple[int, ...], tuple[tuple[str, str], ...]]] = Counter()
    for event in lower_program(program).events:
        if not isinstance(event, LoweredOp):
            continue
        semantic_meta = tuple(sorted((key, repr(value)) for key, value in event.meta.items() if key in {"angle"}))
        result[(event.op, event.qubits, semantic_meta)] += 1
    return result


def run_case(program_name: str, n: int, sam_type: str, method: Method, architecture: InteractionArchitecture) -> dict[str, object]:
    reference = build_reference_program(program_name, n, annotate_parallel=False)
    program = PROGRAM_PLANS[program_name].build_planned(n, architecture) if method.use_plan else reference
    same_gates = gate_multiset(reference) == gate_multiset(program)
    started = time.perf_counter()
    trace, metrics = schedule_program(program, execution_config(method, sam_type, architecture, program_name))
    seconds = time.perf_counter() - started
    return {
        "program": program_name,
        "n": n,
        "sam_type": sam_type,
        "method": method.name,
        "cr_count": method.cr_count,
        "cr_slots": architecture.cr_slots if method.cr_count > 1 else 4,
        "architecture_max_parallel_groups": architecture.max_parallel_groups,
        "architecture_max_group_size": architecture.max_group_size,
        "gate_multiset_match": same_gates,
        "seconds": round(seconds, 6),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--program", choices=PROGRAMS, action="append")
    parser.add_argument("--sam-type", choices=("line-sam", "point-sam"), action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--output", type=Path, default=Path("results/qaoa_ising1d/all_results.csv"))
    args = parser.parse_args()
    program_names = tuple(args.program) if args.program else PROGRAMS
    sam_types = tuple(args.sam_type) if args.sam_type else ("line-sam", "point-sam")
    cases: list[ExperimentCase] = []
    for program_name in program_names:
        annotated = build_reference_program(program_name, args.n, annotate_parallel=True)
        architecture = infer_interaction_architecture(annotated)
        selected_methods = tuple(method for method in methods(architecture) if not args.method or method.name in args.method)
        for sam_type in sam_types:
            for method in selected_methods:
                cases.append(
                    ExperimentCase(
                        labels={"program": program_name, "n": args.n, "sam_type": sam_type, "method": method.name},
                        display=(program_name, sam_type, method.name),
                        run=partial(run_case, program_name, args.n, sam_type, method, architecture),
                    )
                )
    rows = run_experiment_cases(cases)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
