from __future__ import annotations

"""Cuccaro/CDKM adderの汎用schedulerと5-slot pipelineの比較。"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from experiments.common import write_csv
from lsqca_eval.architecture import ArchitectureConfig
from lsqca_eval.scheduling_policy import ExecutionConfig, SchedulingPolicy
from lsqca_eval.scheduler import schedule_program
from lsqca_eval.schedulers.adder_pipeline import schedule_adder_pipeline
from lsqca_eval.programs.cdkm_adder import build_cdkm_adder


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    cr_slots: int
    retain_cr_residents: bool
    layout: str = "sequential"
    split_io: bool = False


CASES = (
    Case("base_4slot", "common", 4, False),
    Case("residency_4slot", "common", 4, True),
    Case("residency_5slot", "common", 5, True),
    Case("pipeline_5slot_sequential_shared_io", "pipeline", 5, True, "sequential", False),
    Case("pipeline_5slot_sequential_split_io", "pipeline", 5, True, "sequential", True),
    Case("pipeline_5slot_parity_shared_io", "pipeline", 5, True, "parity", False),
    Case("pipeline_5slot_parity_split_io", "pipeline", 5, True, "parity", True),
)


def execution_config(sam_type: str, case: Case) -> ExecutionConfig:
    return ExecutionConfig(
        architecture=ArchitectureConfig(
            cr_count=1,
            cr_slots=case.cr_slots,
            banks=4,
            sam_type=sam_type,
            cache_slots_per_bank=0,
            allow_in_memory_single_qubit=True,
            direct_cr_transfer=True,
            cr_to_cr_latency=1,
        ),
        policy=SchedulingPolicy(
            store_policy="home",
            placement_policy="sequential",
            use_staging_prefetch=False,
            stage_demand_loads=False,
            prefetch_policy="none",
            retain_cr_residents=case.retain_cr_residents,
            final_flush=True,
        ),
    )


def run_case(bits: int, sam_type: str, case: Case) -> dict[str, object]:
    program = build_cdkm_adder(bits)
    cfg = execution_config(sam_type, case)
    started = time.perf_counter()
    if case.kind == "common":
        trace, metrics = schedule_program(program, cfg)
    else:
        # 専用pipelineはprologue/epilogueを含めてhomeへ戻すため、
        # common final_flushは呼ばれない。
        trace, metrics = schedule_adder_pipeline(
            program,
            cfg,
            split_io=case.split_io,
            layout_policy=case.layout,  # type: ignore[arg-type]
        )
    seconds = time.perf_counter() - started
    return {
        "bits": bits,
        "total_qubits": len(program.qubits),
        "sam_type": sam_type,
        "case": case.name,
        "scheduler": case.kind,
        "cr_count": 1,
        "cr_slots": case.cr_slots,
        "retain_cr_residents": case.retain_cr_residents,
        "layout": case.layout,
        "cr_io_model": "split_load_store" if case.split_io else "shared",
        "seconds": round(seconds, 6),
        **metrics,
    }
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=31)
    parser.add_argument("--sam-type", choices=("line-sam", "point-sam"), action="append")
    parser.add_argument("--case", choices=tuple(case.name for case in CASES), action="append")
    parser.add_argument("--output", type=Path, default=Path("results/adder/all_results.csv"))
    args = parser.parse_args()
    sam_types = args.sam_type or ["line-sam", "point-sam"]
    cases = tuple(case for case in CASES if not args.case or case.name in args.case)
    rows: list[dict[str, object]] = []
    for sam_type in sam_types:
        for case in cases:
            try:
                row = run_case(args.bits, sam_type, case)
                rows.append(row)
                print("OK", sam_type, case.name, row["total_beats"], flush=True)
            except Exception as exc:
                rows.append({"bits": args.bits, "sam_type": sam_type, "case": case.name, "error": f"{type(exc).__name__}: {exc}"})
                print("ERR", sam_type, case.name, exc, flush=True)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
