from __future__ import annotations

"""2D transverse-field Isingの共通scheduler・tile plan比較driver。"""

import argparse
import gzip
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.common import write_csv
from lsqca_eval.architecture import ArchitectureConfig
from lsqca_eval.scheduling_policy import ExecutionConfig, SchedulingPolicy
from lsqca_eval.scheduler import schedule_program
from lsqca_eval.execution_trace import ScheduledOperation
from lsqca_eval.programs.ising2d import Ising2DWorkload, build_ising2d_program
from lsqca_eval.routing_audit import audit_path_occupancy
from lsqca_eval.schedulers.ising2d_tile import Ising2DTileScheduler


# 2D Ising成果物へ記録する従来のmodel識別子。
MODEL_ID = "lsqca-execution-model+ising2d-tile-plan"


def base_config(sam_type:str,prefetch:bool=True) -> ExecutionConfig:
    """2D Ising比較で共有する既存4CR machine設定を返す。"""

    return ExecutionConfig(
        architecture=ArchitectureConfig(
            cr_count=4, cr_slots=4, banks=4, sam_type=sam_type,
            cache_slots_per_bank=2, allow_in_memory_single_qubit=True,
        ),
        policy=SchedulingPolicy(
            retain_cr_residents=True, use_staging_prefetch=prefetch,
            stage_demand_loads=prefetch,
            prefetch_policy="compute_window_demand_first" if prefetch else "none",
            placement_policy="round_robin", store_policy="score",
            use_execution_plan=False, max_prefetch_per_step=2,
            final_flush=True, record_plan_metadata=True,
        ),
    )


def run_case(wl:Ising2DWorkload,sam_type:str,method:str) -> tuple[dict[str,Any],list[ScheduledOperation]]:
    """1手法を実行しrouting audit済みの結果行とtraceを返す。"""

    program=build_ising2d_program(wl); cfg=base_config(sam_type,method=="low_level_v2")
    started=time.perf_counter()
    if method.startswith("high_level"):
        overlap=method.endswith("pipeline")
        cfg = replace(cfg, policy=replace(cfg.policy, store_policy="home"))
        scheduler=Ising2DTileScheduler(program,wl,cfg,overlap)
        trace,metrics,plan=scheduler.run()
    else:
        trace,metrics=schedule_program(program,cfg); plan={}
    elapsed=time.perf_counter()-started
    path_summary=audit_path_occupancy(trace)["summary"]
    valid=all([metrics["cr_overflow_events"]==0,metrics["final_cr_resident_count"]==0,metrics["final_cache_resident_count"]==0,metrics["final_sam_cell_collision_count"]==0,metrics["final_sam_out_of_layout_count"]==0,path_summary["declared_resource_conflict_count"]==0])
    row={"model_id":MODEL_ID,"method":method,"sam_type":sam_type,"height":wl.height,"width":wl.width,"steps":wl.steps,
         "cr_count":4,"cr_slots":4,"banks":4,"cache_slots_per_bank":2,"rotation_epsilon":cfg.architecture.rotation_epsilon,
         "schedule_seconds":elapsed,"case_valid":valid,**metrics,**plan,**path_summary}
    return row,trace


def write_trace(path:Path,trace:list[ScheduledOperation]) -> None:
    """ScheduledOperationを従来schemaのgzip JSONLとして保存する。"""

    with gzip.open(path,'wt',encoding='utf-8') as f:
        for i,op in enumerate(trace):
            f.write(json.dumps({"trace_index":i,"op":op.op,"qubits":op.qubits,"start":op.start,"end":op.end,"cr_id":op.cr_id,"bank_id":op.bank_id,"reason":op.reason,"meta":op.meta},ensure_ascii=False,default=str)+'\n')


def main() -> None:
    """既定の両SAM・3手法を実行して成果物を出力する。"""

    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/ising2d'); ap.add_argument('--height',type=int,default=10); ap.add_argument('--width',type=int,default=10); ap.add_argument('--steps',type=int,default=8); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    wl=Ising2DWorkload(args.height,args.width,args.steps)
    rows=[]
    for sam in ('line-sam','point-sam'):
        for method in ('low_level_v2','high_level_tile_no_overlap','high_level_tile_pipeline'):
            row,trace=run_case(wl,sam,method); rows.append(row); write_trace(out/f'{sam}__{method}.jsonl.gz',trace)
            print(sam,method,row['total_beats'],row['case_valid'],flush=True)
    for sam in ('line-sam','point-sam'):
        base=next(r for r in rows if r['sam_type']==sam and r['method']=='low_level_v2')['total_beats']
        for r in rows:
            if r['sam_type']==sam:
                r['ratio_to_low_level']=r['total_beats']/base; r['reduction_vs_low_level_percent']=(1-r['total_beats']/base)*100
    write_csv(out/'results.csv',rows)
    (out/'run_metadata.json').write_text(json.dumps({"model_id":MODEL_ID,"workload":wl.__dict__,"methods":[r['method'] for r in rows],"notes":["Same gate decomposition and latency functions for all methods.","High-level methods only specialize tile placement, edge ownership/order, residency, and LD/ST issue times.","No cross-step retention is used; boundary phase is conservatively flushed."]},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
