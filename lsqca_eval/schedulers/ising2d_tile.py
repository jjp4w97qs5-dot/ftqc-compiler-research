from __future__ import annotations

"""2D Isingのtile計画をmachine state上で実行する専用scheduler。"""

from typing import Any

from ..scheduling_policy import ExecutionConfig
from ..execution_trace import ScheduledOperation
from ..machine import (
    execute_cr_gate,
    initialize_machine_state,
    load_sam_to_cr,
    store_cr_to_sam,
    transfer_cr_to_cr,
)
from ..planners.ising2d_plan import (
    classify_edges,
    order_tile_edges_for_reuse,
    qubits_for_tile,
    schedule_boundary_edge_rounds,
    tile_initial_layout,
)
from ..program_ir import Program
from ..programs.ising2d import Ising2DWorkload


# compute-transfer overlap集計の対象となる既存transfer命令集合。
TRANSFER_OPS = {"LD", "ST", "SAM_TO_CACHE", "PREFETCH_CACHE", "CACHE_TO_CR", "CACHE_EVICT_ST", "CACHE_FINAL_ST", "CR_TO_CR"}


class Ising2DTileScheduler:
    """2D Ising tile計画を共通資源関数上で直接scheduleする。"""

    def __init__(self, program: Program, wl: Ising2DWorkload, cfg: ExecutionConfig, overlap: bool):
        # 専用plan実行に必要なProgram、workload、machine stateを初期化する。
        self.wl=wl; self.cfg=cfg; self.overlap=overlap
        layout=tile_initial_layout(
            wl,
            cfg.architecture.cr_count,
            cfg.architecture.sam_type,
        )
        self.st=initialize_machine_state(layout,cfg.architecture)
        self.node_idx=0
        self.reuse_edges=0; self.edge_transitions=0

    def gate(self,op:str,qs:tuple[int,...],cr:int,earliest:int,meta:dict[str,Any]) -> tuple[int,int]:
        """指定CRでgate資源を予約しtraceへ記録する。"""

        earliest=max([earliest,*(self.st.q_ready.get(q,0) for q in qs)])
        start,end=execute_cr_gate(
            op,
            qs,
            self.node_idx,
            None,
            meta,
            cr,
            earliest,
            self.st,
            self.cfg.architecture,
            include_preferred_cr=False,
            metadata_suffix={
                "plan":"tile_pipeline" if self.overlap else "tile_no_overlap"
            },
        )
        self.node_idx += 1
        return start,end

    def load(self,q:int,cr:int,t:int,reason:str) -> int:
        """qubitを必要なCRへ既存transfer経路で移動する。"""

        if self.st.loc[q][0]=="CR" and self.st.loc[q][1]==cr:
            return max(t,self.st.q_ready[q])
        if self.st.loc[q][0]=="CR":
            return transfer_cr_to_cr(q,cr,t,self.st,self.cfg.architecture)
        return load_sam_to_cr(q,cr,t,self.st,self.cfg.architecture,reason=reason)

    def store(self,q:int,t:int,reason:str="planned_store") -> int:
        """qubitを計画済みhome bankへstoreする。"""

        bank,row,col=self.st.home[q]
        return store_cr_to_sam(
            q,
            t,
            self.st,
            self.cfg.architecture,
            bank=bank,
            row=row,
            col=col,
        )

    def flush_cr(self,cr:int,t:int) -> int:
        """指定CRのresident qubitを昇順でstoreする。"""

        for q in sorted(list(self.st.cr_res[cr])):
            t=self.store(q,t,"phase_flush")
        return t

    def schedule_rx_sequence(self,cr:int,sites:list[int],start_floor:int,step:int) -> int:
        """tile内のfield H-RZ-H列を従来のLD/ST順でscheduleする。"""

        if not sites: return start_floor
        if not self.overlap:
            t=start_floor
            for q in sites:
                t=self.load(q,cr,t,"field_ld")
                _,t=self.gate("H",(q,),cr,t,{"phase":"field","trotter_step":step,"tile":cr})
                _,t=self.gate("RZ",(q,),cr,t,{"phase":"field","trotter_step":step,"tile":cr})
                _,t=self.gate("H",(q,),cr,t,{"phase":"field","trotter_step":step,"tile":cr})
                t=self.store(q,t,"field_st")
            return t
        current=sites[0]
        ready=self.load(current,cr,start_floor,"field_initial_ld")
        previous=None
        for i,current in enumerate(sites):
            next_q=sites[i+1] if i+1<len(sites) else None
            h1s,h1e=self.gate("H",(current,),cr,ready,{"phase":"field","trotter_step":step,"tile":cr})
            rzs, rze=self.gate("RZ",(current,),cr,h1e,{"phase":"field","trotter_step":step,"tile":cr})
            io=rzs
            if previous is not None:
                io=self.store(previous,io,"field_pipeline_st")
            if next_q is not None:
                io=self.load(next_q,cr,io,"field_pipeline_ld")
            _,h2e=self.gate("H",(current,),cr,rze,{"phase":"field","trotter_step":step,"tile":cr})
            ready=max(h2e, self.st.q_ready.get(next_q,0) if next_q is not None else h2e)
            previous=current
        return self.flush_cr(cr,max(ready,max((self.st.q_ready[q] for q in self.st.cr_res[cr]),default=ready)))

    def schedule_rzz_sequence(self,cr:int,edges:list[tuple[int,int,str]],start_floor:int,step:int) -> int:
        """tile内部RZZ列をendpoint reuseと従来のLD/ST順でscheduleする。"""

        if not edges: return start_floor
        if not self.overlap:
            t=start_floor
            current_res:set[int]=set()
            for k,e in enumerate(edges):
                current={e[0],e[1]}; nxt={edges[k+1][0],edges[k+1][1]} if k+1<len(edges) else set()
                for q in sorted(current-current_res): t=self.load(q,cr,t,"edge_ld")
                _,t=self.gate("CX",(e[0],e[1]),cr,t,{"phase":"interaction","trotter_step":step,"tile":cr,"edge_kind":"internal","edge_index":k})
                _,t=self.gate("RZ",(e[1],),cr,t,{"phase":"interaction","trotter_step":step,"tile":cr,"edge_kind":"internal","edge_index":k})
                _,t=self.gate("CX",(e[0],e[1]),cr,t,{"phase":"interaction","trotter_step":step,"tile":cr,"edge_kind":"internal","edge_index":k})
                keep=current & nxt
                if k+1<len(edges):
                    self.edge_transitions+=1; self.reuse_edges += int(bool(keep))
                for q in sorted(current-keep): t=self.store(q,t,"edge_st")
                current_res=keep
            return self.flush_cr(cr,t)
        first=edges[0]
        ready=start_floor
        for q in first[:2]: ready=self.load(q,cr,ready,"edge_initial_ld")
        dead_prev:set[int]=set()
        for k,e in enumerate(edges):
            current={e[0],e[1]}; nxt={edges[k+1][0],edges[k+1][1]} if k+1<len(edges) else set()
            if k+1<len(edges):
                self.edge_transitions += 1; self.reuse_edges += int(bool(current & nxt))
            c1s,c1e=self.gate("CX",(e[0],e[1]),cr,ready,{"phase":"interaction","trotter_step":step,"tile":cr,"edge_kind":"internal","edge_index":k})
            rzs,rze=self.gate("RZ",(e[1],),cr,c1e,{"phase":"interaction","trotter_step":step,"tile":cr,"edge_kind":"internal","edge_index":k})
            io=rzs
            for q in sorted(dead_prev): io=self.store(q,io,"edge_pipeline_st")
            for q in sorted(nxt-current): io=self.load(q,cr,io,"edge_pipeline_ld")
            _,c2e=self.gate("CX",(e[0],e[1]),cr,rze,{"phase":"interaction","trotter_step":step,"tile":cr,"edge_kind":"internal","edge_index":k})
            ready=max(c2e,max((self.st.q_ready.get(q,0) for q in nxt),default=c2e))
            dead_prev=current-nxt
        return self.flush_cr(cr,max(ready,max((self.st.q_ready[q] for q in self.st.cr_res[cr]),default=ready)))

    def schedule_boundary_round(self,round_:list[tuple[int,tuple[int,int,str]]],start_floor:int,step:int,round_idx:int) -> int:
        """1 boundary roundのRZZとLD/STをowner CRごとにscheduleする。"""

        ends=[]
        for owner,e in round_:
            t=start_floor
            for q in e[:2]: t=self.load(q,owner,t,"boundary_ld")
            _,t=self.gate("CX",(e[0],e[1]),owner,t,{"phase":"interaction","trotter_step":step,"tile":owner,"edge_kind":"boundary","boundary_round":round_idx})
            _,t=self.gate("RZ",(e[1],),owner,t,{"phase":"interaction","trotter_step":step,"tile":owner,"edge_kind":"boundary","boundary_round":round_idx})
            _,t=self.gate("CX",(e[0],e[1]),owner,t,{"phase":"interaction","trotter_step":step,"tile":owner,"edge_kind":"boundary","boundary_round":round_idx})
            for q in e[:2]: t=self.store(q,t,"boundary_st")
            ends.append(t)
        return max(ends,default=start_floor)

    def run(self) -> tuple[list[ScheduledOperation],dict[str,Any],dict[str,Any]]:
        """全Trotter stepを実行しtrace・metrics・plan集計を返す。"""

        internal,boundary=classify_edges(self.wl)
        orders={cr:order_tile_edges_for_reuse(internal[cr],self.wl.width) for cr in range(4)}
        rounds=schedule_boundary_edge_rounds(boundary,self.wl)
        step_floor=0
        for step in range(self.wl.steps):
            tile_ends=[]
            for cr in range(4):
                field_end=self.schedule_rx_sequence(cr,qubits_for_tile(self.wl,cr),step_floor,step)
                tile_ends.append(self.schedule_rzz_sequence(cr,orders[cr],field_end,step))
            boundary_floor=max(tile_ends)
            for ridx,round_ in enumerate(rounds):
                boundary_floor=self.schedule_boundary_round(round_,boundary_floor,step,ridx)
            step_floor=boundary_floor
        for cr in range(self.cfg.architecture.cr_count):
            for q in list(self.st.cr_res.get(cr,set())):
                step_floor=self.store(int(q),step_floor)
        metrics=finalize_metrics(self.st,self.cfg,self.node_idx)
        plan={
            "internal_edge_count":sum(len(x) for x in internal.values()),
            "boundary_edge_count":len(boundary),
            "boundary_round_count":len(rounds),
            "edge_transition_count":self.edge_transitions,
            "shared_endpoint_transition_count":self.reuse_edges,
            "shared_endpoint_transition_rate":self.reuse_edges/max(1,self.edge_transitions),
            "overlap_enabled":self.overlap,
        }
        return self.st.trace,metrics,plan


def interval_union(intervals:list[tuple[int,int]]) -> int:
    """区間集合のunion長を計算する。"""

    total=0; cur=None
    for s,e in sorted(intervals):
        if e<=s: continue
        if cur is None: cur=[s,e]
        elif s<=cur[1]: cur[1]=max(cur[1],e)
        else: total+=cur[1]-cur[0]; cur=[s,e]
    return total+(0 if cur is None else cur[1]-cur[0])


def overlap_union(a:list[tuple[int,int]],b:list[tuple[int,int]]) -> int:
    """2種類の区間集合が同時にactiveな総時間を計算する。"""

    events=[]
    for s,e in a: events += [(s,1,0),(e,-1,0)]
    for s,e in b: events += [(s,1,1),(e,-1,1)]
    active=[0,0]; prev=None; out=0
    for t,d,k in sorted(events,key=lambda x:(x[0],x[1])):
        if prev is not None and active[0]>0 and active[1]>0: out += t-prev
        active[k]+=d; prev=t
    return out


def avg_active_cr(trace:list[ScheduledOperation],total:int,cr_count:int) -> float:
    """既存定義で全beatに対する平均active CR数を計算する。"""

    events=[]
    for op in trace:
        if op.reason=="cr_gate" and op.cr_id is not None and op.end>op.start:
            events += [(op.start,1),(op.end,-1)]
    active=0; prev=0; area=0
    for t,d in sorted(events,key=lambda x:(x[0],x[1])):
        area += active*(t-prev); active+=d; prev=t
    return area/max(1,total)


def finalize_metrics(st:Any,cfg:ExecutionConfig,gate_count:int) -> dict[str,Any]:
    """専用scheduler stateから従来と同じmetrics辞書を生成する。"""

    total=max((op.end for op in st.trace),default=0)
    def busy(prefix:str)->int:
        # 指定prefixに属する資源のbusy beatを合計する。
        return sum(v for k,v in st.statistics.resource_busy.items() if k.startswith(prefix))
    compute=[(op.start,op.end) for op in st.trace if op.reason in {"cr_gate","single_in_cr","inmemory_single","cache_single"}]
    transfer=[(op.start,op.end) for op in st.trace if op.op in TRANSFER_OPS]
    transfer_union=interval_union(transfer); overlap=overlap_union(compute,transfer)
    final_sam=[(loc[1],loc[2],loc[3]) for loc in st.loc.values() if loc[0]=="SAM"]
    metrics = {
        "total_beats":total,"gate_count":gate_count,"ld_count":st.statistics.ld_count,"st_count":st.statistics.st_count,
        "transfer_ops":st.statistics.ld_count+st.statistics.st_count+st.statistics.cache_prefetch_count+st.statistics.cr_to_cr_transfer_count,
        "cache_prefetch_count":st.statistics.cache_prefetch_count,"cache_hit_count":st.statistics.cache_hit_count,
        "prefetch_hidden_hits":st.statistics.prefetch_hidden_hits,"prefetch_hide_rate":st.statistics.prefetch_hidden_hits/max(1,st.statistics.cache_prefetch_count),
        "inmemory_count":st.statistics.inmemory_count,"route_wait":st.statistics.route_wait,
        "bank_wait":st.statistics.wait_by_category.get("bank",0),"mem_port_wait":st.statistics.wait_by_category.get("mem_port",0),
        "cr_port_wait":st.statistics.wait_by_category.get("cr_port",0),"cr_compute_wait":st.statistics.wait_by_category.get("cr_compute",0),
        "hub_wait":st.statistics.wait_by_category.get("hub",0),"compute_beats":st.statistics.compute_beats,"inmemory_beats":st.statistics.inmemory_beats,
        "transfer_beats":st.statistics.transfer_beats,"bank_busy_beats":busy("bank:"),"mem_port_busy_beats":busy("mem_port:"),
        "cr_port_busy_beats":busy("cr_port:"),"cr_compute_busy_beats":busy("cr_compute:"),"hub_busy_beats":busy("hub:"),
        "local_transfers":st.statistics.local_transfers,"nonlocal_transfers":st.statistics.nonlocal_transfers,
        "cr_to_cr_transfer_count":st.statistics.cr_to_cr_transfer_count,
        "local_transfer_ratio":st.statistics.local_transfers/max(1,st.statistics.local_transfers+st.statistics.nonlocal_transfers),
        "trace_len":len(st.trace),"max_cr_occupancy":st.statistics.max_cr_occupancy,"cr_overflow_events":st.statistics.cr_overflow_events,
        "cr_slot_eviction_events":st.statistics.cr_slot_eviction_events,"final_cr_resident_count":sum(len(v) for v in st.cr_res.values()),
        "final_cache_resident_count":sum(q is not None for x in st.cache_res.values() for q in x),
        "sam_cell_count":sum(len(x) for x in st.sam_cells.values()),"initial_distinct_sam_columns":len({x[2] for x in st.home.values()}),
        "final_sam_cell_collision_count":len(final_sam)-len(set(final_sam)),"final_sam_out_of_layout_count":sum((loc[2],loc[3]) not in st.sam_cells[loc[1]] for loc in st.loc.values() if loc[0]=="SAM"),
        "compute_transfer_overlap_beats":overlap,"transfer_union_beats":transfer_union,"exposed_transfer_beats":transfer_union-overlap,
        "avg_active_cr":avg_active_cr(st.trace,total,cfg.architecture.cr_count),
    }
    if st.magic_state is not None:
        # 共通machineで追跡したlocal MSF指標を専用schedulerにも追加する。
        metrics.update(st.magic_state.metrics(total))
    return metrics
