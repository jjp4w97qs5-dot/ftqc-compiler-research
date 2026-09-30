from __future__ import annotations

"""1D Ising kernel用のreuse-aware interaction計画。"""

from ..execution_plan import parallel_groups_begin, parallel_groups_end
from ..program_ir import Op, Program, Stmt
from .commuting_interactions import InteractionArchitecture, PlannedPair, plan_commuting_pairs


def _rzz_ops(left: str, right: str, angle: float) -> list[Op]:
    # planned nearest-neighbor interactionを既存のCX-RZ-CX列へ展開する。
    return [Op("CX", (left, right)), Op("RZ", (right,), (angle,)), Op("CX", (left, right))]


def build_ising1d_plan(n: int, architecture: InteractionArchitecture) -> Program:
    """1D Ising interactionをreuse-aware stream/waveへ並べ替える。"""

    q = [f"q{i}" for i in range(n)]
    pairs = [PlannedPair(q[i], q[i + 1], (0.2,)) for i in range(n - 1)]
    plan = plan_commuting_pairs(q, pairs, cr_count=architecture.cr_count)
    body: list[Stmt] = [Op("H", (item,)) for item in q]
    for wave_index, wave in enumerate(plan.waves):
        groups = [pair.qubits for pair in wave.groups]
        name = f"ising_planned_wave_{wave_index}"
        body.append(parallel_groups_begin(groups, name=name, planned_crs=wave.crs))
        for pair in wave.groups:
            body.extend(_rzz_ops(pair.left, pair.right, float(pair.payload[0])))
        body.append(parallel_groups_end(groups, name=name))
    body.extend(Op("RZ", (item,), (0.05,)) for item in q)
    return Program(name=f"ising1d_n{n}_planned", qubits=q, body=body)


def ising1d_placement_policy(sam_type: str) -> str:
    """SAM種別に対応する従来の1D Ising配置policyを返す。"""

    return "round_robin" if sam_type == "line-sam" else "sequential"
