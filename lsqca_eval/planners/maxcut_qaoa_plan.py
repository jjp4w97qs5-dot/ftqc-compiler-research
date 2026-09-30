from __future__ import annotations

"""MaxCut QAOA kernel用のreuse-aware interaction計画。"""

from ..execution_plan import parallel_groups_begin, parallel_groups_end
from ..program_ir import Block, Op, Program, Stmt
from ..programs.maxcut_qaoa import default_maxcut_edges
from .commuting_interactions import InteractionArchitecture, PlannedPair, plan_commuting_pairs


def _rzz_ops(left: str, right: str, angle: float) -> list[Op]:
    # planned cost interactionを既存のCX-RZ-CX列へ展開する。
    return [Op("CX", (left, right)), Op("RZ", (right,), (angle,)), Op("CX", (left, right))]


def build_maxcut_qaoa_plan(n: int, architecture: InteractionArchitecture) -> Program:
    """MaxCut cost interactionをreuse-aware stream/waveへ並べ替える。"""

    q = [f"q{i}" for i in range(n)]
    pairs = [PlannedPair(q[u], q[v], (0.125,)) for u, v in default_maxcut_edges(n)]
    plan = plan_commuting_pairs(q, pairs, cr_count=architecture.cr_count)
    cost: list[Stmt] = []
    for wave_index, wave in enumerate(plan.waves):
        groups = [pair.qubits for pair in wave.groups]
        name = f"maxcut_planned_wave_{wave_index}"
        cost.append(parallel_groups_begin(groups, name=name, planned_crs=wave.crs))
        for pair in wave.groups:
            cost.extend(_rzz_ops(pair.left, pair.right, float(pair.payload[0])))
        cost.append(parallel_groups_end(groups, name=name))
    mixer: list[Stmt] = []
    for item in q:
        mixer.extend([Op("H", (item,)), Op("RZ", (item,), (0.25,)), Op("H", (item,))])
    return Program(
        name=f"maxcut_qaoa_n{n}_planned",
        qubits=q,
        body=[Block("qaoa_cost", "cost_planned", cost), Block("qaoa_mixer", "mixer", mixer)],
    )


def maxcut_qaoa_placement_policy(_sam_type: str) -> str:
    """MaxCut planで従来使用していた配置policyを返す。"""

    return "plan_affinity"
