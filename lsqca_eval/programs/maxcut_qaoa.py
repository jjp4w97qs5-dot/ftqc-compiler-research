from __future__ import annotations

"""MaxCut QAOA kernel builder。"""

from ..execution_plan import parallel_groups_begin, parallel_groups_end
from ..program_ir import Block, Op, Program, Stmt


def default_maxcut_edges(n: int) -> list[tuple[int, int]]:
    """cycleと短いchordを持つ決定的な評価用MaxCut graphを返す。"""
    if n <= 1:
        return []
    edges = [(i, (i + 1) % n) for i in range(n)]
    if n > 3:
        edges.extend((i, (i + 2) % n) for i in range(0, n, 2))
    return sorted({tuple(sorted(edge)) for edge in edges})


def greedy_edge_coloring(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """qubitを共有しないedge matchingへgreedyに分割する。"""
    layers: list[list[tuple[int, int]]] = []
    for edge in edges:
        u, v = edge
        for layer in layers:
            used = {node for item in layer for node in item}
            if u not in used and v not in used:
                layer.append(edge)
                break
        else:
            layers.append([edge])
    return layers


def _emit_rzz(a: str, b: str, angle: float) -> list[Op]:
    return [Op("CX", (a, b)), Op("RZ", (b,), (angle,)), Op("CX", (a, b))]


def _emit_rx(q: str, angle: float) -> list[Op]:
    return [Op("H", (q,)), Op("RZ", (q,), (angle,)), Op("H", (q,))]


def build_maxcut_qaoa(
    n: int,
    edges: list[tuple[int, int]] | None = None,
    p: int = 1,
    *,
    annotate_parallel: bool = False,
) -> Program:
    """MaxCut QAOA cost/mixer layerを構築する。

    annotate_parallel=True の場合だけ、元のedge-coloring matchingを
    PARALLEL_GROUPS annotationとして保持する。
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if p <= 0:
        raise ValueError("p must be positive")
    q = [f"q{i}" for i in range(n)]
    graph_edges = default_maxcut_edges(n) if edges is None else list(edges)
    edge_layers = greedy_edge_coloring(graph_edges)
    body: list[Stmt] = []
    for layer_index in range(p):
        cost_body: list[Stmt] = []
        for matching_index, matching in enumerate(edge_layers):
            groups = [(q[u], q[v]) for u, v in matching]
            name = f"maxcut_cost_{layer_index}_{matching_index}"
            if annotate_parallel and groups:
                cost_body.append(parallel_groups_begin(groups, name=name))
            for u, v in matching:
                cost_body.extend(_emit_rzz(q[u], q[v], 0.125 * (layer_index + 1)))
            if annotate_parallel and groups:
                cost_body.append(parallel_groups_end(groups, name=name))
        mixer_body: list[Stmt] = []
        for item in q:
            mixer_body.extend(_emit_rx(item, 0.25 * (layer_index + 1)))
        body.append(Block("qaoa_cost", f"cost_{layer_index}", cost_body, {"layer": layer_index}))
        body.append(Block("qaoa_mixer", f"mixer_{layer_index}", mixer_body, {"layer": layer_index}))
    return Program(name=f"maxcut_qaoa_n{n}_p{p}", qubits=q, body=body)
