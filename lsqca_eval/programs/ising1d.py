from __future__ import annotations

"""1D nearest-neighbor Ising kernel builder。"""

from ..execution_plan import parallel_groups_begin, parallel_groups_end
from ..program_ir import Block, Op, Program, Stmt


def _emit_rzz(a: str, b: str, angle: float) -> list[Op]:
    return [Op("CX", (a, b)), Op("RZ", (b,), (angle,)), Op("CX", (a, b))]


def build_ising1d_kernel(n: int, *, layers: int = 1, annotate_parallel: bool = False) -> Program:
    """1D Ising-type kernelを構築する。

    初期H、nearest-neighbor ZZ layer、全qubit RZ layerを含む。
    annotate_parallel=True の場合、even/odd matchingをannotationとして保持する。
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if layers <= 0:
        raise ValueError("layers must be positive")
    q = [f"q{i}" for i in range(n)]
    body: list[Stmt] = [Op("H", (item,)) for item in q]
    for layer in range(layers):
        for parity in range(2):
            pairs = [(q[i], q[i + 1]) for i in range(parity, n - 1, 2)]
            layer_body: list[Stmt] = []
            name = f"ising_layer_{layer}_parity_{parity}"
            if annotate_parallel and pairs:
                layer_body.append(parallel_groups_begin(pairs, name=name))
            for a, b in pairs:
                layer_body.extend(_emit_rzz(a, b, 0.2 + 0.01 * layer))
            if annotate_parallel and pairs:
                layer_body.append(parallel_groups_end(pairs, name=name))
            body.append(Block("ising_parity", name, layer_body, {"layer": layer, "parity": parity}))
        for item in q:
            body.append(Op("RZ", (item,), (0.05,)))
    return Program(name=f"ising_n{n}_layers{layers}", qubits=q, body=body)
