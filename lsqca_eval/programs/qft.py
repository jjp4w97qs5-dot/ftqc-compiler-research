from __future__ import annotations

"""標準QFT用の構造付きProgramIR builder。"""

import math
from typing import Any

from ..program_ir import Op, Program, Stmt


def _emit_cp(
    control: str,
    target: str,
    angle: float,
    *,
    meta: dict[str, Any] | None = None,
) -> list[Op]:
    """controlled phaseをglobal phaseまで正しい5-gate列へ展開する。"""

    # 各sub-operationへ同じmacro情報と一意な役割名を付ける。
    base = dict(meta or {})

    def tagged(role: str) -> dict[str, Any]:
        """CP内のsub-operation識別子を既存metadataへ追加する。"""

        return {**base, "macro_subop": role} if base else {}

    half = angle / 2.0
    return [
        Op("RZ", (control,), (half,), meta=tagged("rz_control_half")),
        Op("RZ", (target,), (half,), meta=tagged("rz_target_half")),
        Op("CX", (control, target), meta=tagged("cx1")),
        Op("RZ", (target,), (-half,), meta=tagged("rz_target_minus_half")),
        Op("CX", (control, target), meta=tagged("cx2")),
    ]


def build_qft_original_order(n: int) -> Program:
    """swapを除く標準QFTを元の逐次順で構築する。"""
    if n <= 0:
        raise ValueError("n must be positive")
    q = [f"q{i}" for i in range(n)]
    body: list[Stmt] = []
    for j in range(n):
        body.append(Op("H", (q[j],)))
        for k in range(j + 1, n):
            body.extend(_emit_cp(q[k], q[j], math.pi / (2 ** (k - j))))
    return Program(name=f"qft_n{n}", qubits=q, body=body)
