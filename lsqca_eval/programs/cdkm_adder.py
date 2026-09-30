from __future__ import annotations

"""Cuccaro/CDKM ripple-carry adder用の構造付きProgramIR builder。"""

from ..program_ir import Op, Program, Stmt


def _emit_ccx(c0: str, c1: str, target: str) -> list[Op]:
    # Toffoliを既存評価器の基本ゲート列へ展開する。
    return [
        Op("H", (target,)),
        Op("CX", (c1, target)),
        Op("TDG", (target,)),
        Op("CX", (c0, target)),
        Op("T", (target,)),
        Op("CX", (c1, target)),
        Op("TDG", (target,)),
        Op("CX", (c0, target)),
        Op("T", (c1,)),
        Op("T", (target,)),
        Op("H", (target,)),
        Op("CX", (c0, c1)),
        Op("T", (c0,)),
        Op("TDG", (c1,)),
        Op("CX", (c0, c1)),
    ]


def emit_maj(carry: str, a: str, b: str) -> list[Op]:
    # Cuccaro adderのMAJ macroを基本ゲートへ展開する。
    return [
        Op("CX", (a, b)),
        Op("CX", (a, carry)),
        *_emit_ccx(carry, b, a),
    ]


def emit_uma(carry: str, a: str, b: str) -> list[Op]:
    # Cuccaro adderのUMA macroを基本ゲートへ展開する。
    return [
        *_emit_ccx(carry, b, a),
        Op("CX", (a, carry)),
        Op("CX", (carry, b)),
    ]


def _annotate_bit_iteration(
    ops: list[Op],
    *,
    phase: str,
    bit_index: int,
    carry_in_storage: str,
    bit_operand: str,
    carry_out_storage: str,
) -> list[Op]:
    """Attach the semantic Cuccaro bit-iteration without changing any gate.

    In the Cuccaro carry chain, define ``chain_a[0] = cin`` and
    ``chain_a[i + 1] = a[i]``.  Bit iteration ``i`` therefore touches the
    semantic triple ``(chain_a[i], b[i], chain_a[i + 1])``.  This is distinct
    from assigning an iteration id to every decomposed Clifford+T statement.
    """
    triplet = (carry_in_storage, bit_operand, carry_out_storage)
    return [
        Op(
            op.gate,
            op.qubits,
            op.params,
            {
                **op.meta,
                "phase": phase,
                "source_loop": "cdkm_bit_carry_chain",
                "loop_iter": bit_index,
                "bit_index": bit_index,
                "term": term,
                "macro": "MAJ" if phase == "cdkm_forward" else "UMA",
                "semantic_iteration_id": True,
                "iteration_triplet": triplet,
                "carry_in_storage": carry_in_storage,
                "bit_operand": bit_operand,
                "carry_out_storage": carry_out_storage,
                # Consecutive bit iterations share the carry-storage qubit.
                "loop_carried_dependence_distance": 1,
            },
        )
        for term, op in enumerate(ops)
    ]


def build_cdkm_adder(n: int) -> Program:
    """n bit CDKM/Cuccaro adderを構築する。"""
    if n <= 0:
        raise ValueError("n must be positive")
    carry = "cin"
    a = [f"a{i}" for i in range(n)]
    b = [f"b{i}" for i in range(n)]
    cout = "cout"
    qubits = [carry, *a, *b, cout]
    body: list[Stmt] = []

    for i in range(n):
        carry_in_storage = carry if i == 0 else a[i - 1]
        carry_out_storage = a[i]
        local = (carry_in_storage, carry_out_storage, b[i])
        body.extend(_annotate_bit_iteration(
            emit_maj(*local),
            phase="cdkm_forward",
            bit_index=i,
            carry_in_storage=carry_in_storage,
            bit_operand=b[i],
            carry_out_storage=carry_out_storage,
        ))

    body.append(Op("CX", (a[-1], cout), meta={
        "phase": "cdkm_carry_out",
        "source_loop": "cdkm_bit_carry_chain",
        "bit_index": n - 1,
        "carry_out_storage": a[-1],
        "output_qubit": cout,
    }))

    for i in reversed(range(n)):
        carry_in_storage = carry if i == 0 else a[i - 1]
        carry_out_storage = a[i]
        local = (carry_in_storage, carry_out_storage, b[i])
        body.extend(_annotate_bit_iteration(
            emit_uma(*local),
            phase="cdkm_reverse",
            bit_index=i,
            carry_in_storage=carry_in_storage,
            bit_operand=b[i],
            carry_out_storage=carry_out_storage,
        ))

    return Program(name=f"cdkm_adder_n{n}", qubits=qubits, body=body)
