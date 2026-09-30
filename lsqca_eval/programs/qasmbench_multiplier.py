from __future__ import annotations

"""QASMBench/Cirq multiplierの15-qubit参照回路と拡大生成器。

The public QASMBench multiplier uses 5w qubits for a w-bit multiplication and
repeats the following w times:

1. prepare a shifted partial-product row controlled by one multiplier bit,
2. add that row into a w-bit accumulator with a reversible ripple-carry adder,
3. uncompute the temporary partial-product row.

This module preserves that structure and provides a legal wavefront schedule
that pipelines partial-product preparation/uncomputation with the carry chain.
15-qubit native gate列だけを固定commitと一対一対応させ、それ以外の規模は
同じmacro規則による派生・拡大回路として扱う。machine modelとlatencyは定義しない。
"""

from dataclasses import dataclass

from ..execution_plan import (
    scope_begin,
    scope_end,
)
from ..program_ir import Op, Program, Stmt
from .cdkm_adder import _emit_ccx


NativeGate = tuple[str, tuple[str, ...]]

# 監査で固定した公式QASMBench参照情報。
QASMBENCH_REFERENCE_COMMIT = "357b942396d5c2b7cbc1c229c585a6ef5ccaebac"
QASMBENCH_N15_QASM_SHA256 = "e8ad585e5be09f96dd60f7495bd649562c6c05f8bf5d8a416b85a9d834e7eab8"


@dataclass(frozen=True)
class Macro:
    """A high-level operation with a fixed qubit footprint."""

    name: str
    kind: str
    qubits: tuple[str, ...]
    native_ops: tuple[NativeGate, ...]
    meta: dict[str, object]


def _registers(width: int) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Return QASMBench's 5w-qubit register layout.

    The first 3w qubits are interleaved per bit as (carry, addend, accumulator),
    followed by the multiplicand and multiplier registers.  This matches the
    public multiplier_n15 circuit: q[3i], q[3i+1], q[3i+2], q[3w+i], q[4w+i].
    """
    if width <= 0:
        raise ValueError("width must be positive")
    carry = [f"q{3 * i}" for i in range(width)]
    addend = [f"q{3 * i + 1}" for i in range(width)]
    accum = [f"q{3 * i + 2}" for i in range(width)]
    multiplicand = [f"q{3 * width + i}" for i in range(width)]
    multiplier = [f"q{4 * width + i}" for i in range(width)]
    qubits = [f"q{i}" for i in range(5 * width)]
    return qubits, carry, addend, accum, multiplicand, multiplier


def _ccx_native(c0: str, c1: str, target: str) -> NativeGate:
    return ("CCX", (c0, c1, target))


def _cx_native(control: str, target: str) -> NativeGate:
    return ("CX", (control, target))


def _partial_macro(
    multiplier_q: str,
    multiplicand_q: str,
    addend_q: str,
    *,
    outer_index: int,
    bit_index: int,
    action: str,
) -> Macro:
    return Macro(
        name=f"partial_{action}_j{outer_index}_k{bit_index}",
        kind=f"partial_{action}",
        qubits=(multiplier_q, multiplicand_q, addend_q),
        native_ops=(_ccx_native(multiplier_q, multiplicand_q, addend_q),),
        meta={
            "multiplier_phase": f"partial_{action}",
            "outer_index": outer_index,
            "bit_index": bit_index,
            "shift": outer_index,
        },
    )


def _forward_macro(carry: list[str], addend: list[str], accum: list[str], i: int, *, outer_index: int) -> Macro:
    local = (carry[i], addend[i], accum[i], carry[i + 1])
    return Macro(
        name=f"adder_forward_j{outer_index}_i{i}",
        kind="adder_forward",
        qubits=local,
        native_ops=(
            _ccx_native(addend[i], accum[i], carry[i + 1]),
            _cx_native(addend[i], accum[i]),
            _ccx_native(carry[i], accum[i], carry[i + 1]),
        ),
        meta={
            "multiplier_phase": "adder_forward",
            "outer_index": outer_index,
            "bit_index": i,
        },
    )


def _top_macro(carry: list[str], addend: list[str], accum: list[str], *, outer_index: int) -> Macro:
    i = len(addend) - 1
    local = (carry[i], addend[i], accum[i])
    return Macro(
        name=f"adder_top_j{outer_index}",
        kind="adder_top",
        qubits=local,
        native_ops=(
            _cx_native(addend[i], accum[i]),
            _cx_native(carry[i], accum[i]),
        ),
        meta={
            "multiplier_phase": "adder_top",
            "outer_index": outer_index,
            "bit_index": i,
        },
    )


def _backward_macro(carry: list[str], addend: list[str], accum: list[str], i: int, *, outer_index: int) -> Macro:
    local = (carry[i], addend[i], accum[i], carry[i + 1])
    return Macro(
        name=f"adder_backward_j{outer_index}_i{i}",
        kind="adder_backward",
        qubits=local,
        native_ops=(
            _ccx_native(carry[i], accum[i], carry[i + 1]),
            _cx_native(addend[i], accum[i]),
            _ccx_native(addend[i], accum[i], carry[i + 1]),
            _cx_native(addend[i], accum[i]),
            _cx_native(carry[i], accum[i]),
        ),
        meta={
            "multiplier_phase": "adder_backward",
            "outer_index": outer_index,
            "bit_index": i,
        },
    )


def _lower_macro(macro: Macro, *, schedule_wave: int | None = None, wave_kind: str | None = None) -> list[Op]:
    """Lower one macro using the repository's existing CCX decomposition."""
    result: list[Op] = []
    native_index = 0
    subop_index = 0
    extra = dict(macro.meta)
    extra.update({"macro_name": macro.name, "macro_kind": macro.kind})
    if schedule_wave is not None:
        extra["schedule_wave"] = schedule_wave
    if wave_kind is not None:
        extra["wave_kind"] = wave_kind
    for gate, qubits in macro.native_ops:
        if gate == "CCX":
            lowered = _emit_ccx(*qubits)
        elif gate == "CX":
            lowered = [Op("CX", qubits)]
        else:
            raise ValueError(f"Unsupported native gate: {gate}")
        for op in lowered:
            result.append(
                Op(
                    op.gate,
                    op.qubits,
                    op.params,
                    meta={
                        **extra,
                        "native_gate": gate,
                        "native_gate_index": native_index,
                        "macro_subop": subop_index,
                    },
                )
            )
            subop_index += 1
        native_index += 1
    return result


def _iteration_macros(width: int, outer_index: int) -> tuple[
    dict[int, Macro], list[Macro], list[Macro], dict[int, Macro]
]:
    """Build all macros of one controlled-add iteration."""
    _, carry, addend, accum, multiplicand, multiplier = _registers(width)
    prepare = {
        k: _partial_macro(
            multiplier[outer_index],
            multiplicand[k - outer_index],
            addend[k],
            outer_index=outer_index,
            bit_index=k,
            action="prepare",
        )
        for k in range(outer_index, width)
    }
    forward = [
        _forward_macro(carry, addend, accum, i, outer_index=outer_index)
        for i in range(width - 1)
    ]
    forward.append(_top_macro(carry, addend, accum, outer_index=outer_index))
    backward = [
        _backward_macro(carry, addend, accum, i, outer_index=outer_index)
        for i in reversed(range(width - 1))
    ]
    uncompute = {
        k: _partial_macro(
            multiplier[outer_index],
            multiplicand[k - outer_index],
            addend[k],
            outer_index=outer_index,
            bit_index=k,
            action="uncompute",
        )
        for k in range(outer_index, width)
    }
    return prepare, forward, backward, uncompute


def qasm_order_macros(width: int) -> list[Macro]:
    """15-qubit参照回路から抽出した規則でmacro列を生成する。"""
    result: list[Macro] = []
    for j in range(width):
        prepare, forward, backward, uncompute = _iteration_macros(width, j)
        result.extend(prepare[k] for k in range(j, width))
        result.extend(forward)
        result.extend(backward)
        result.extend(uncompute[k] for k in range(j, width))
    return result


def _append_scoped_macro(body: list[Stmt], macro: Macro, *, annotate_scope: bool) -> None:
    if annotate_scope:
        body.append(scope_begin(macro.qubits, name=macro.name))
    body.extend(_lower_macro(macro))
    if annotate_scope:
        body.append(scope_end(macro.qubits, name=macro.name))


def build_qasmbench_multiplier(
    total_qubits: int,
    *,
    annotate_scopes: bool = False,
) -> Program:
    """15-qubit参照規則に基づくmultiplier kernelを構築する。

    The benchmark kernel requires total_qubits = 5 * width.  Input preparation
    and measurements are intentionally excluded from performance comparison;
    the multiply unit itself is identical across all compared schedules.
    """
    if total_qubits <= 0 or total_qubits % 5 != 0:
        raise ValueError("QASMBench multiplier requires total_qubits = 5 * width")
    width = total_qubits // 5
    qubits, *_ = _registers(width)
    body: list[Stmt] = []
    for macro in qasm_order_macros(width):
        _append_scoped_macro(body, macro, annotate_scope=annotate_scopes)
    suffix = "scope" if annotate_scopes else "original"
    return Program(
        name=f"qasmbench_multiplier_{suffix}_w{width}_n{total_qubits}",
        qubits=qubits,
        body=body,
    )


def multiplier_provenance(total_qubits: int) -> dict[str, str | int]:
    """規模ごとの参照対応範囲を機械可読な辞書で返す。"""

    if total_qubits <= 0 or total_qubits % 5 != 0:
        raise ValueError("QASMBench multiplier requires total_qubits = 5 * width")
    return {
        "total_qubits": total_qubits,
        "official_commit": QASMBENCH_REFERENCE_COMMIT,
        "reference_qasm_sha256": QASMBENCH_N15_QASM_SHA256,
        "provenance_class": (
            "exact_native_gate_sequence" if total_qubits == 15 else "derived_scaled_circuit"
        ),
    }


def native_gate_counts(width: int) -> dict[str, int]:
    """Closed-form gate counts of the multiplier unit before CCX decomposition."""
    ccx = 5 * width * width - 3 * width
    cx = 4 * width * width - 2 * width
    return {"CCX": ccx, "CX": cx, "total": ccx + cx}


def decomposed_gate_counts(width: int) -> dict[str, int]:
    """Closed-form counts under the repository's 15-gate CCX decomposition."""
    native = native_gate_counts(width)
    ccx = native["CCX"]
    direct_cx = native["CX"]
    result = {
        "H": 2 * ccx,
        "CX": 6 * ccx + direct_cx,
        "T": 4 * ccx,
        "TDG": 3 * ccx,
    }
    result["total"] = sum(result.values())
    return result
