from __future__ import annotations

"""QASMBench multiplierの依存関係wavefrontと明示CR計画。"""

from ..execution_plan import parallel_groups_begin, parallel_groups_end
from ..program_ir import Program, Stmt
from ..programs.qasmbench_multiplier import Macro, _iteration_macros, _lower_macro, _registers


def iteration_wavefronts(width: int, outer_index: int) -> list[tuple[str, int, list[Macro]]]:
    """1回のcontrolled-addを既存ASAP依存式でmacro waveへ分割する。"""

    prepare, forward, backward_desc, uncompute = _iteration_macros(width, outer_index)
    waves: list[tuple[str, int, list[Macro]]] = []

    # preparation chainとforward/top adder chainのlevelを決める。
    p_level = {k: k - outer_index for k in range(outer_index, width)}
    a_level: dict[int, int] = {}
    for i in range(width):
        previous = a_level[i - 1] if i > 0 else -1
        prepared = p_level[i] if i >= outer_index else -1
        a_level[i] = max(previous, prepared) + 1
    forward_levels: dict[int, list[Macro]] = {}
    for k, level in p_level.items():
        forward_levels.setdefault(level, []).append(prepare[k])
    for i, level in a_level.items():
        forward_levels.setdefault(level, []).append(forward[i])
    for level in sorted(forward_levels):
        macros = sorted(
            forward_levels[level],
            key=lambda m: (0 if m.kind.startswith("adder_") else 1, m.name),
        )
        waves.append(("forward", level, macros))

    # carry restoration chainとuncompute chainのlevelを決める。
    u_level = {i: (width - 2) - i for i in range(width - 1)}
    q_level: dict[int, int] = {width - 1: 0}
    for k in range(width - 2, outer_index - 1, -1):
        q_level[k] = max(q_level[k + 1], u_level[k]) + 1
    backward_levels: dict[int, list[Macro]] = {}
    for macro in backward_desc:
        i = int(macro.meta["bit_index"])
        backward_levels.setdefault(u_level[i], []).append(macro)
    for k, level in q_level.items():
        backward_levels.setdefault(level, []).append(uncompute[k])
    for level in sorted(backward_levels):
        macros = sorted(
            backward_levels[level],
            key=lambda m: (0 if m.kind == "adder_backward" else 1, m.name),
        )
        waves.append(("backward", level, macros))

    return waves


def wave_order_macros(width: int) -> list[tuple[str, int, int, list[Macro]]]:
    """全outer iterationのwavefrontを元のouter-loop順で返す。"""

    result: list[tuple[str, int, int, list[Macro]]] = []
    for j in range(width):
        for wave_kind, level, macros in iteration_wavefronts(width, j):
            result.append((wave_kind, j, level, macros))
    return result


def build_qasmbench_multiplier_wave(
    total_qubits: int,
    *,
    annotate_parallel: bool = True,
    explicit_crs: bool = False,
    cr_count: int = 4,
) -> Program:
    """元と同一の乗算をwavefront順と任意の明示CR割当で構築する。"""

    if total_qubits <= 0 or total_qubits % 5 != 0:
        raise ValueError("QASMBench multiplier requires total_qubits = 5 * width")
    if cr_count <= 0:
        raise ValueError("cr_count must be positive")
    width = total_qubits // 5
    qubits, *_ = _registers(width)
    body: list[Stmt] = []
    global_wave = 0
    for wave_kind, outer_index, level, macros in wave_order_macros(width):
        if len(macros) > cr_count:
            raise AssertionError("Derived wave exceeds available CR count")
        groups = [macro.qubits for macro in macros]
        use_directive = annotate_parallel and bool(groups)
        name = f"qasmbench_mul_j{outer_index}_{wave_kind}_level{level}"
        if use_directive:
            cr_ids = list(range(len(groups))) if explicit_crs else None
            body.append(parallel_groups_begin(groups, name=name, cr_ids=cr_ids))
        for macro in macros:
            body.extend(
                _lower_macro(
                    macro,
                    schedule_wave=global_wave,
                    wave_kind=wave_kind,
                )
            )
        if use_directive:
            body.append(parallel_groups_end(groups, name=name))
        global_wave += 1
    suffix = "explicit" if explicit_crs else ("groups" if annotate_parallel else "order")
    return Program(
        name=f"qasmbench_multiplier_wave_{suffix}_w{width}_n{total_qubits}",
        qubits=qubits,
        body=body,
    )
