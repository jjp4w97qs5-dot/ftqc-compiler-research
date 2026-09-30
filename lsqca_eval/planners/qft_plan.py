from __future__ import annotations

"""標準QFTのwave順・CR割当・blocked実行計画。"""

import math

from ..execution_plan import cr_load, cr_rotate, cr_store, parallel_groups_begin, parallel_groups_end, phase_barrier
from ..program_ir import Op, Program, Stmt
from ..programs.qft import _emit_cp


def build_qft_diagonal_waves(n: int, *, annotate_parallel: bool = True) -> Program:
    """既存QFT回路を合法な対角wave順で構築する。"""

    if n <= 0:
        raise ValueError("n must be positive")
    q = [f"q{i}" for i in range(n)]
    body: list[Stmt] = []
    for wave in range(2 * n - 1):
        hs: list[int] = []
        if wave % 2 == 0 and wave // 2 < n:
            hs.append(wave // 2)
        cps: list[tuple[int, int]] = []
        lo = max(0, wave - (n - 1))
        hi = min(n - 1, (wave - 1) // 2)
        for i in range(lo, hi + 1):
            j = wave - i
            if 0 <= i < j < n:
                cps.append((i, j))
        groups = [(q[i],) for i in hs] + [(q[i], q[j]) for i, j in cps]
        if annotate_parallel and len(groups) > 1:
            body.append(parallel_groups_begin(groups, name=f"qft_wave_{wave}"))
        for i in hs:
            body.append(Op("H", (q[i],)))
        for i, j in cps:
            body.extend(_emit_cp(q[j], q[i], math.pi / (2 ** (j - i))))
        if annotate_parallel and len(groups) > 1:
            body.append(parallel_groups_end(groups, name=f"qft_wave_{wave}"))
    return Program(name=f"qft_wave_n{n}", qubits=q, body=body)


def build_qft_explicit_cr_waves(n: int, *, cr_count: int = 4) -> Program:
    """対角waveの各groupへ実行CRを明示したQFTを構築する。"""

    if n <= 0:
        raise ValueError("n must be positive")
    if cr_count <= 0:
        raise ValueError("cr_count must be positive")
    q = [f"q{i}" for i in range(n)]
    body: list[Stmt] = []
    for wave in range(2 * n - 1):
        groups: list[tuple[str, ...]] = []
        ops: list[Op | list[Op]] = []
        if wave % 2 == 0 and wave // 2 < n:
            i = wave // 2
            groups.append((q[i],))
            ops.append(Op("H", (q[i],)))
        lo = max(0, wave - (n - 1))
        hi = min(n - 1, (wave - 1) // 2)
        for i in range(lo, hi + 1):
            j = wave - i
            if 0 <= i < j < n:
                groups.append((q[i], q[j]))
                ops.append(_emit_cp(q[j], q[i], math.pi / (2 ** (j - i))))
        if not groups:
            continue
        cr_ids = [index % cr_count for index in range(len(groups))]
        name = f"qft_wave_explicit_{wave}"
        body.append(parallel_groups_begin(groups, name=name, cr_ids=cr_ids))
        for item in ops:
            if isinstance(item, Op):
                body.append(item)
            else:
                body.extend(item)
        body.append(parallel_groups_end(groups, name=name))
        body.append(phase_barrier(q, name=f"qft_wave_barrier_{wave}"))
    return Program(name=f"qft_explicit_cr_waves_n{n}", qubits=q, body=body)


def build_qft_blocked_cr_plan(n: int, *, cr_count: int = 4) -> Program:
    """明示的な配置・移動を含むblocked QFT計画を構築する。"""

    if n <= 0:
        raise ValueError("n must be positive")
    if cr_count <= 0:
        raise ValueError("cr_count must be positive")
    if n % cr_count != 0:
        raise ValueError("current blocked QFT requires n divisible by cr_count")
    q = [f"q{i}" for i in range(n)]
    blocks = [list(range(start, start + cr_count)) for start in range(0, n, cr_count)]
    body: list[Stmt] = []

    def cp_ops(i: int, j: int, **meta: object) -> list[Op]:
        # blocked計画用metadataを共通CP分解へ渡す。
        angle = math.pi / (2 ** (j - i))
        base = {"_plan_no_prefetch": True, "qft_i": i, "qft_j": j, **meta}
        return _emit_cp(q[j], q[i], angle, meta=base)

    def emit_parallel(groups: list[tuple[int, ...]], cr_ids: list[int], ops: list[list[Op] | Op], *, name: str) -> None:
        # 同一roundの演算列を既存順のparallel directiveへ変換する。
        if not groups:
            return
        qgroups = [tuple(q[x] for x in group) for group in groups]
        body.append(parallel_groups_begin(qgroups, name=name, cr_ids=cr_ids))
        for item in ops:
            if isinstance(item, Op):
                body.append(item)
            else:
                body.extend(item)
        body.append(parallel_groups_end(qgroups, name=name))

    lanes = list(range(cr_count))
    for b, target in enumerate(blocks):
        target_qs = [q[x] for x in target]
        body.append(cr_load(target_qs, dst_crs=lanes, barrier_qubits=q, name=f"qft_load_target_b{b}"))

        for a in range(b):
            source = blocks[a]
            source_qs = [q[x] for x in source]
            body.append(
                cr_load(
                    source_qs,
                    dst_crs=lanes,
                    barrier_qubits=q,
                    name=f"qft_load_source_b{a}_for_b{b}",
                )
            )

            for round_id in range(cr_count):
                groups: list[tuple[int, ...]] = []
                cr_ids: list[int] = []
                ops: list[list[Op]] = []
                for lane, j in enumerate(target):
                    source_offset = (lane - round_id) % cr_count
                    i = source[source_offset]
                    groups.append((i, j))
                    cr_ids.append(lane)
                    ops.append(cp_ops(i, j, tile=f"cross_{a}_{b}", round_id=round_id))
                emit_parallel(groups, cr_ids, ops, name=f"qft_cross_b{a}_b{b}_r{round_id}")

                if round_id + 1 < cr_count:
                    dst_crs = [((source_offset + round_id) % cr_count + 1) % cr_count for source_offset in range(cr_count)]
                    body.append(
                        cr_rotate(
                            source_qs,
                            dst_crs=dst_crs,
                            barrier_qubits=q,
                            name=f"qft_rotate_b{a}_b{b}_r{round_id}",
                        )
                    )

            body.append(
                cr_store(
                    source_qs,
                    barrier_qubits=q,
                    name=f"qft_store_source_b{a}_for_b{b}",
                )
            )

        body.append(cr_load(target_qs, dst_crs=lanes, barrier_qubits=q, name=f"qft_restore_target_b{b}"))
        m = len(target)
        for local_wave in range(2 * m - 1):
            groups: list[tuple[int, ...]] = []
            cr_ids: list[int] = []
            ops: list[list[Op] | Op] = []
            if local_wave % 2 == 0:
                r = local_wave // 2
                if r < m:
                    i = target[r]
                    groups.append((i,))
                    cr_ids.append(r)
                    ops.append(Op("H", (q[i],), meta={"_plan_no_prefetch": True, "block": b, "local_wave": local_wave}))
            lo = max(0, local_wave - (m - 1))
            hi = min(m - 1, (local_wave - 1) // 2)
            for r in range(lo, hi + 1):
                sidx = local_wave - r
                if 0 <= r < sidx < m:
                    i, j = target[r], target[sidx]
                    groups.append((i, j))
                    cr_ids.append(r)
                    ops.append(cp_ops(i, j, tile=f"diag_{b}", round_id=local_wave))
            emit_parallel(groups, cr_ids, ops, name=f"qft_diag_b{b}_w{local_wave}")

        body.append(cr_store(target_qs, barrier_qubits=q, name=f"qft_store_target_b{b}"))
        body.append(phase_barrier(q, name=f"qft_block_complete_{b}"))

    return Program(name=f"qft_systolic_existing_model_p{cr_count}_n{n}", qubits=q, body=body)
