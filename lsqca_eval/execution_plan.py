from __future__ import annotations

"""高水準plannerがschedulerへ渡す構造annotation。"""

from .program_ir import PlanDirective


def scope_begin(qubits: list[str] | tuple[str, ...], *, name: str | None = None) -> PlanDirective:
    """指定qubit集合を同一CRへ寄せるsoft affinity scopeを開始する。"""
    return PlanDirective("SCOPE_BEGIN", tuple(qubits), policy="affinity", meta={} if name is None else {"name": name})


def scope_end(qubits: list[str] | tuple[str, ...], *, name: str | None = None) -> PlanDirective:
    """scope_beginで設定したsoft affinityを終了する。"""
    return PlanDirective("SCOPE_END", tuple(qubits), policy="release", meta={} if name is None else {"name": name})


def parallel_groups_begin(
    groups: list[list[str] | tuple[str, ...]],
    *,
    name: str | None = None,
    cr_ids: list[int] | tuple[int, ...] | None = None,
    planned_crs: list[int] | tuple[int, ...] | None = None,
) -> PlanDirective:
    """互いに独立なgroup集合を宣言する。

    cr_ids は明示的CR割当、planned_crs はplannerが決めたCR割当を表す。
    いずれも指定しない場合はschedulerのgroup assignment policyで選ぶ。
    """
    normalized = tuple(tuple(group) for group in groups)
    meta: dict[str, object] = {} if name is None else {"name": name}
    if cr_ids is not None:
        if len(cr_ids) != len(normalized):
            raise ValueError("cr_ids must have the same length as groups")
        meta["cr_ids"] = tuple(int(c) for c in cr_ids)
    if planned_crs is not None:
        if len(planned_crs) != len(normalized):
            raise ValueError("planned_crs must have the same length as groups")
        meta["planned_crs"] = tuple(int(c) for c in planned_crs)
    return PlanDirective("PARALLEL_GROUPS_BEGIN", groups=normalized, policy="parallel_groups", meta=meta)


def parallel_groups_end(
    groups: list[list[str] | tuple[str, ...]], *, name: str | None = None
) -> PlanDirective:
    """parallel_groups_beginのgroup scopeを終了する。"""
    return PlanDirective(
        "PARALLEL_GROUPS_END",
        groups=tuple(tuple(group) for group in groups),
        policy="release_parallel_groups",
        meta={} if name is None else {"name": name},
    )


def phase_barrier(qubits: list[str] | tuple[str, ...], *, name: str | None = None) -> PlanDirective:
    """高水準phase順序を保存する0-cost dependency barrier。"""
    return PlanDirective("PHASE_BARRIER", tuple(qubits), policy="dependency_barrier", meta={} if name is None else {"name": name})


def cr_rotate(
    qubits: list[str] | tuple[str, ...],
    *,
    dst_crs: list[int] | tuple[int, ...],
    barrier_qubits: list[str] | tuple[str, ...] = (),
    name: str | None = None,
) -> PlanDirective:
    """CR resident qubit群の一斉CR間移動を宣言する。"""
    if len(qubits) != len(dst_crs):
        raise ValueError("dst_crs must have the same length as qubits")
    meta: dict[str, object] = {"dst_crs": tuple(int(c) for c in dst_crs)}
    if name is not None:
        meta["name"] = name
    all_qubits = tuple(dict.fromkeys((*tuple(qubits), *tuple(barrier_qubits))))
    return PlanDirective("CR_ROTATE", all_qubits, groups=(tuple(qubits),), policy="collective_migration", meta=meta)


def cr_load(
    qubits: list[str] | tuple[str, ...],
    *,
    dst_crs: list[int] | tuple[int, ...],
    barrier_qubits: list[str] | tuple[str, ...] = (),
    name: str | None = None,
) -> PlanDirective:
    """指定qubitを指定CRへ明示的にload/migrateする高水準計画。"""
    if len(qubits) != len(dst_crs):
        raise ValueError("dst_crs must have the same length as qubits")
    meta: dict[str, object] = {"dst_crs": tuple(int(c) for c in dst_crs)}
    if name is not None:
        meta["name"] = name
    all_qubits = tuple(dict.fromkeys((*tuple(qubits), *tuple(barrier_qubits))))
    return PlanDirective("CR_LOAD", all_qubits, groups=(tuple(qubits),), policy="explicit_load", meta=meta)


def cr_store(
    qubits: list[str] | tuple[str, ...],
    *,
    barrier_qubits: list[str] | tuple[str, ...] = (),
    name: str | None = None,
) -> PlanDirective:
    """指定qubitをSAMへ明示的にstoreする高水準計画。"""
    meta: dict[str, object] = {} if name is None else {"name": name}
    all_qubits = tuple(dict.fromkeys((*tuple(qubits), *tuple(barrier_qubits))))
    return PlanDirective("CR_STORE", all_qubits, groups=(tuple(qubits),), policy="explicit_store", meta=meta)
