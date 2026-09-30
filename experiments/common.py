from __future__ import annotations

"""複数の実験driverで共有する実行・集計・CSV出力処理。"""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from lsqca_eval.program_ir import Program


@dataclass(frozen=True)
class ProgramMethod:
    """qubit数から評価対象Programを構築する実験手法。"""

    name: str
    build: Callable[[int], Program]


@dataclass(frozen=True)
class ExperimentCase:
    """共通case runnerへ渡す1実験ケース。"""

    labels: dict[str, object]
    display: tuple[object, ...]
    run: Callable[[], dict[str, object]]


def parallel_stats(trace: list, total_beats: int) -> dict[str, float | int]:
    """CR上のgate実行区間から既存の並列度指標を計算する。"""

    ops = [op for op in trace if op.cr_id is not None and op.reason in {"cr_gate", "single_in_cr"} and op.end > op.start]
    events: dict[int, list[list[int]]] = {}
    for index, op in enumerate(ops):
        events.setdefault(op.start, [[], []])[0].append(index)
        events.setdefault(op.end, [[], []])[1].append(index)
    active: set[int] = set()
    previous: int | None = None
    distribution: Counter[int] = Counter()
    for beat in sorted(events):
        if previous is not None and beat > previous:
            distribution[len(active)] += beat - previous
        for index in events[beat][1]:
            active.discard(index)
        for index in events[beat][0]:
            active.add(index)
        previous = beat
    denom = max(1, total_beats)
    return {
        "avg_active_cr": round(sum(count * beats for count, beats in distribution.items()) / denom, 6),
        "parallel_fraction": round(sum(beats for count, beats in distribution.items() if count >= 2) / denom, 6),
        "full4_fraction": round(distribution[4] / denom, 6),
        "max_active_cr": max(distribution, default=0),
    }


def run_experiment_cases(cases: Iterable[ExperimentCase]) -> list[dict[str, object]]:
    """既存driverと同じ成功・失敗行および標準出力を生成する。"""

    rows: list[dict[str, object]] = []
    for case in cases:
        try:
            row = case.run()
            rows.append(row)
            print("OK", *case.display, row["total_beats"], flush=True)
        except Exception as exc:
            rows.append({**case.labels, "error": f"{type(exc).__name__}: {exc}"})
            print("ERR", *case.display, exc, flush=True)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """全行で最初に現れた順に列を並べてCSVを出力する。"""

    normalized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in normalized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)
