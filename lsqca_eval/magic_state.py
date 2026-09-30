from __future__ import annotations

"""各CRに固定されたmagic-state factoryと有限bufferを追跡する。"""

from dataclasses import dataclass, replace
import math
from typing import Literal


# Magic-state供給を無制限または有限bufferとして扱う実行mode。
MagicStateMode = Literal["abundant", "finite"]


class MagicStateModelError(ValueError):
    """Magic-state供給設定または実行順が不正な場合のerror。"""


@dataclass(frozen=True)
class MagicStateConfig:
    """各CRに共通するMSF生成時間とbuffer条件を保持する。"""

    # Magic-state供給を無制限または有限として切り替える。
    mode: MagicStateMode = "abundant"
    # 1個のmagic stateを生成するbeat数。
    generation_beats: int = 15
    # 各CRが持つlocal bufferの最大状態数。
    buffer_capacity: int = 2
    # beat 0で各CR bufferに存在する状態数。
    initial_buffer_occupancy: int = 0

    def __post_init__(self) -> None:
        """再現不能な供給設定を構築時に拒否する。"""

        if self.mode not in {"abundant", "finite"}:
            raise MagicStateModelError(f"Unknown magic-state mode: {self.mode}")
        if self.generation_beats < 1:
            raise MagicStateModelError("generation_beats must be at least 1")
        if self.buffer_capacity < 1:
            raise MagicStateModelError("buffer_capacity must be at least 1")
        if not 0 <= self.initial_buffer_occupancy <= self.buffer_capacity:
            raise MagicStateModelError(
                "initial_buffer_occupancy must be between 0 and buffer_capacity"
            )


@dataclass(frozen=True)
class MagicStateReservation:
    """1 gateによるlocal magic-state消費と供給待ちの集計結果。"""

    # Gateが逐次消費したmagic-state数。
    state_count: int
    # 本来の注入間隔に加えて生じた供給待ちbeat数。
    wait_beats: int
    # 最初と最後のmagic-state消費時刻。
    first_consumption_time: int | None
    last_consumption_time: int | None
    # Gate要求時と最後の消費直後のbuffer量。
    buffer_before: int
    buffer_after: int
    # Gate要求時にbufferが空だったかを示す。
    buffer_empty_at_request: bool
    # 各stateについてdesired時刻から実供給時刻まで待った半開区間。
    # aggregate wait_beatsと同値だが、stall overlap解析用に時間軸を保持する。
    wait_intervals: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class MagicStateCRSnapshot:
    """将来のCR選択最適化が参照できるlocal MSFの非破壊snapshot。"""

    # Snapshotを取得したCRと評価時刻。
    cr_id: int
    time: int
    # 評価時刻に見込まれるbuffer量と次の完成時刻。
    buffer_occupancy: int
    next_completion_time: int | None
    # 評価時刻までに見込まれる生成・既消費数。
    generated_count: int
    consumed_count: int


@dataclass
class _CRMagicState:
    """1個のCRに固定されたMSFとbufferの可変実行状態。"""

    # 生成・消費状態を反映済みの最新時刻。
    time: int
    # 現在bufferに保存されているmagic-state数。
    buffer_occupancy: int
    # 次の状態が完成する時刻。buffer満杯による停止中はNone。
    next_completion_time: int | None
    # 実行中に完成・消費したmagic-state総数。
    generated_count: int = 0
    consumed_count: int = 0
    # 観測された最大buffer量と供給待ち総数。
    peak_buffer_occupancy: int = 0
    wait_beats: int = 0


class MagicStateRuntime:
    """各CRの独立MSFを前向きに更新し、gate要求へ状態を供給する。"""

    def __init__(self, config: MagicStateConfig, cr_count: int) -> None:
        """同一設定のlocal MSFを全CRへ1基ずつ構築する。"""

        if cr_count < 1:
            raise MagicStateModelError("cr_count must be at least 1")
        self.config = config
        self._cr_states: list[_CRMagicState] = []
        for _ in range(cr_count):
            if config.mode == "finite":
                occupancy = config.initial_buffer_occupancy
                next_completion = (
                    config.generation_beats
                    if occupancy < config.buffer_capacity
                    else None
                )
            else:
                # abundant modeでは物理bufferを使わず需要数だけを追跡する。
                occupancy = 0
                next_completion = None
            self._cr_states.append(_CRMagicState(
                time=0,
                buffer_occupancy=occupancy,
                next_completion_time=next_completion,
                peak_buffer_occupancy=occupancy,
            ))

        # Gate単位の需要・待ち統計。
        self.request_gate_count = 0
        self.required_count = 0
        self.wait_gate_count = 0
        self.buffer_empty_gate_count = 0
        self._gate_waits: list[int] = []
        # 直近consume_for_gateの結果。観測用でschedule判断には使用しない。
        self.last_reservation: MagicStateReservation | None = None

    @property
    def cr_count(self) -> int:
        """Local MSFを持つCR数を返す。"""

        return len(self._cr_states)

    def _state(self, cr_id: int) -> _CRMagicState:
        """範囲検証済みの指定CR供給状態を返す。"""

        if not 0 <= cr_id < self.cr_count:
            raise MagicStateModelError(f"Invalid CR for magic state: {cr_id}")
        return self._cr_states[cr_id]

    def _assert_conservation(self, state: _CRMagicState) -> None:
        """有限bufferの生成・消費・在庫保存則を検査する。"""

        if self.config.mode != "finite":
            return
        expected = self.config.initial_buffer_occupancy + state.generated_count
        actual = state.consumed_count + state.buffer_occupancy
        if expected != actual:
            raise MagicStateModelError(
                f"Magic-state conservation failed: expected={expected}, actual={actual}"
            )
        if not 0 <= state.buffer_occupancy <= self.config.buffer_capacity:
            raise MagicStateModelError(
                f"Magic-state buffer is out of range: {state.buffer_occupancy}"
            )

    def _advance_finite(self, state: _CRMagicState, target_time: int) -> None:
        """指定時刻までのMSF完成分を有限bufferへ反映する。"""

        if target_time < state.time:
            raise MagicStateModelError(
                f"Magic-state runtime cannot move backward: "
                f"{target_time} < {state.time}"
            )
        while (
            state.next_completion_time is not None
            and state.next_completion_time <= target_time
        ):
            if state.buffer_occupancy >= self.config.buffer_capacity:
                raise MagicStateModelError("MSF completed while its buffer was full")
            completion = state.next_completion_time
            state.buffer_occupancy += 1
            state.generated_count += 1
            state.peak_buffer_occupancy = max(
                state.peak_buffer_occupancy,
                state.buffer_occupancy,
            )
            state.next_completion_time = (
                None
                if state.buffer_occupancy == self.config.buffer_capacity
                else completion + self.config.generation_beats
            )
        state.time = target_time
        self._assert_conservation(state)

    def _consume_one_finite(self, state: _CRMagicState, desired_time: int) -> int:
        """1個の状態をdesired以降の最短時刻にlocal bufferから消費する。"""

        self._advance_finite(state, desired_time)
        actual_time = desired_time
        if state.buffer_occupancy == 0:
            if state.next_completion_time is None:
                raise MagicStateModelError("Empty buffer has no active MSF production")
            actual_time = state.next_completion_time
            self._advance_finite(state, actual_time)
        if state.buffer_occupancy <= 0:
            raise MagicStateModelError("Magic-state production did not satisfy demand")

        state.buffer_occupancy -= 1
        state.consumed_count += 1
        if state.next_completion_time is None:
            # 満杯で停止していたMSFは、消費時点から新しい生成を開始する。
            state.next_completion_time = actual_time + self.config.generation_beats
        self._assert_conservation(state)
        return actual_time

    def snapshot_at(self, cr_id: int, time: int) -> MagicStateCRSnapshot:
        """Runtimeを変更せず、指定時刻のlocal buffer状態を投影する。"""

        if time < 0:
            raise MagicStateModelError("snapshot time must not be negative")
        state = self._state(cr_id)
        if time < state.time:
            raise MagicStateModelError(
                f"Snapshot precedes the current state for CR {cr_id}: "
                f"{time} < {state.time}"
            )
        projected = replace(state)
        if self.config.mode == "finite":
            self._advance_finite(projected, time)
        return MagicStateCRSnapshot(
            cr_id=cr_id,
            time=time,
            buffer_occupancy=projected.buffer_occupancy,
            next_completion_time=projected.next_completion_time,
            generated_count=projected.generated_count,
            consumed_count=projected.consumed_count,
        )

    def consume_for_gate(
        self,
        cr_id: int,
        earliest: int,
        state_count: int,
        injection_spacing: int = 3,
    ) -> MagicStateReservation:
        """指定CRでgate用状態を逐次消費し、追加待ち時間を返す。"""

        if earliest < 0:
            raise MagicStateModelError("earliest must not be negative")
        if state_count < 0:
            raise MagicStateModelError("state_count must not be negative")
        if injection_spacing < 1:
            raise MagicStateModelError("injection_spacing must be at least 1")
        state = self._state(cr_id)
        if earliest < state.time:
            raise MagicStateModelError(
                f"Magic-state request precedes the current state for CR {cr_id}: "
                f"{earliest} < {state.time}"
            )
        if state_count == 0:
            reservation = MagicStateReservation(
                state_count=0,
                wait_beats=0,
                first_consumption_time=None,
                last_consumption_time=None,
                buffer_before=state.buffer_occupancy,
                buffer_after=state.buffer_occupancy,
                buffer_empty_at_request=False,
                wait_intervals=(),
            )
            self.last_reservation = reservation
            return reservation

        self.request_gate_count += 1
        self.required_count += state_count

        if self.config.mode == "abundant":
            state.consumed_count += state_count
            state.time = earliest + (state_count - 1) * injection_spacing
            self._gate_waits.append(0)
            reservation = MagicStateReservation(
                state_count=state_count,
                wait_beats=0,
                first_consumption_time=earliest,
                last_consumption_time=earliest + (state_count - 1) * injection_spacing,
                buffer_before=0,
                buffer_after=0,
                buffer_empty_at_request=False,
                wait_intervals=(),
            )
            self.last_reservation = reservation
            return reservation

        self._advance_finite(state, earliest)
        buffer_before = state.buffer_occupancy
        buffer_empty = buffer_before == 0
        first_consumption: int | None = None
        last_consumption: int | None = None
        wait_beats = 0
        wait_intervals: list[tuple[int, int]] = []
        desired_time = earliest
        for _ in range(state_count):
            actual_time = self._consume_one_finite(state, desired_time)
            if first_consumption is None:
                first_consumption = actual_time
            last_consumption = actual_time
            if actual_time > desired_time:
                wait_intervals.append((desired_time, actual_time))
            wait_beats += actual_time - desired_time
            desired_time = actual_time + injection_spacing

        state.wait_beats += wait_beats
        self._gate_waits.append(wait_beats)
        if wait_beats:
            self.wait_gate_count += 1
        if buffer_empty:
            self.buffer_empty_gate_count += 1
        reservation = MagicStateReservation(
            state_count=state_count,
            wait_beats=wait_beats,
            first_consumption_time=first_consumption,
            last_consumption_time=last_consumption,
            buffer_before=buffer_before,
            buffer_after=state.buffer_occupancy,
            buffer_empty_at_request=buffer_empty,
            wait_intervals=tuple(wait_intervals),
        )
        self.last_reservation = reservation
        return reservation

    @staticmethod
    def _nearest_rank(values: list[int], percentile: float) -> int:
        """整数待ち時間列のnearest-rank percentileを返す。"""

        if not values:
            return 0
        ordered = sorted(values)
        rank = max(1, math.ceil(percentile * len(ordered)))
        return ordered[rank - 1]

    def metrics(self, total_beats: int) -> dict[str, object]:
        """実行終了時までMSFを進め、共通magic-state指標を返す。"""

        if total_beats < 0:
            raise MagicStateModelError("total_beats must not be negative")
        if self.config.mode == "finite":
            for state in self._cr_states:
                if total_beats < state.time:
                    raise MagicStateModelError(
                        "total_beats precedes the current magic-state runtime"
                    )
                self._advance_finite(state, total_beats)

        generated_by_cr = tuple(state.generated_count for state in self._cr_states)
        consumed_by_cr = tuple(state.consumed_count for state in self._cr_states)
        wait_by_cr = tuple(state.wait_beats for state in self._cr_states)
        peak_by_cr = tuple(state.peak_buffer_occupancy for state in self._cr_states)
        final_by_cr = tuple(state.buffer_occupancy for state in self._cr_states)
        wait_total = sum(self._gate_waits)
        return {
            "magic_state_mode": self.config.mode,
            "magic_state_generation_beats": self.config.generation_beats,
            "magic_state_buffer_capacity_per_cr": self.config.buffer_capacity,
            "magic_state_initial_buffer_per_cr": self.config.initial_buffer_occupancy,
            "magic_state_request_gate_count": self.request_gate_count,
            "magic_state_required_count": self.required_count,
            "magic_state_generated_count": sum(generated_by_cr),
            "magic_state_consumed_count": sum(consumed_by_cr),
            "magic_state_wait_beats": wait_total,
            "magic_state_wait_gate_count": self.wait_gate_count,
            "magic_state_buffer_empty_gate_count": self.buffer_empty_gate_count,
            "magic_state_wait_max_beats": max(self._gate_waits, default=0),
            "magic_state_wait_mean_beats": round(
                wait_total / max(1, self.request_gate_count),
                6,
            ),
            "magic_state_wait_p50_beats": self._nearest_rank(self._gate_waits, 0.50),
            "magic_state_wait_p95_beats": self._nearest_rank(self._gate_waits, 0.95),
            "magic_state_generated_by_cr": generated_by_cr,
            "magic_state_consumed_by_cr": consumed_by_cr,
            "magic_state_wait_beats_by_cr": wait_by_cr,
            "magic_state_peak_buffer_by_cr": peak_by_cr,
            "magic_state_final_buffer_by_cr": final_by_cr,
        }
