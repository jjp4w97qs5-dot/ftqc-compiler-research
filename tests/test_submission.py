"""提出対象6プログラムの意味・計画・抽象資源を検証する。"""
from __future__ import annotations

import cmath
from collections import Counter, defaultdict
from dataclasses import replace
import math
import random
import unittest

from experiments import adder, ising2d, multiplier, qaoa_ising1d, qft
from lsqca_eval.lowering import LoweredOp, lower_program
from lsqca_eval.metadata import META_ANGLE
from lsqca_eval.planners.commuting_interactions import infer_interaction_architecture
from lsqca_eval.planners.qasmbench_multiplier_plan import build_qasmbench_multiplier_wave, wave_order_macros
from lsqca_eval.planners.qft_plan import build_qft_blocked_cr_plan, build_qft_diagonal_waves
from lsqca_eval.program_ir import Program
from lsqca_eval.programs.cdkm_adder import build_cdkm_adder
from lsqca_eval.programs.ising2d import Ising2DWorkload, build_ising2d_program
from lsqca_eval.programs.qasmbench_multiplier import build_qasmbench_multiplier, qasm_order_macros
from lsqca_eval.programs.qft import build_qft_original_order
from lsqca_eval.routing_audit import audit_path_occupancy
from lsqca_eval.scheduler import schedule_program
from lsqca_eval.schedulers.adder_pipeline import schedule_adder_pipeline

# 状態ベクトル比較で許容する最大振幅誤差。
TOLERANCE = 1e-10


def _bit_mask(qubit: int, qubit_count: int) -> int:
    """q0をMSBとするstate index上のbit maskを返す。"""

    return 1 << (qubit_count - 1 - qubit)


def _apply_lowered_op(
    state: list[complex],
    qubit_count: int,
    operation: LoweredOp,
) -> None:
    """test対象の基本gateをstatevectorへ適用する。"""

    gate = operation.op
    if gate == "H":
        mask = _bit_mask(operation.qubits[0], qubit_count)
        scale = 1.0 / math.sqrt(2.0)
        for index in range(len(state)):
            if index & mask:
                continue
            partner = index | mask
            zero = state[index]
            one = state[partner]
            state[index] = (zero + one) * scale
            state[partner] = (zero - one) * scale
        return
    if gate in {"RZ", "T", "TDG"}:
        mask = _bit_mask(operation.qubits[0], qubit_count)
        if gate == "RZ":
            angle = float(operation.meta[META_ANGLE])
            phase_zero = cmath.exp(-0.5j * angle)
            phase_one = cmath.exp(0.5j * angle)
        elif gate == "T":
            phase_zero = 1.0 + 0.0j
            phase_one = cmath.exp(0.25j * math.pi)
        else:
            phase_zero = 1.0 + 0.0j
            phase_one = cmath.exp(-0.25j * math.pi)
        for index in range(len(state)):
            state[index] *= phase_one if index & mask else phase_zero
        return
    if gate == "CX":
        control = _bit_mask(operation.qubits[0], qubit_count)
        target = _bit_mask(operation.qubits[1], qubit_count)
        for index in range(len(state)):
            if index & control and not index & target:
                partner = index | target
                state[index], state[partner] = state[partner], state[index]
        return
    raise AssertionError(f"Unsupported statevector test gate: {gate}")


def _simulate(program: Program, initial: list[complex]) -> list[complex]:
    """plan directiveを無視し、Programの量子gateだけを適用する。"""

    state = list(initial)
    for event in lower_program(program).events:
        if isinstance(event, LoweredOp):
            _apply_lowered_op(state, len(program.qubits), event)
    return state


def _basis_state(qubit_count: int, index: int) -> list[complex]:
    """指定indexのcomputational basis stateを返す。"""

    state = [0.0j] * (1 << qubit_count)
    state[index] = 1.0 + 0.0j
    return state


def _reverse_bits(value: int, width: int) -> int:
    """width bitの順序を反転した整数を返す。"""

    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _swap_free_qft(initial: list[complex], qubit_count: int) -> list[complex]:
    """正符号のB_n F_nを定義式から計算する。"""

    dimension = 1 << qubit_count
    scale = 1.0 / math.sqrt(dimension)
    fourier = [0.0j] * dimension
    for output in range(dimension):
        total = 0.0j
        for source, amplitude in enumerate(initial):
            total += amplitude * cmath.exp(2j * math.pi * source * output / dimension)
        fourier[output] = total * scale
    return [fourier[_reverse_bits(output, qubit_count)] for output in range(dimension)]


def _max_error_up_to_global_phase(actual: list[complex], expected: list[complex]) -> float:
    """二state間のglobal phaseを除いた最大振幅誤差を返す。"""

    overlap = sum(reference.conjugate() * value for value, reference in zip(actual, expected))
    phase = overlap / abs(overlap) if abs(overlap) > 1e-15 else 1.0 + 0.0j
    return max(abs(value - phase * reference) for value, reference in zip(actual, expected))


def _random_state(n: int) -> list[complex]:
    """再現可能な重ね合わせ入力を用意する。"""
    generator = random.Random(20260930 + n)
    state = [complex(generator.uniform(-1, 1), generator.uniform(-1, 1)) for _ in range(1 << n)]
    norm = math.sqrt(sum(abs(value) ** 2 for value in state))
    return [value / norm for value in state]


def _gates(program: Program) -> list[LoweredOp]:
    """計画指示を除いた基本ゲート列を取り出す。"""
    return [event for event in lower_program(program).events if isinstance(event, LoweredOp)]


class SubmissionTest(unittest.TestCase):
    """公開スナップショットだけで動く意味・構造・資源検証。"""

    def assert_valid_trace(self, trace: list, metrics: dict, slots: int) -> None:
        """終了状態と時間区間から抽象資源の排他性を独立に検査する。"""
        for key in (
            "cr_overflow_events", "final_cr_resident_count", "final_cache_resident_count",
            "final_sam_cell_collision_count", "final_sam_out_of_layout_count",
        ):
            self.assertEqual(0, metrics[key], key)
        self.assertLessEqual(metrics["max_cr_occupancy"], slots)
        self.assertEqual(max(op.end for op in trace), metrics["total_beats"])
        self.assertEqual(0, audit_path_occupancy(trace)["summary"]["declared_resource_conflict_count"])
        # 転送に宣言された資源、演算CR、SAM内演算、量子ビットの区間を照合する。
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for op in trace:
            self.assertGreaterEqual(op.start, 0)
            self.assertGreaterEqual(op.end, op.start)
            # BARRIERは子転送全体を示す同期記録で、追加の資源占有ではない。
            if op.end == op.start or op.op.endswith("_BARRIER"):
                continue
            resources = set(op.meta.get("route_resources", ()))
            resources.update(f"qubit:{q}" for q in op.qubits)
            if op.reason in {"cr_gate", "single_in_cr"}:
                resources.add(f"cr_compute:{op.cr_id}")
            elif op.reason == "inmemory_single":
                resources.add(f"bank:{op.bank_id}")
            for resource in resources:
                intervals[resource].append((op.start, op.end))
        for resource, spans in intervals.items():
            end = 0
            for start, next_end in sorted(spans):
                self.assertGreaterEqual(start, end, resource)
                end = next_end

    def assert_scheduled_gates(self, program: Program, trace: list) -> None:
        """ゲートの欠落・重複と、同じ量子ビット上の順序変更を検出する。"""
        expected = _gates(program)
        actual = [op for op in trace if op.reason in {"cr_gate", "single_in_cr", "inmemory_single", "cache_single"}]
        self.assertEqual(Counter((op.op, op.qubits) for op in expected), Counter((op.op, op.qubits) for op in actual))
        for q in range(len(program.qubits)):
            before = [(op.op, op.qubits) for op in expected if q in op.qubits]
            after = [(op.op, op.qubits) for op in sorted(actual, key=lambda op: (op.start, op.end)) if q in op.qubits]
            self.assertEqual(before, after, f"qubit {q}")

    def test_qft_semantics_against_fourier_definition(self) -> None:
        """n=4の全基底とn=8の重ね合わせでswapなしQFTの定義式と比較する。"""
        for n in (4, 8):
            inputs = [_basis_state(n, i) for i in range(1 << n)] if n == 4 else [_random_state(n)]
            expected = [value for state in inputs for value in _swap_free_qft(state, n)]
            for builder in (build_qft_original_order, build_qft_diagonal_waves, build_qft_blocked_cr_plan):
                with self.subTest(n=n, builder=builder.__name__):
                    actual = [value for state in inputs for value in _simulate(builder(n), state)]
                    self.assertLessEqual(_max_error_up_to_global_phase(actual, expected), TOLERANCE)

    def test_qft_blocked_execution(self) -> None:
        """block境界・循環転送・ゲート実行・終了状態を両SAMで検証する。"""
        for n in (4, 8, 16, 32, 64):
            program = build_qft_blocked_cr_plan(n)
            for sam in ("line-sam", "point-sam"):
                with self.subTest(n=n, sam=sam):
                    trace, metrics = schedule_program(program, qft.execution_config(sam))
                    self.assert_valid_trace(trace, metrics, 4)
                    self.assert_scheduled_gates(program, trace)
                    audit = qft.audit_blocked_plan(trace)
                    self.assertEqual((n // 4) * (n // 4 - 1) // 2, audit["cross_tile_count_audited"])
                    self.assertEqual(0, audit["cross_tile_sam_ops"])
                    # 循環転送で取り出した量子ビットが過不足なくCRへ戻ることを確認する。
                    outgoing = Counter((op.reason.split(":", 1)[-1], op.qubits) for op in trace if op.op == "CR_ROTATE_OUT")
                    incoming = Counter((op.reason.split(":", 1)[-1], op.qubits) for op in trace if op.op == "CR_ROTATE_IN")
                    self.assertEqual(outgoing, incoming)
                    if n > 4:
                        self.assertGreater(sum(outgoing.values()), 0)

    def test_commuting_interaction_preservation(self) -> None:
        """MaxCut/1D Isingの角度を含む多重集合と小規模の回路意味を確認する。"""
        for name, plan in qaoa_ising1d.PROGRAM_PLANS.items():
            for n in (8, 16, 64):
                with self.subTest(program=name, n=n):
                    reference = plan.build_reference(n, False)
                    architecture = infer_interaction_architecture(plan.build_reference(n, True))
                    program = plan.build_planned(n, architecture)
                    self.assertEqual(qaoa_ising1d.gate_multiset(reference), qaoa_ising1d.gate_multiset(program))
                    if n == 8:
                        state = _random_state(n)
                        self.assertLessEqual(_max_error_up_to_global_phase(_simulate(program, state), _simulate(reference, state)), TOLERANCE)
                    method = next(item for item in qaoa_ising1d.methods(architecture) if item.name == "hl_full_plan_4cr")
                    config = qaoa_ising1d.execution_config(method, "line-sam", architecture, name)
                    trace, metrics = schedule_program(program, config)
                    self.assert_valid_trace(trace, metrics, config.architecture.cr_slots)
                    self.assert_scheduled_gates(program, trace)

    def test_adder_five_slot_pipeline(self) -> None:
        """最小反復から代表31 bitまで、carry順・5 slot制約・home復帰を確認する。"""
        case = next(item for item in adder.CASES if item.name == "pipeline_5slot_sequential_shared_io")
        for bits in (1, 2, 5, 31):
            program = build_cdkm_adder(bits)
            for sam in ("line-sam", "point-sam"):
                with self.subTest(bits=bits, sam=sam):
                    trace, metrics = schedule_adder_pipeline(program, adder.execution_config(sam, case), split_io=False, layout_policy="sequential")
                    self.assert_valid_trace(trace, metrics, 5)
                    self.assert_scheduled_gates(program, trace)
                    self.assertEqual(0, metrics["final_home_mismatches"])
                    if bits >= 2:
                        self.assertGreater(metrics["compute_transfer_overlap_beats"], 0)

    def test_ising2d_resources_and_layer_boundaries(self) -> None:
        """小格子と複数stepでtile実行の資源・ゲート数・層順・終了状態を確認する。"""
        for height, width, steps in ((2, 2, 1), (4, 4, 2), (4, 6, 2)):
            workload = Ising2DWorkload(height, width, steps)
            expected = Counter((op.op, op.qubits) for op in _gates(build_ising2d_program(workload)))
            for sam in ("line-sam", "point-sam"):
                for method in ("high_level_tile_no_overlap", "high_level_tile_pipeline"):
                    with self.subTest(shape=(height, width, steps), sam=sam, method=method):
                        metrics, trace = ising2d.run_case(workload, sam, method)
                        self.assertTrue(metrics["case_valid"])
                        self.assert_valid_trace(trace, metrics, 4)
                        gates = [op for op in trace if op.reason == "cr_gate"]
                        self.assertEqual(expected, Counter((op.op, op.qubits) for op in gates))
                        for q in range(workload.qubits):
                            # 各量子ビットで磁場層→相互作用層→次stepの順を保つ。
                            phases = [(op.meta["trotter_step"], int(op.meta["phase"] == "interaction")) for op in sorted(gates, key=lambda op: op.start) if q in op.qubits]
                            self.assertEqual(sorted(phases), phases)

    def test_multiplier_structure_and_counts(self) -> None:
        """native/分解後ゲート数、macro保存、wave内の非共有を確認する。"""
        for width in (1, 2, 3, 5, 30):
            with self.subTest(width=width):
                original_macros = qasm_order_macros(width)
                waves = wave_order_macros(width)
                planned_macros = [macro for _, _, _, group in waves for macro in group]
                self.assertEqual(Counter(macro.name for macro in original_macros), Counter(macro.name for macro in planned_macros))
                self.assertEqual({macro.name: macro.native_ops for macro in original_macros}, {macro.name: macro.native_ops for macro in planned_macros})
                for _, _, _, group in waves:
                    operands = [q for macro in group for q in macro.qubits]
                    self.assertEqual(len(operands), len(set(operands)))
                # 部分積の生成/消去とcarry鎖から導いた閉形式で、生成器と独立に照合する。
                ccx = width * (width + 1) + 4 * width * (width - 1)
                cx = width * (4 * (width - 1) + 2)
                native = Counter(gate for macro in original_macros for gate, _ in macro.native_ops)
                self.assertEqual(Counter(CCX=ccx, CX=cx), native)
                expected = Counter(H=2 * ccx, CX=6 * ccx + cx, T=4 * ccx, TDG=3 * ccx)
                reference = build_qasmbench_multiplier(5 * width)
                program = build_qasmbench_multiplier_wave(5 * width)
                self.assertEqual(expected, Counter(op.op for op in _gates(program)))
                self.assertEqual(qaoa_ising1d.gate_multiset(reference), qaoa_ising1d.gate_multiset(program))
        # 15量子ビット参照規模を実際にscheduleしてゲートと資源も確認する。
        program = build_qasmbench_multiplier_wave(15)
        trace, metrics = schedule_program(program, multiplier.execution_config("line-sam"))
        self.assert_valid_trace(trace, metrics, 4)
        self.assert_scheduled_gates(program, trace)

    def test_multiplier_native_reordering_equivalence(self) -> None:
        """w=1/2の全基底入力でnative可逆回路の並べ替えを独立に確認する。"""
        def apply_native(macros: list, value: int) -> int:
            """CX/CCXをbit反転として解釈し、回路生成器とは独立に実行する。"""
            for macro in macros:
                for gate, qubits in macro.native_ops:
                    indices = [int(q[1:]) for q in qubits]
                    if all(value & (1 << q) for q in indices[:-1]):
                        value ^= 1 << indices[-1]
            return value

        for width in (1, 2):
            before = qasm_order_macros(width)
            after = [macro for _, _, _, group in wave_order_macros(width) for macro in group]
            for basis in range(1 << (5 * width)):
                self.assertEqual(apply_native(before, basis), apply_native(after, basis))

    def test_invalid_plan_dimensions_are_rejected(self) -> None:
        """QFTのblock端数、Multiplierの不正規模、Adderのslot不足を拒否する。"""
        for n in (0, -4, 5, 6):
            with self.subTest(qft_n=n), self.assertRaises(ValueError):
                build_qft_blocked_cr_plan(n)
        for n in (0, -5, 14):
            with self.subTest(multiplier_n=n), self.assertRaises(ValueError):
                build_qasmbench_multiplier_wave(n)
        case = next(item for item in adder.CASES if item.name == "pipeline_5slot_sequential_shared_io")
        config = adder.execution_config("line-sam", case)
        config = replace(config, architecture=replace(config.architecture, cr_slots=4))
        with self.assertRaises(ValueError):
            schedule_adder_pipeline(build_cdkm_adder(2), config, split_io=False, layout_policy="sequential")


if __name__ == "__main__":
    unittest.main()
