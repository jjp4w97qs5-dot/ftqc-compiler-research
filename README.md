# FTQC Compiler Research — 高水準構造を用いた実行計画

誤り耐性量子計算に向けて、**量子ビットを記憶領域から演算領域へ移すコストを、高水準のプログラム構造を使って減らす**研究です。
量子回路を基本ゲートまで展開すると、元のループ、格子、グラフや、順序を入れ替えられる演算のまとまりが見えにくくなります。本研究では、それらを使って演算順序・配置・転送を計画し、並列性とデータの再利用を両立させる方法を検討しました。

本リポジトリは、提出レポートの高水準実行計画に対応する **Python製compiler/simulatorの検証済み抜粋**です。

## 自分で設計・実装した部分

- LSQCAのload/store型構成を基に、複数の演算領域（CR）と記憶領域（SAM）、cacheを扱う抽象資源モデルと低水準スケジューラを実装しました。
- 6プログラムについて、展開前の構造や演算の可換性を利用する計画を手作業で設計し、計画生成器と実行処理を実装しました。任意のプログラムから自動的に最適計画を発見するものではありません。
- 実験コード、実行トレース、検証テストを整備し、複数設定との比較と計画の構成要素を変えた実験で効果を評価しました。

LSQCAの原アーキテクチャや、QFT・加算器等の量子アルゴリズム自体の提案は先行研究によります。本研究の中心は、それらに対する実行計画と評価系です。

## 提出レポートの結果

別途提出する「LSQCA拡張モデルにおける高水準プログラム構造を用いた実行計画（研究中間報告）」
（`report_v12_revised.md`、表5・表6）では、展開後の回路だけを使う低水準スケジューラの
**CR保持2通り × prefetch2通り × 初期配置3通り = 12候補の最良値（方式C）**と、
高水準計画を使う方式Dを比較しました。

| プログラム | レポートの条件 | 総実行時間の短縮率 C→D |
|---|---|---:|
| Adder | 31 bit / 64量子ビット | 14.3% |
| QFT | 128量子ビット | 8.8% |
| MaxCut QAOA | 64頂点、1層 | 27.6% |
| 1D Ising | 64量子ビット、1層 | 10.3% |
| 2D Ising | 10×10格子、8ステップ | 20.5% |
| Multiplier | 150量子ビット | 9.7% |

この条件では6種すべてで **8.8〜27.6%短縮**しました。時間はモデル内の `beat`（CX 1回 = 1 beat）であり、物理装置の実行秒数ではありません。
主評価はLine-SAM、任意角RZ = 150 beat、magic stateの供給が十分な条件です。Dの設定はプログラムごとに異なり、AdderのDは **1 CR × 5 slots**、他は4 CR構成です。共通・専用スケジューラの違いもあるため、短縮率を高水準情報だけの独立した効果とは解釈していません。

### QFT：並列化だけでは改善しなかった例

CP演算の可換性を使ってwavefront順に並べても、同時に触る量子ビットが多く、CR内で十分に再利用できるとは限りません。そこで4量子ビットずつの区間に分け、区間の対ごとのblock内で量子ビットを再利用する計画を設計しました。非対角blockでは、片側をCRに保持し、もう片側をcache経由でCR間循環させます。

| レポート上の構成 | 対応するmethod | 総実行時間 [beat] | SAM境界転送 [回] |
|---|---|---:|---:|
| 方式C（12候補内最良） | 本snapshotでは全探索対象外 | 1,076,375 | 113,792 |
| 元の順序（Dと同じ低水準設定） | `base_order` | 1,256,395 | 101,524 |
| wavefront順 | `wave_order` | 1,312,923 | 98,636 |
| wavefront順＋並列実行群 | `wave_groups` | 1,301,462 | 100,142 |
| blocking・block内並列化（D） | `blocked_cr_plan` | 981,446 | 6,900 |

wavefront順は元の順序より遅く、blockingでは元の順序比 **21.9%**、C比 **8.8%**短縮しました。
**113,792→6,900回はC→D**の比較です。元の順序→Dは101,524→6,900回となり、「約10万回→約7千回」という説明に対応します。並列性と局所性を合わせて考える必要性を示す結果ですが、遅延原因の詳細な切り分けは今後の課題です。

**上記の数値は提出レポートの評価結果です。** 本snapshotの今回の検証範囲は後述のテストと実行例です。表5・付録の方式Cの12候補全数値や、表6の全数値を、このrepoで再現確認済みとはしていません。各driver内の `low_level_4cr`、`low_level_v2` 等も、それぞれ固定設定の比較用であり、方式Cの全探索を表しません。

## コードの流れ

```mermaid
flowchart LR
    A[Program IR: 回路・構造] --> B[Planner: 順序・配置・転送計画]
    B --> C[Lowering: 基本ゲートと計画指示]
    C --> D[共通または専用Scheduler]
    D --> E[LSQCA Resource Model: 資源予約・転送・演算]
    E --> F[Trace / Metrics: 実行履歴と評価指標]
```

[Program IR](lsqca_eval/program_ir.py)と[計画指示](lsqca_eval/execution_plan.py)で回路と実行計画を表し、[lowering](lsqca_eval/lowering.py)で基本ゲートへ展開します。
[共通scheduler](lsqca_eval/scheduler.py)は依存関係、CR選択、保持・退避・prefetchを扱います。
[architecture](lsqca_eval/architecture.py)がレイテンシと資源構成を定め、[machine](lsqca_eval/machine.py)が資源予約、転送、演算、状態・トレースの更新を担います。
Adderと2D Isingは、同じmachineの資源操作を使う専用schedulerで計画を実行します。

| 対象 | 利用した構造と計画 | 主な実装 / 実験driver |
|---|---|---|
| Adder | carry鎖の反復、量子ビットの共有、5 slotsでLD/STを演算に重ねる | [adder_pipeline.py](lsqca_eval/schedulers/adder_pipeline.py) / [adder.py](experiments/adder.py) |
| QFT | CPの可換性、wavefront、block内再利用とCR間循環 | [qft_plan.py](lsqca_eval/planners/qft_plan.py) / [qft.py](experiments/qft.py) |
| MaxCut QAOA | cost層の可換な相互作用、辺順序、CR割当て・初期配置 | [maxcut_qaoa_plan.py](lsqca_eval/planners/maxcut_qaoa_plan.py) / [qaoa_ising1d.py](experiments/qaoa_ising1d.py) |
| 1D Ising | 鎖の区間分割、相互作用の再利用、転送の重ね合わせ | [ising1d_plan.py](lsqca_eval/planners/ising1d_plan.py) / [共通相互作用planner](lsqca_eval/planners/commuting_interactions.py) |
| 2D Ising | 格子tile、tile内順序、境界辺の並列化 | [ising2d_plan.py](lsqca_eval/planners/ising2d_plan.py) / [ising2d_tile.py](lsqca_eval/schedulers/ising2d_tile.py) / [ising2d.py](experiments/ising2d.py) |
| Multiplier | 部分積とcarry鎖の依存関係、UNCOMPUTEの可換性 | [qasmbench_multiplier_plan.py](lsqca_eval/planners/qasmbench_multiplier_plan.py) / [multiplier.py](experiments/multiplier.py) |

入力回路の生成器は[programs/](lsqca_eval/programs/)にあります。MultiplierはQASMBenchの15量子ビット回路の構造を参照し、150量子ビット等は同じ規則から生成した派生回路です。

## 実行方法

Python **3.11以上**と標準ライブラリだけを使用します。追加パッケージのインストールは不要です。repoのルートで実行してください。

```bash
# 提出版のテスト
python3 -m unittest discover -s tests -v

# Adder：レポートDに対応する5-slot・shared I/Oの計画
python3 -m experiments.adder --bits 31 --sam-type line-sam --case pipeline_5slot_sequential_shared_io --output results/adder.csv

# QFT：小規模で元順序・wavefront・blockingの5構成を比較
python3 -m experiments.qft --n 16 --sam-type line-sam --output results/qft.csv

# MaxCut QAOAと1D Ising：両方の高水準計画を実行
python3 -m experiments.qaoa_ising1d --n 64 --sam-type line-sam --method hl_full_plan_4cr --output results/qaoa_ising1d.csv

# 2D Ising：小格子で共通schedulerとtile計画を比較（両SAMを出力）
python3 -m experiments.ising2d --height 4 --width 4 --steps 2 --output results/ising2d

# Multiplier：15量子ビットでmacro順・wavefrontの5構成を比較
python3 -m experiments.multiplier --n 15 --sam-type line-sam --output results/multiplier.csv
```

QFTのblocked planは量子ビット数がCR数（実行例では4）の倍数である必要があります。Multiplierは5の倍数です。
レポート規模を指定する場合は、QFTを `--n 128`、2D Isingを `--height 10 --width 10 --steps 8`、Multiplierを `--n 150` とします。
上記の小規模例は動作確認用です。規模によって方式の優劣は変わり、レポート規模の短縮率を示すものではありません。

実験結果はCSVに出力します。2D Isingでは圧縮JSONLのtraceも保存します。driverによっては実行失敗を `ERR` とCSVの `error` 列へ記録して処理を続けるため、終了コードだけで成功を判断せず、結果行も確認してください。生成先の `results/` は公開対象から除外しています。

指標の読み方：`total_beats` はスケジュール終了時刻、`seconds` / `schedule_seconds` はPython処理の実測秒数です。
既存CSVの `transfer_ops` 等をレポートのSAM境界転送と同一視せず、SAM境界転送はtraceの `source_kind` / `target_kind` がSAMとcache/CRの間を跨ぐものを数えます。
bank待ち・CR待ちは操作ごとの待ち時間の累積であり、総実行時間を超える場合があります。

## テストと確認範囲

[tests/test_submission.py](tests/test_submission.py)に、外部fixtureを必要としない8テストをまとめています。

2026-09-30にPython 3.12.4で8テストが合格し、38モジュールのimportと上記実行例19条件の出力も確認しました。
追加で、Line-SAM・レポート代表規模の方式Dの総実行時間6値が、表5と一致することを確認しました（Adder 2,183、QFT 981,446、MaxCut 6,504、1D Ising 5,056、2D Ising 105,484、Multiplier 125,192 beat）。QFT n=128では5構成の総実行時間を照合し、blocked planのSAM境界転送6,900回もtraceの端点から確認しています。方式Cの12候補全探索は今回実行していません。

- **QFT**：元順序・wavefront・blocked planを、n=4の全基底入力とn=8の重ね合わせ入力で独立したFourier変換の定義式と比較します。n=4/8/16/32/64の両SAMで循環転送、block内SAMアクセス、ゲート実行順、抽象資源、終了状態も検査します。
- **MaxCut / 1D Ising**：角度を含むゲート多重集合をn=8/16/64で照合し、n=8では状態ベクトルによる回路意味も比較します。
- **Adder**：1/2/5/31 bitでcarry鎖のゲート順、5-slot制約、転送の重なり、homeへの復帰を確認します。
- **2D Ising**：2×2、4×4、4×6で両SAM・重ね合わせ有無を検査し、ゲート数、層・stepの順序、資源競合、終了状態を確認します。
- **Multiplier**：w=1/2/3/5/30でmacroとゲート数を確認し、w=1/2の全基底入力でnative CX/CCX回路の並べ替え前後を比較します。n=15ではスケジュールも検査します。
- 不正なQFT/Multiplier規模と、Adderのslot不足が拒否されることを確認します。

確認対象は**抽象モデル内の回路・資源制約**です。物理的な格子手術経路、符号距離、誤り率、装置面積、factory配置・供給不足は、この評価の対象に含めていません。ゲート多重集合の一致だけを回路の意味的同値性の証明とはしていません。

## スナップショットの出典と範囲

提出元 `FTQC_compiler` のmain、コミット `044c93750691495daf5f2f1c4835c1653c32f60c` を基準に抽出しました。元repoのcommit履歴は引き継いでいません。
元コードからの動作変更は、2D Isingのtile内辺列が空の場合に空列を返す境界処理だけです。レポート規模10×10・8 stepsのLine-SAMでは、重ね合わせ有無の両方について、修正前後のtraceと実測秒数を除くmetricsの完全一致を確認しました。

レポートに直接対応するcore、6プログラム、planner、専用scheduler、実験driverを含みます。
研究途中の設計文書、対象外プログラム、後続研究の実験、保存済み `results/`、現行の低水準suiteと312-case fixtureは含めていません。
`magic_state.py` はcoreのimport依存として残していますが、上記の実行例では供給不足モデルを有効にしません。
