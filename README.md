# FTQC Compiler Research

高水準プログラム構造を用いて、誤り耐性量子計算の演算順序・配置・データ転送を計画するPython製compiler/simulatorです。LSQCAのload/store型アーキテクチャを基に、量子ビットを記憶領域（SAM）から演算領域（CR）へ移すコストと、演算の並列実行を評価します。

基本ゲートへの展開後には見えにくくなるループ、格子、グラフ、演算の可換性を計画に利用します。6プログラムの計画生成器、共通・専用スケジューラ、抽象資源モデル、実験driver、検証テストを含みます。計画はプログラムごとに設計したもので、任意の入力から最適計画を自動発見するシステムではありません。

## 構成

```mermaid
flowchart LR
    A[Program IR: 回路・構造] --> B[Planner: 順序・配置・転送計画]
    B --> C[Lowering: 基本ゲートと計画指示]
    C --> D[共通または専用Scheduler]
    D --> E[LSQCA Resource Model: 資源予約・転送・演算]
    E --> F[Trace / Metrics: 実行履歴と評価指標]
```

- [Program IR](lsqca_eval/program_ir.py)・[計画指示](lsqca_eval/execution_plan.py)：回路と実行計画の表現。
- [Lowering](lsqca_eval/lowering.py)：基本ゲートと計画指示への展開。
- [Scheduler](lsqca_eval/scheduler.py)：依存関係、CR選択、保持・退避・prefetch、計画の解釈。
- [Architecture](lsqca_eval/architecture.py)・[Machine](lsqca_eval/machine.py)：資源構成、レイテンシ、資源予約、転送・演算と状態更新。
- [Trace](lsqca_eval/execution_trace.py)・[Routing audit](lsqca_eval/routing_audit.py)：実行履歴と宣言された抽象資源の競合検査。

Adderと2D Isingは専用スケジューラを使います。資源予約と転送・演算の実行には、共通スケジューラと同じmachineの操作を利用します。

## 対象プログラム

| 対象 | 利用する構造と計画 | 主な実装 / 実験driver |
|---|---|---|
| Adder | carry鎖の反復と量子ビット共有を利用し、5 slotsでLD/STを演算に重ねる | [adder_pipeline.py](lsqca_eval/schedulers/adder_pipeline.py) / [adder.py](experiments/adder.py) |
| QFT | CPの可換性、wavefront、block内再利用とCR間循環 | [qft_plan.py](lsqca_eval/planners/qft_plan.py) / [qft.py](experiments/qft.py) |
| MaxCut QAOA | cost層の可換な相互作用、辺順序、CR割当て・初期配置 | [maxcut_qaoa_plan.py](lsqca_eval/planners/maxcut_qaoa_plan.py) / [qaoa_ising1d.py](experiments/qaoa_ising1d.py) |
| 1D Ising | 鎖の区間分割、相互作用の再利用、転送の重ね合わせ | [ising1d_plan.py](lsqca_eval/planners/ising1d_plan.py) / [qaoa_ising1d.py](experiments/qaoa_ising1d.py) |
| 2D Ising | 格子tile、tile内順序、境界辺の並列化 | [ising2d_plan.py](lsqca_eval/planners/ising2d_plan.py) / [ising2d_tile.py](lsqca_eval/schedulers/ising2d_tile.py) / [ising2d.py](experiments/ising2d.py) |
| Multiplier | 部分積とcarry鎖の依存関係、UNCOMPUTEの可換性 | [qasmbench_multiplier_plan.py](lsqca_eval/planners/qasmbench_multiplier_plan.py) / [multiplier.py](experiments/multiplier.py) |

入力回路の生成器は[programs/](lsqca_eval/programs/)にあります。MaxCut QAOAと1D Isingは[可換相互作用のplanner](lsqca_eval/planners/commuting_interactions.py)を共有します。MultiplierはQASMBenchの15量子ビット回路の構造を参照し、150量子ビット等は同じ規則から生成した派生回路です。

## 実行方法

Python 3.11以上と標準ライブラリだけを使用します。追加パッケージのインストールは不要です。repoのルートで実行してください。

```bash
# テスト
python3 -m unittest discover -s tests -v

# Adder：5-slot pipeline
python3 -m experiments.adder --bits 31 --sam-type line-sam --case pipeline_5slot_sequential_shared_io --output results/adder.csv

# QFT：元順序・wavefront・blockingの5構成
python3 -m experiments.qft --n 16 --sam-type line-sam --output results/qft.csv

# MaxCut QAOAと1D Ising：高水準計画
python3 -m experiments.qaoa_ising1d --n 64 --sam-type line-sam --method hl_full_plan_4cr --output results/qaoa_ising1d.csv

# 2D Ising：共通schedulerとtile計画（両SAMを出力）
python3 -m experiments.ising2d --height 4 --width 4 --steps 2 --output results/ising2d

# Multiplier：macro順・wavefrontの5構成
python3 -m experiments.multiplier --n 15 --sam-type line-sam --output results/multiplier.csv
```

QFTのblocked planは量子ビット数がCR数（実行例では4）の倍数である必要があります。Multiplierは5の倍数です。下記の評価規模で実行する場合は、QFTを `--n 128`、2D Isingを `--height 10 --width 10 --steps 8`、Multiplierを `--n 150` とします。小規模の実行例は動作確認用であり、規模によって方式の優劣は変わります。

結果はCSVに出力します。2D Isingでは圧縮JSONLのtraceも保存します。driverによっては実行失敗を `ERR` とCSVの `error` 列に記録して処理を続けるため、終了コードとともに結果行も確認してください。生成される `results/` はGitの追跡対象外です。

## 評価結果

研究レポート「LSQCA拡張モデルにおける高水準プログラム構造を用いた実行計画」の表5・表6の結果を示します。

比較対象の方式Cは、展開後の回路だけを使う低水準スケジューラについて、CR保持2通り × prefetch2通り × 初期配置3通りの12候補から選んだ最良値です。方式Dは高水準構造を利用する計画で、この条件では6種すべてで総実行時間を8.8〜27.6%短縮しました。

| プログラム | 評価条件 | C [beat] | D [beat] | 短縮率 C→D |
|---|---|---:|---:|---:|
| Adder | 31 bit / 64量子ビット | 2,546 | 2,183 | 14.3% |
| QFT | 128量子ビット | 1,076,375 | 981,446 | 8.8% |
| MaxCut QAOA | 64頂点、1層 | 8,986 | 6,504 | 27.6% |
| 1D Ising | 64量子ビット、1層 | 5,637 | 5,056 | 10.3% |
| 2D Ising | 10×10格子、8ステップ | 132,665 | 105,484 | 20.5% |
| Multiplier | 150量子ビット | 138,669 | 125,192 | 9.7% |

主評価はLine-SAM、任意角RZ = 150 beat、magic stateの供給が十分な条件です。Dの設定はプログラムごとに異なり、AdderのDは1 CR × 5 slots、他は4 CR構成です。共通・専用スケジューラの違いもあるため、短縮率を高水準情報だけの独立した効果とは解釈していません。

### QFT：並列性と局所性

CP演算の可換性を使ってwavefront順に並べても、同時に触る量子ビットが多く、CR内で十分に再利用できるとは限りません。blocked planでは4量子ビットずつの区間に分け、区間の対ごとのblock内で再利用します。非対角blockでは片側をCRに保持し、もう片側をcache経由でCR間循環させます。

| 構成 | method | 総実行時間 [beat] | SAM境界転送 [回] |
|---|---|---:|---:|
| 方式C（12候補内最良） | — | 1,076,375 | 113,792 |
| 元の順序（Dと同じ低水準設定） | `base_order` | 1,256,395 | 101,524 |
| wavefront順 | `wave_order` | 1,312,923 | 98,636 |
| wavefront順＋並列実行群 | `wave_groups` | 1,301,462 | 100,142 |
| blocking・block内並列化（D） | `blocked_cr_plan` | 981,446 | 6,900 |

wavefront順は元の順序より遅く、blockingでは元の順序比21.9%、C比8.8%短縮しました。113,792→6,900回はC→Dの比較で、元の順序→Dは101,524→6,900回です。並列性と局所性を合わせて考える必要性を示す結果ですが、遅延原因の詳細な切り分けは今後の課題です。

### 指標と再現範囲

- `total_beats`：モデル内のスケジュール終了時刻。CX 1回を1 beatとし、物理装置の実行秒数は表しません。
- `seconds` / `schedule_seconds`：Python処理の実測秒数。
- SAM境界転送：traceの `source_kind` / `target_kind` がSAMとcache/CRの間を跨ぐ転送の回数。既存CSVの `transfer_ops` 等とは集計範囲が異なります。
- bank待ち・CR待ち：各操作の待ち時間の累積値。総実行時間を超える場合があります。

2026-09-30にPython 3.12.4で、上表のDの総実行時間6値が一致することを確認しました。QFT n=128では5構成の総実行時間を照合し、blocked planのSAM境界転送6,900回もtraceの端点から確認しています。

方式Cの12候補全探索は本repoの実験driverに含めておらず、上表のCは研究レポートの評価値です。各driverの `low_level_4cr`、`low_level_v2` 等は固定設定の比較用であり、Cの全探索を表しません。レポート表5・付録の全数値を再現する一式ではありません。

## 検証

[tests/](tests/)の8テストは外部fixtureを必要とせず、Python 3.12.4で合格を確認しています。38モジュールのimportと、上記実行例19条件の出力も確認済みです。

| 対象 | 検証内容 |
|---|---|
| QFT | n=4全基底・n=8重ね合わせ入力を独立したFourier変換の定義式と比較。n=4/8/16/32/64の両SAMで循環転送、block内SAMアクセス、ゲート順、抽象資源、終了状態を検査 |
| MaxCut / 1D Ising | n=8/16/64で角度を含むゲート多重集合を照合。n=8では状態ベクトルで回路意味も比較 |
| Adder | 1/2/5/31 bitでcarry鎖のゲート順、5-slot制約、転送の重なり、homeへの復帰を確認 |
| 2D Ising | 2×2、4×4、4×6の両SAM・重ね合わせ有無で、ゲート数、層・step順、資源競合、終了状態を確認 |
| Multiplier | w=1/2/3/5/30でmacroとゲート数を照合。w=1/2全基底入力でnative CX/CCX回路の並べ替え前後を比較。n=15のスケジュールも検査 |
| 入力制約 | 不正なQFT/Multiplier規模とAdderのslot不足を拒否 |

検証対象は抽象モデル内の回路・資源制約です。ゲート多重集合の一致だけを意味的同値性の証明とはしていません。物理的な格子手術経路、符号距離、誤り率、装置面積、factory配置・供給不足は上記評価の対象外です。LSQCAの原アーキテクチャや各量子アルゴリズムは先行研究に基づき、本repoではそれらに対する実行計画と評価系を扱います。
