# SO-101×ACT模倣学習

omnicampusアカウント名：Moriyama5556

Physical AI 応用1の最終課題向けに、3タスクの触覚なし実行経路を整理した公開用ソースです。
実験時の共通ランナーからセンサ接続・記録・描画、他の入力方式、ローカル固定パスを除き、
タスク設定をJSONへ分離しました。撮影時の原本をそのまま並べたものではなく、公開用のリファクタリング版です。
公開用版では実機の再撮影・再学習はしていません。背景マスク試験とモデル読込み・推論のオフライン確認を実施しました。

## 構成

| ファイル | 内容 |
|---|---|
| record.py | LeRobot標準CLIによる2カメラ・関節状態のテレオペ収録 |
| preprocess_dataset.py | External動画への固定背景マスク、画像統計の再計算 |
| train.py | 触覚なしACTの学習コマンド生成と学習ログ保存 |
| evaluate.py | 学習済みACTの読込み、推論、実機指令、動画・関節・結果の保存 |
| vision.py | 学習／推論で共通のマスク、手先カメラの調整 |
| configs/ | 採用モデルの学習条件とタスク設定 |
| assets/ | 再起立用ROIと積み上げ／再起立の初期姿勢 |
| results/ | 触覚なしの試行結果と、比較結果の集計値のみ |
| smoke_checkpoint.py | ロボットを接続しないモデル推論確認 |

このパッケージは触覚データ、触覚対応コード、学習画像、モデル重みを含みません。
モデル実行には別途、触覚なしの学習済みcheckpoint一式（重みと正規化用processor）が必要です。
新規学習には自身で収録したデータを用います。授業の提出要件はソースコードの提出であり、
この配布物だけで撮影済み結果が再現されるという意味ではありません。

## 実験結果・条件

| タスク | 学習デモ | 学習Step | 評価データ分割 | 完全成功率 |
|---|---:|---:|---:|---:|
| stack：角柱上に円柱、その上にニンニク形ソフビ | 30 Ep | 10,000 | 0% | 15%（3/20） |
| prism：角柱を倒し、再把持して正立 | 29 Ep | 10,000 | 0% | 30%（3/10） |
| cup：重ね合わせ、2つ同時に持ち上げ、下ろす | 20 Ep | 5,000 | 20% | 30%（6/20） |

ACT、ResNet18、batch 4、seed 1000、chunk size 100、n_action_steps 100、学習率1e-5。
画像640×480×2、関節状態6次元、指令6次元。学習画像augmentationは無効。
重さはアルミ円柱114g、アルミ角柱145g（実測値）。
成功判定は操作者のSキー入力による全工程完了判定です。Qで中断した試行を成功扱いにしません。

積み上げの撮影モデルは2,500Step checkpointから再開し、10,000Stepへ到達したものです。
公開用train.pyは新規学習を開始します。configs内の歴史情報historical_resumeで区別しています。
公開用ランナーの積み上げ上限は60秒です。元評価の上限設定は3,600秒でしたが、採用成功動画は約21秒です。
再学習は乱数、収録状態、カメラ、制御周期などで結果が変わります。

## 環境

撮影・確認環境：Windows、Python 3.12、LeRobot 0.6.1、PyTorch 2.11.0+cu128。
LeRobotは [huggingface/lerobot](https://github.com/huggingface/lerobot) の
commit `0d383d09f2051444de211739196a28cc94736861` を基準にしています。
本パッケージは、その公開APIに依存します。第三者ライブラリは元のライセンスに従って利用してください。

```powershell
git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout 0d383d09f2051444de211739196a28cc94736861
pip install -e ./lerobot
pip install -r requirements.txt
```

PyTorchは利用するGPU／CUDAに対応した構成を別途準備してください。
ロボットの校正とカメラ設定は使用環境に合わせます。保存姿勢は同じ校正条件用です。

## 収録・前処理・学習

例：積み上げ。prism／cupもtask引数を変えて使います。

```powershell
python record.py --task stack --repo-id local/stack_raw --output datasets/stack_raw --print-only
python record.py --task stack --repo-id local/stack_raw --output datasets/stack_raw
python preprocess_dataset.py --task stack --source datasets/stack_raw --output datasets/stack
python train.py --task stack --dataset datasets/stack --repo-id local/stack --output models/stack --print-only
python train.py --task stack --dataset datasets/stack --repo-id local/stack --output models/stack
```

成功デモを選別してから学習します。データセットのmeta/info.jsonの観測は画像2種と関節状態のみとし、
収録時と推論時で同じ固定マスクを適用します。preprocess_dataset.pyは入力データを変更せずコピーを作ります。
画像統計の再計算は5フレーム間隔・空間8画素間隔のサンプルです。画像の正規化は記録設定どおりImageNet統計を使用します。
手先カメラはRaw RGBを使用します。prismの露出・コントラスト・鮮明度はカメラの設定画面で収録前に揃えます。
要求値とドライバの実際の値は異なる場合があるため、読戻しと映像を確認してください。

## 推論と評価

```powershell
python smoke_checkpoint.py --checkpoint models/stack/checkpoints/010000/pretrained_model
python evaluate.py --task stack --checkpoint models/stack/checkpoints/010000/pretrained_model --output runs/stack_check --dry-run
python evaluate.py --task stack --checkpoint models/stack/checkpoints/010000/pretrained_model --output runs/stack_eval --trials 20
```

起動時に保存されるpreview_external.png／preview_hand.pngでカメラ割当と前処理を確認します。
実行時はRUNを入力し、積み上げ／再起立はHで初期姿勢へ戻してからRで開始します。
S＝全工程成功、F＝失敗、Q＝中断。失敗理由は終了後に記録します。
dry-runは推論による動作指令とホーム移動を送信しませんが、ロボットへの接続自体は行います。
操作停止は推論による動作指令の停止であり、モータ保持は継続する設定です。

動画は30fpsで保存されます。実効制御周期が30Hzに届かない場合、再生時間は実行時間と一致しません。
joint_positions.csvのelapsed_sを用いて編集時に補正します。

## オフライン確認

```powershell
python -m unittest discover -s tests -v
python smoke_checkpoint.py --checkpoint <触覚なしcheckpointのディレクトリ>
```

本パッケージの確認範囲はvalidation.jsonに記載しています。センサや実機への接続はテストしていません。
GitHubへはこのフォルダの内容を配置してください。データや評価フォルダを丸ごと追加しない構成になっています。
