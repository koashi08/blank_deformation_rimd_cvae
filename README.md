# RIMD to DeltaXY CVAE

ブランク変形解析における座標補正予測のためのRIMDベース条件付き変分オートエンコーダー

## 概要

本プロジェクトは、RIMD（回転不変メッシュ記述子）特徴量と条件付き変分オートエンコーダー（CVAE）を用いて、1stepブランク変形解析における座標補正値（Δx, Δy）を予測する機械学習アプローチを実装しています。

## 特徴

- RIMDベース幾何学的特徴量抽出
- 条件付き変分オートエンコーダーの実装
- 柔軟な実験設定システム
- Jupyterノートブックベースのワークフロー
- 包括的な評価・可視化ツール

## インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd blank_deformation_rimd_cvae

# 依存関係のインストール
pip install -r requirements.txt

# または開発モードでインストール
pip install -e .
```

## クイックスタート

1. **データ前処理**: `notebooks/01_preprocessing.ipynb` を実行
2. **学習**: `notebooks/02_training.ipynb` を実行
3. **評価**: `notebooks/03_evaluation.ipynb` を実行
4. **分析**: `notebooks/04_analysis.ipynb` を実行

## プロジェクト構成

```
blank_deformation_rimd_cvae/
├── config/                 # 設定ファイル
├── src/                    # ソースコード
│   ├── data/               # データ処理
│   ├── models/             # モデル定義
│   ├── training/           # 学習ロジック
│   ├── evaluation/         # 評価モジュール
│   └── utils/              # ユーティリティ
├── notebooks/              # Jupyterノートブック
├── experiments/            # 実験結果
└── data/                   # データセット格納
```

## 設定

実験は設定システムを使用して構成されます：

```python
from config.experiment_configs import get_baseline_config, get_cvae_config

# ベースライン実験
config = get_baseline_config()

# CVAE実験
config = get_cvae_config()

# カスタム実験
config = get_baseline_config()
config.exp_id = "my_experiment"
config.model.hidden_dim = 256
```

## 実験タイプ

### 1. ベースライン実験（MLP）
- 標準的な多層パーセプトロン
- RIMD特徴量を用いた決定論的予測
- 基本的な幾何学的正則化

### 2. CVAE実験
- 条件付き変分オートエンコーダー
- 不確実性のモデリング
- βウォームアップスケジュールによる学習

### 3. GNN実験（将来実装予定）
- グラフニューラルネットワーク
- グラフ構造を活用した予測

### 4. 大規模モデル実験
- より深いネットワーク構造
- 高次元隠れ層

## 主要コンポーネント

### RIMD特徴量
- 回転不変幾何記述子
- エッジ特徴量（長さ、角度、変形勾配）
- ノード特徴量（局所幾何情報）

### 損失関数
- 再構成損失（Huber loss）
- ラプラシアン平滑化
- ドリフト抑制
- 2D-ARAP正則化（オプション）

### 評価メトリクス
- RMSE（平均二乗誤差の平方根）
- MAE（平均絶対誤差）
- Gain（改善率）
- 統計的分析（正規性検定、相関分析）

## 使用方法

1. **データ準備**: 変形メッシュデータを `data/` ディレクトリに配置
2. **前処理**: RIMD特徴量を計算し、データを分割
3. **学習**: 選択した設定でモデルを学習
4. **評価**: テストデータで性能を評価
5. **分析**: 結果を可視化し、詳細分析を実行

## 出力ファイル

### 学習結果
- `experiments/{exp_id}/models/`: 学習済みモデル
- `experiments/{exp_id}/logs/`: 学習ログ
- `experiments/{exp_id}/config.json`: 実験設定

### 評価結果
- `experiments/{exp_id}/evaluation/`: 詳細評価結果
- `experiments/{exp_id}/predictions/`: 予測結果
- `experiments/{exp_id}/visualizations/`: 可視化ファイル

## ライセンス

MIT License