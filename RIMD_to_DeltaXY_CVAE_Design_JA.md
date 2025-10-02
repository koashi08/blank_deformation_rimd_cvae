# 実装設計書：RIMD → 座標誤差 Δx,Δy 直接出力

## 0. 概要

- **目的**：1step 解析のブランク展開に対し、逐次解析（正解）との**座標誤差** $\delta u_i=(\Delta x_i,\Delta y_i)$ を**直接**予測し、  
  $X^{pred}_{2D}=X^{1step}_{2D}+\hat{\delta u}$ で補正形状を得る。  
- **入力**：1step の **RIMD特徴**（辺の回転差ログ $\phi_{ij}\in\mathbb{R}^3$、頂点の $\log S_i\in\mathbb{R}^6$）＋補助幾何。  
- **出力**：頂点ごとの 2D 変位 $\delta u$（標準化後の値）。  
- **前提**：節点対応は一致、座標系は統一（x中心0 / y片端0 / z=0）、規模は **V=1071**、Eは概ね $\approx 2V$。  
- **ねらい**：RIMDの**回転不変**・**局所性**を活かして学習を安定化しつつ、再構成ソルバ無しで**高速推論**。

---

## 1. 設計方針（と理由）

1) **RIMDを条件に採用**  
   - 回転・並進に不変で、曲げ主導の変形で分布が安定。形状間一般化に有利。

2) **ターゲットは座標誤差（Δx,Δy）**  
   - 直接加算で補正完了、ARAP/Poisson系の再構成不要 → **実装がシンプル**・**推論が高速**。  
   - ターゲット次元が $2V$ と低めで**少データ（100ケース）でも安定**。

3) **順列不変な条件エンコーダ＋ノード共有デコーダ**  
   - 辺・頂点特徴を小MLPで埋め込み、**mean-pool**でグローバル条件 $h_c$。  
   - 各頂点は $[node\_feat_i; h_c]$ から**同一MLP**で $\delta u_i$ を出力 → パラメタ節約。

4) **軽い幾何正則化**  
   - 予測変位場の**ラプラシアン平滑化**と**ドリフト抑制**で非物理なギザギザ／一様平行移動を回避。  
   - 必要に応じ 2D-ARAP 風の極薄正則化で安定化。

5) **CVAE は段階導入**  
   - まず**非VAE（条件付き回帰）**で配管・精度確認。  
   - 次に **CVAE（グローバル潜在 z）** で不確実性表現／p95 改善を狙う（βウォームアップ必須）。

---

## 2. データ仕様・前処理

### 2.1 生データ
- ノード単位のxyz座標値(1step法:step_blk_coord_x/step_prod_coord_x, 逐次解析:nv_blk_coord_x/nv_prod_coord_x 等)はdf_node.pklに保存済み。
  - NodeIDがcaseごとに1~1071まで与えられている。
- 要素単位の構成情報データ(ElementID n1 n2 n3 n4)はdf_element.pklに保存済み。
- 両方ともcaseカラムでケースNoを識別（No001, No002, ...）

### 2.2 三角化と RIMD 算出
- 四角形メッシュは**一貫規約の対角線**で三角化（全ケース同一規約）。  
- ブランク展開形状→製品形状の対で、各頂点の**変形勾配** $T_i$ を cotan 重み一環最小二乗で推定、**極分解** $T_i=R_iS_i$。  

- **RIMD**：  
  - 辺：$\phi_{ij}=\log(R_i^T R_j)\in\mathbb{R}^3$
  - 頂点：$\sigma_i=\log S_i\in\mathbb{R}^6$

### 2.3 ターゲット（座標誤差）
- $\delta u_i=(x^{nv}_i-x^{1step}_i,\; y^{nv}_i-y^{1step}_i)$

### 2.4 入出力テンソル（ケース No001 の例）
```
 No001/
  rimd_edges_1step.npy   # [E, 3]   φ^1step_ij
  rimd_nodes_1step.npy   # [V, 6]   σ^1step_i
  edge_index.npy         # [2, E]   (i, j)
  delta_xy.npy           # [V, 2]   (Δx, Δy)
  xy_1step.npy           # [V, 2]   1step座標（後処理用）
  xy_nv.npy              # [V, 2]   逐次解析座標（評価用）
  geometry_features.npy  # [V, Gn]  補助幾何特徴（面積、角度等）
  edge_geometry.npy      # [E, Ge]  辺の幾何特徴
```

### 2.5 標準化（データセット基準）
1. **代表寸法 $S$**：train の 1step 座標から BB 対角長の**中央値**を採用（データセット一定）。  
   $\tilde{\delta u}=\delta u/S$ で粗スケーリング（数値安定・相似性付与）。
2. **Z標準化**（train で fit、全 split に適用）：  
   - 辺：$[\phi,\text{geo\_edge}]\ \Rightarrow\ (\mu_e,\sigma_e)$  
   - 頂点：$[\sigma,\text{geo\_node}]\ \Rightarrow\ (\mu_v,\sigma_v)$  
   - ターゲット：$\tilde{\delta u}\ \Rightarrow\ (\mu_y,\sigma_y)$  
3. **逆変換**（推論）：$\hat{\delta u}=((\hat y\cdot\sigma_y+\mu_y)\cdot S)$

> スケーラは 保存（$\mu,\sigma,S$）。辺・頂点・ターゲットで**別々**に管理。

---

## 3. モデル設計

### 3.1 入力特徴
- **エッジ特徴**：$[\ \phi^{1step}_{ij}\ (3);\ \text{geo\_edge}_{ij}\ (Ge)\ ]$  
- **ノード特徴**：$[\ \sigma^{1step}_i\ (6);\ \text{geo\_node}_i\ (Gn);\ \bar{\phi}^{1step}_i\ (3);\ (x^{1step}_i,y^{1step}_i)\ (2)\ ]$  
  - $\bar{\phi}^{1step}_i$：頂点 i の一環辺 $\phi$ の平均（近傍曲げ強度の簡易要約）。

### 3.2 ベースライン（非VAE：条件付き回帰 MLP）
- **Edge-Enc-MLP**：in=(3+Ge) → 128（GELU＋LayerNorm）→ **mean-pool** → $h_e\in\mathbb{R}^{128}$
- **Node-Enc-MLP**：in=(6+Gn+3+2) → 128 → **mean-pool** → $h_v\in\mathbb{R}^{128}$
- **Global-MLP**：$[h_e;h_v]$ → $h_c\in\mathbb{R}^{128}$
- **Node-Head-MLP（共有）**：$[\text{node\_feat}_i;h_c]$ → 128 → 128 → 64 → **2**
  - 出力は**標準化後**の $\tilde{\delta u}_{\text{std}}$。Dropout0.1、LayerNorm、最終層は線形。
  - **Residual Connection**: Node特徴の元次元に射影してresidual接続を追加（勾配流改善）

> 置換案：GNN（GraphSAGE/GCN 2–3層）＋ FiLM（後述の $g$ で層を変調）。最初は MLP 推奨。
> **アテンション機構**: 重要な近傍特徴を選択的に重み付けするself-attention層の検討。

### 3.3 CVAE 拡張
- **Encoder（学習時のみ）**：$\tilde{\delta u}$ を小MLP(32→64)で埋め、**mean-pool** → $u\in\mathbb{R}^{64}$。  
  $[u;h_c]$ → MLP(256→128) → **$\mu,\log\sigma^2\in\mathbb{R}^{d_z}$**、$d_z=8\sim16$。  
- **Decoder**：$[z;h_c]$ → **Context-MLP** → $g\in\mathbb{R}^{128}$、  
  各頂点で $[\text{node\_feat}_i;g]$ → Node-Head-MLP → $\tilde{\delta u}_{\text{std}}$。  
- **条件付き事前** $p(z|c)$：必要時に $\mu_0(c),\sigma_0(c)$ を小MLPで作る（後回しで可）。
- **3.2 ベースライン（非VAE：条件付き回帰 MLP）**と**3.3 CVAE 拡張**はフラグ一つで切り替え可能にできるよう実装する。

---

## 4. 損失関数

### 4.1 基本
- **再構成（主）**：**Huber**（標準化後のスケールで $\delta=1.0$ 目安）  
  $$
  L_{\text{rec}}=\sum_i \rho\!\big(\hat{\tilde{\delta u}}_{i,\text{std}}-\tilde{\delta u}_{i,\text{std}}\big)
  $$
- **ラプラシアン平滑化**（逆標準化後の mm 単位でも可）  
  $$
  L_{\text{lap}}=\sum_{(i,j)\in E} \|\hat{\delta u}_i-\hat{\delta u}_j\|^2
  $$
- **ドリフト抑制**  
  $$
  L_{\text{drift}}=\Big\|\sum_i \hat{\delta u}_i\Big\|^2
  $$
- **総損失（非VAE）**：  
  $L=L_{\text{rec}}+\lambda_{\text{lap}}L_{\text{lap}}+\lambda_{\text{drift}}L_{\text{drift}}$

### 4.2 CVAE 追加
- **KL**：$L_{\text{KL}}=\mathrm{KL}(q_\phi(z|\cdot)\,\|\,\mathcal N(0,I))$  
- **βウォームアップ**：0→1 を **約10エポック**、最終 $\beta\in[0.5,1.0]$ を比較  
- **総損失（CVAE）**：  
  $L=L_{\text{rec}}+\beta L_{\text{KL}}+\lambda_{\text{lap}}L_{\text{lap}}+\lambda_{\text{drift}}L_{\text{drift}}$

### 4.3 オプション
- **ヘテロスケダスティック**：平均 $\mu$ と分散 $\sigma^2$ を出し、**対数尤度**で $L_{\text{rec}}$ を置換。
- **2D-ARAP 風**：$J_i$ の極分解で $\|J_i-R_i\|_F^2$ を微小係数で追加。
- **境界制約**：固定境界（クランプ部等）での変位を0に制約する項。
- **物理一貫性**：エネルギー保存を近似的に満たす正則化項。
- **多解像度損失**：粗い〜細かいスケールでの誤差を階層的に評価。

> 初期値：$\lambda_{\text{lap}}=0.001 \sim 0.005$、$\lambda_{\text{drift}}=0.0001$。

---

## 5. 学習プロトコル

- **分割**：60/20/20（形状単位）。形状の複雑度による層別抽出を検討。
- **最適化**：AdamW(lr=1e-3, wd=1e-4)、batch=形状 1–4/step、epoch=100–300、**早期終了**（val $L_{\text{rec}}$）。
- **学習率スケジューリング**：CosineAnnealingLR または ReduceLROnPlateau の併用。
- **正規化**：LayerNorm、Dropout 0.1。勾配クリッピング（max_norm=1.0）追加。
- **CVAE**：$d_z=8\sim16$、βウォームアップ 10 epoch、最終β={0.5,0.7,1.0} 比較。
- **データ拡張**：RIMDの回転不変性を活かした幾何変換（スケール・微小回転）。
- **ログ**：train/val の $L_{\text{rec}}$、KL、**頂点距離（mm）**の median/p90/p95、**Gain** を毎 epoch 記録。
- **チェックポイント**：最良val性能モデルの自動保存とearly stopping patience=20設定。

---

## 6. 推論フロー

1) 入力特徴を**標準化**（学習時と同じスケーラ）。  
2) **非VAE**：$\hat{\tilde{\delta u}}_{\text{std}}=\text{Model}(c)$。  
   **CVAE**：既定は $z=0$（平均復元）、必要なら $z\sim \mathcal N(0,I)$ を複数サンプル。  
3) **逆標準化**：$\hat{\delta u}=((\hat y\cdot\sigma_y+\mu_y)\cdot S)$。  
4) **補正**：$X^{\text{pred}}_{2D}=X^{\text{1step}}_{2D}+\hat{\delta u}$。

---

## 7. 評価指標・可視化

- **Baseline-0（基準手法）**：
  1step解析結果をそのまま使用する手法を「**Baseline-0**」として定義。
  つまり、予測座標は $X^{Baseline-0} = X^{1step}$（補正なし）。

- **頂点距離（mm）**：RMSE / MAE と **median/p90/p95**（全体／曲げ線近傍で層別）。
- **頂点平均誤差**(ユークリッド距離, x座標, y座標)、**頂点最大誤差**(ユークリッド距離, x座標, y座標)、**寸法誤差**(x方向, y方向)
- **Gain（改善率）**：
  全ての手法の改善率は**Baseline-0からの相対値**として計算：
  $$
  \text{Gain}=\frac{\|X^{Baseline-0}-X^{nv}\|-\|X^{pred}-X^{nv}\|}{\|X^{Baseline-0}-X^{nv}\|}\times 100\%
  $$
  $$
  =\frac{\|X^{1step}-X^{nv}\|-\|X^{pred}-X^{nv}\|}{\|X^{1step}-X^{nv}\|}\times 100\%
  $$
  - **正の値**：予測手法がBaseline-0より優れている（改善）
  - **負の値**：予測手法がBaseline-0より劣っている（悪化）
  - **0%**：Baseline-0と同等の性能
- **ヒートマップ**：$|X^{pred}-X^{nv}|$ をメッシュ上で可視化（外れ箇所確認）。
- **CVAEのみ**：複数 $z$ サンプルの分散マップで**不確実性可視化**。
- **分布比較**：予測値と真値の誤差分布をKSテストで統計的比較。
- **特徴重要度**：SHAP値やpermutation importanceによる入力特徴の重要度分析。
- **収束解析**：学習曲線の傾きと飽和点の定量評価。
- **汎化性評価**：形状タイプ別・変形度別の性能分析。

---

## 8. アブレーション計画

### 8.1 基準手法（必須比較対象）
- **Baseline-0**：1step解析結果をそのまま使用（補正なし）
  - 全ての実験でBaseline-0との比較を必須とし、Gain計算の基準とする

### 8.2 主要実験項目
- **損失関数**：MSE vs Huber vs Smooth L1、$\lambda_{\text{lap}} \in \{0.001, 0.003, 0.005\}$。
- **入力特徴**：$(x,y)$ を入れる／抜く、$\bar\phi$ の有無、補助幾何の選択。
- **正規化手法**：BatchNorm vs LayerNorm vs GroupNorm、Dropout率の比較。
- **アーキテクチャ**：非VAE vs CVAE（$d_z=\{8,16\}$, β={0.5,0.7,1.0}）。
- **ネットワーク構造**：MLP vs GNN(GraphSAGE/GCN/GAT 2–3層) vs Transformer。
- **プーリング戦略**：mean-pool vs max-pool vs attention-pool vs set2set。
- **活性化関数**：GELU vs ReLU vs Swish vs Mish。
- **最適化手法**：Adam vs AdamW vs SGD+momentum、学習率スケジューリング。

### 8.3 性能評価基準
全ての手法をBaseline-0と比較し、以下を評価：
- **絶対精度**：RMSE, MAE, P95 Error
- **相対改善**：Baseline-0からのGain（改善率）
- **統計的有意性**：paired t-test等による有意差検定  

---

## 9. リスクと対策

- **過平滑化**：$\lambda_{\text{lap}}$ が大きすぎると曲げ線ピークが潰れる → 小さく開始し、val p95 を見ながら微調整。
- **ドリフト退化**：ラプラシアンのみだと一様平行移動が残る → **ドリフト抑制項**を必ず入れる。
- **CVAE崩壊**：zを使わない／KLが0に張り付く → βウォームアップ、最終βを下げる、$d_z$ を小さく、必要なら $p(z|c)$。
- **標準化リーク**：val/test を含めて fit しない。**train のみ**で fit。
- **三角化不一致**：四角→三角の規約がケースで変わると壊れる → **固定関数**で一貫実行。
- **勾配消失・爆発**：勾配クリッピングとスキップ接続、適切な重み初期化（Xavier/He）。
- **過学習**：早期終了、正則化強化、データ拡張、アンサンブル手法の検討。
- **数値不安定性**：混合精度学習時のloss scaling、極値の事前チェック。
- **メモリ不足**：勾配蓄積、動的batching、checkpointing の活用。
- **再現性**：乱数シード固定、deterministic演算の設定。

---

## 10. 実装アーキテクチャとファイル構成

### 10.1 プロジェクト構造
```
blank_deformation_rimd_cvae/
├── config/                      # 実験設定ファイル
│   ├── base_config.py           # ベース設定クラス
│   ├── experiment_configs.py    # 実験別設定
│   └── __init__.py
├── src/                        # メインコード
│   ├── data/                    # データ処理
│   │   ├── preprocessing.py
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── __init__.py
│   ├── models/                  # モデル定義
│   │   ├── base_model.py
│   │   ├── mlp_model.py
│   │   ├── gnn_model.py
│   │   ├── cvae_model.py
│   │   └── __init__.py
│   ├── training/                # 学習ロジック
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── __init__.py
│   ├── evaluation/              # 評価モジュール
│   │   ├── evaluator.py
│   │   ├── visualization.py
│   │   └── __init__.py
│   ├── utils/                   # ユーティリティ
│   │   ├── experiment_manager.py
│   │   ├── logger.py
│   │   ├── reproducibility.py
│   │   └── __init__.py
│   └── __init__.py
├── notebooks/                  # Jupyterノートブック
│   ├── 01_preprocessing.ipynb   # 前処理
│   ├── 02_training.ipynb        # 学習
│   ├── 03_evaluation.ipynb      # 評価
│   └── 04_analysis.ipynb        # 結果分析
├── experiments/                # 実験結果
│   ├── exp_001/
│   │   ├── config.json
│   │   ├── model_weights.pth
│   │   ├── scalers.pkl
│   │   ├── metrics.json
│   │   ├── logs/
│   │   └── visualizations/
│   └── exp_002/
├── data/                       # データセット
│   ├── raw/                     # 元データ
│   ├── processed/               # 前処理済み
│   └── splits/                  # train/val/test分割
├── requirements.txt
├── setup.py
└── README.md
```

### 10.2 設定管理システム

#### 10.2.1 ベース設定クラス (config/base_config.py)
```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
from pathlib import Path

@dataclass
class DataConfig:
    """データ関連設定"""
    data_root: str = "data/processed"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    batch_size: int = 4
    num_workers: int = 4

@dataclass
class ModelConfig:
    """モデル関連設定"""
    model_type: str = "mlp"  # "mlp", "gnn", "transformer"
    use_cvae: bool = False

    # ネットワークサイズ
    hidden_dim: int = 128
    latent_dim: int = 8  # CVAEのみ
    num_layers: int = 3

    # 正則化
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    gradient_clip_norm: float = 1.0

@dataclass
class LossConfig:
    """損失関数設定"""
    loss_type: str = "huber"  # "mse", "huber", "smooth_l1"
    huber_delta: float = 1.0

    # 正則化係数
    lambda_lap: float = 3e-3
    lambda_drift: float = 1e-4
    lambda_arap: float = 0.0  # 2D-ARAP正則化

    # CVAE関連
    beta_warmup_epochs: int = 10
    beta_final: float = 0.7

@dataclass
class TrainingConfig:
    """学習関連設定"""
    # 最適化
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # スケジューリング
    scheduler: str = "cosine"  # "cosine", "reduce_on_plateau", "none"

    # 学習制御
    max_epochs: int = 200
    early_stopping_patience: int = 20
    save_top_k: int = 3

    # 再現性
    seed: int = 42
    deterministic: bool = True

@dataclass
class ExperimentConfig:
    """実験全体設定"""
    exp_id: str = "exp_001"
    description: str = ""

    # サブ設定
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: Path):
        """設定をJSONで保存"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path):
        """設定をJSONから読み込み"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
```

#### 10.2.2 実験別設定 (config/experiment_configs.py)
```python
from .base_config import ExperimentConfig, ModelConfig, LossConfig

def get_baseline_config() -> ExperimentConfig:
    """ベースライン実験設定"""
    return ExperimentConfig(
        exp_id="baseline_mlp",
        description="ベースラインMLPモデル、非VAE"
    )

def get_cvae_config() -> ExperimentConfig:
    """CVAE実験設定"""
    config = get_baseline_config()
    config.exp_id = "cvae_mlp"
    config.description = "CVAEで不確実性モデリング"
    config.model.use_cvae = True
    return config

def get_gnn_config() -> ExperimentConfig:
    """GNN実験設定"""
    config = get_baseline_config()
    config.exp_id = "gnn_baseline"
    config.description = "GraphSAGEベースモデル"
    config.model.model_type = "gnn"
    return config

def get_ablation_configs() -> Dict[str, ExperimentConfig]:
    """アブレーションスタディ設定"""
    configs = {}

    # 損失関数比較
    for loss_type in ["mse", "huber", "smooth_l1"]:
        config = get_baseline_config()
        config.exp_id = f"ablation_loss_{loss_type}"
        config.loss.loss_type = loss_type
        configs[config.exp_id] = config

    # 正則化係数比較
    for lambda_lap in [1e-3, 3e-3, 5e-3]:
        config = get_baseline_config()
        config.exp_id = f"ablation_lambda_{lambda_lap:.0e}"
        config.loss.lambda_lap = lambda_lap
        configs[config.exp_id] = config

    return configs
```

### 10.3 Jupyterノートブック構成

#### 10.3.1 01_preprocessing.ipynb
```python
# セル1: 初期設定
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

from src.data.preprocessing import RIMDPreprocessor
from src.utils.experiment_manager import ExperimentManager
from config.experiment_configs import get_baseline_config

# セル2: 設定読み込み
config = get_baseline_config()
exp_manager = ExperimentManager(config)

# セル3: データ前処理
preprocessor = RIMDPreprocessor(config.data)
preprocessor.process_all_cases()
preprocessor.create_splits()

# セル4: スケーラ作成と保存
scalers = preprocessor.fit_scalers()
exp_manager.save_scalers(scalers)
```

#### 10.3.2 02_training.ipynb
```python
# セル1: 初期設定
from config.experiment_configs import get_baseline_config, get_cvae_config
from src.training.trainer import RIMDTrainer
from src.utils.experiment_manager import ExperimentManager

# セル2: 実験選択（簡単切り替え）
# config = get_baseline_config()  # ベースライン
config = get_cvae_config()        # CVAE実験

exp_manager = ExperimentManager(config)
exp_manager.setup_experiment()

# セル3: データローダー作成
from src.data.dataset import RIMDDataModule
datamodule = RIMDDataModule(config.data, exp_manager.scalers)

# セル4: モデル作成
from src.models import create_model
model = create_model(config.model)

# セル5: 学習実行
trainer = RIMDTrainer(config, exp_manager)
trainer.fit(model, datamodule)

# セル6: 結果保存
exp_manager.save_results(trainer.get_metrics())
```

#### 10.3.3 03_evaluation.ipynb
```python
# セル1: 学習済みモデル読み込み
from src.utils.experiment_manager import ExperimentManager

exp_id = "cvae_mlp_20241201_001"  # 実験ID指定
exp_manager = ExperimentManager.load_experiment(exp_id)
model, scalers = exp_manager.load_model_and_scalers()

# セル2: テストデータ評価
from src.evaluation.evaluator import RIMDEvaluator
evaluator = RIMDEvaluator(model, scalers, exp_manager.config)
test_metrics = evaluator.evaluate_test_set()

# セル3: 詳細分析
detailed_results = evaluator.detailed_analysis()
print(f"Median error: {test_metrics['median_error_mm']:.2f}mm")
print(f"P95 error: {test_metrics['p95_error_mm']:.2f}mm")
print(f"Gain: {test_metrics['gain_percent']:.1f}%")

# セル4: 可視化
from src.evaluation.visualization import ErrorVisualizer
visualizer = ErrorVisualizer(exp_manager.exp_dir)
visualizer.create_all_plots(detailed_results)
```

#### 10.3.4 04_analysis.ipynb
```python
# セル1: 複数実験の比較
from src.utils.experiment_manager import ExperimentComparer

exp_ids = ["baseline_mlp", "cvae_mlp", "gnn_baseline"]
comparer = ExperimentComparer(exp_ids)
comparison_results = comparer.compare_experiments()

# セル2: アブレーション結果分析
ablation_results = comparer.analyze_ablation_study()
comparer.plot_ablation_heatmap()

# セル3: 結果レポート作成
comparer.generate_report("final_results.pdf")
```

### 10.4 実験管理システム

#### 10.4.1 実験マネージャー (src/utils/experiment_manager.py)
```python
class ExperimentManager:
    """実験の設定、実行、結果管理"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.exp_dir = Path(f"experiments/{config.exp_id}")
        self.setup_directories()

    def setup_experiment(self):
        """実験環境のセットアップ"""
        # 設定保存
        self.config.save(self.exp_dir / "config.json")

        # ログ設定
        self.setup_logging()

        # 再現性設定
        self.setup_reproducibility()

    def save_scalers(self, scalers: Dict):
        """スケーラ保存"""
        joblib.dump(scalers, self.exp_dir / "scalers.pkl")

    def save_model(self, model, epoch: int, metrics: Dict):
        """モデル保存"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, self.exp_dir / f"model_epoch_{epoch}.pth")

    def save_results(self, metrics: Dict):
        """最終結果保存"""
        with open(self.exp_dir / "final_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    @classmethod
    def load_experiment(cls, exp_id: str):
        """保存済み実験の読み込み"""
        exp_dir = Path(f"experiments/{exp_id}")
        config = ExperimentConfig.load(exp_dir / "config.json")
        return cls(config)
```

## 11. 実装手順（チェックリスト）

### 11.1 環境構築
1. **プロジェクト初期化**：ディレクトリ構造作成、requirements.txt作成
2. **依存ライブラリインストール**：PyTorch, PyTorch Geometric, NumPy, Pandas等
3. **設定システム構築**：ExperimentConfigクラスの実装

### 11.2 データ処理パイプライン
1. **前処理モジュール実装**：四角→三角（規約固定）、RIMD算出、座標誤差算出
2. **データセットクラス実装**：PyTorch Dataset/DataLoader対応
3. **標準化処理**：代表寸法算出、Z標準化、スケーラ保存
4. **01_preprocessing.ipynb実行**：全データの前処理と分割

### 11.3 モデル実装
1. **ベースモデルクラス**：共通インターフェース定義
2. **MLPモデル実装**：Edge/Node Encoder, Global MLP, Node Head
3. **CVAE拡張実装**：Encoder/Decoder, 再パラメータ化トリック
4. **損失関数実装**：Huber, Laplacian, Drift, KL損失

### 11.4 学習システム
1. **トレーナークラス実装**：学習ループ、検証、早期終了
2. **メトリックス算出**：mm単位誤差、Gain、percentile等
3. **02_training.ipynb実行**：ベースラインモデル学習

### 11.5 評価システム
1. **評価モジュール実装**：詳細メトリックス、統計解析
2. **可視化モジュール実装**：ヒートマップ、学習曲線、誤差分布
3. **03_evaluation.ipynb実行**：テストデータ評価と可視化

### 11.6 実験管理と拡張
1. **CVAE化**：非VAE→CVAEへの切り替え、βウォームアップ実装
2. **アブレーションスタディ**：複数設定の一括実験
3. **GNN実装**（オプション）：GraphSAGE/GCNモデル実装
4. **04_analysis.ipynb実行**：結果比較とレポート生成

### 11.7 結果管理とデプロイ
1. **モデル保存**：最適重み、スケーラ、設定保存
2. **結果レポート**：実験結果の網羅的まとめ
3. **推論システム**：新しいデータへの適用パイプライン

## 12. ハイパラ初期値（目安）

- **ネットワーク幅**：128 基調（Enc/Head）。最終 64→2。
- **正則化**：Dropout 0.1、LayerNorm、勾配クリッピング max_norm=1.0。
- **最適化**：AdamW(lr=1e-3, wd=1e-4)、β1=0.9, β2=0.999, eps=1e-8。
- **学習率スケジューリング**：CosineAnnealingLR (T_max=epoch数) または patience=10のReduceLROnPlateau。
- **CVAE**：$d_z=8$ or 16、βウォームアップ=10epoch、最終β=0.7。
- **損失係数**：
  - λ_lap = 3e-3（ラプラシアン平滑化）
  - λ_drift = 1e-4（ドリフト抑制）
  - ※小さく始めて調整
- **バッチサイズ**：GPUメモリに応じ1-4、勾配蓄積でeffective batch sizeを調整。
- **重み初期化**：Xavier uniform (線形層)、ゼロ初期化 (最終出力層)。

## 13. 実験実行ガイド

### 13.1 基本ワークフロー
```bash
# 1. 環境構築
git clone <repo>
cd blank_deformation_rimd_cvae
pip install -r requirements.txt

# 2. データ前処理
jupyter notebook notebooks/01_preprocessing.ipynb
# -> 全セル実行

# 3. ベースライン学習
jupyter notebook notebooks/02_training.ipynb
# -> config = get_baseline_config() で実行

# 4. 評価
jupyter notebook notebooks/03_evaluation.ipynb
# -> exp_idを指定して実行

# 5. CVAE実験
# 02_training.ipynbで config = get_cvae_config() に変更して再実行

# 6. 結果比較
jupyter notebook notebooks/04_analysis.ipynb
```

### 13.2 設定カスタマイズ方法
```python
# 新しい実験設定作成
from config.experiment_configs import get_baseline_config

# ベース設定をコピー
config = get_baseline_config()

# カスタマイズ
config.exp_id = "my_experiment"
config.model.hidden_dim = 256
config.loss.lambda_lap = 5e-3
config.training.learning_rate = 5e-4

# 02_training.ipynbで使用
```

### 13.3 アブレーションスタディ実行
```python
# 複数設定の一括実行
from config.experiment_configs import get_ablation_configs

configs = get_ablation_configs()
for exp_id, config in configs.items():
    print(f"Running {exp_id}...")
    # 学習コード実行
    trainer = RIMDTrainer(config, ExperimentManager(config))
    # ...
```

### 13.4 トラブルシューティング
- **メモリエラー**: batch_sizeを減らす、gradient_accumulation使用
- **収束しない**: learning_rateを下げる、正則化強化
- **CVAE崩壊**: β_finalを下げる、latent_dimを小さく
- **過平滑化**: lambda_lapを小さく
- **学習率スケジューリング**：CosineAnnealingLR (T_max=epoch数) または patience=10のReduceLROnPlateau。
- **CVAE**：$d_z=8$ or 16、βウォームアップ=10epoch、最終β=0.7。
- **損失係数**：
  - λ_lap = 3e-3（初期値）
  - λ_drift = 1e-4（初期値）
  - ※小さく始めて調整
- **バッチサイズ**：GPUメモリに応じて1-4、勾配蓄積でeffective batch sizeを調整。
- **重み初期化**：Xavier uniform (線形層)、ゼロ初期化 (最終出力層)。

---

