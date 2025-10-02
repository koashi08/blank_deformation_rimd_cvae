from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json
from pathlib import Path


@dataclass
class DataConfig:
    """データ関連設定"""
    data_root: str = "../data/raw_data"  # 実データ（df_node.pkl, df_element.pkl）のディレクトリ
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

    # 特徴量設定
    use_position_features: bool = True
    use_phi_mean_features: bool = True


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
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # スケジューリング
    scheduler: str = "cosine"  # "cosine", "reduce_on_plateau", "none"
    scheduler_patience: int = 10  # ReduceLROnPlateauの場合

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

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書に変換"""
        return asdict(self)

    def save(self, path: Path):
        """設定をJSONで保存"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """辞書から設定を作成"""
        return cls(
            exp_id=data.get('exp_id', 'exp_001'),
            description=data.get('description', ''),
            data=DataConfig(**data.get('data', {})),
            model=ModelConfig(**data.get('model', {})),
            loss=LossConfig(**data.get('loss', {})),
            training=TrainingConfig(**data.get('training', {}))
        )

    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """設定をJSONから読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def update_exp_id_with_timestamp(self):
        """実験IDにタイムスタンプを追加"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.exp_id.endswith(f"_{timestamp}"):
            self.exp_id = f"{self.exp_id}_{timestamp}"