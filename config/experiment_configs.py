from typing import Dict
from .base_config import ExperimentConfig, ModelConfig, LossConfig, TrainingConfig


def get_baseline_config() -> ExperimentConfig:
    """ベースライン実験設定（非VAE MLP）"""
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


def get_large_model_config() -> ExperimentConfig:
    """大きなモデルの実験設定"""
    config = get_baseline_config()
    config.exp_id = "large_mlp"
    config.description = "大規模MLPモデル"
    config.model.hidden_dim = 256
    config.model.num_layers = 4
    return config


def get_ablation_configs() -> Dict[str, ExperimentConfig]:
    """アブレーションスタディ設定"""
    configs = {}

    # 損失関数比較
    for loss_type in ["mse", "huber", "smooth_l1"]:
        config = get_baseline_config()
        config.exp_id = f"ablation_loss_{loss_type}"
        config.description = f"損失関数アブレーション: {loss_type}"
        config.loss.loss_type = loss_type
        configs[config.exp_id] = config

    # 正則化係数比較 (Laplacian)
    for lambda_lap in [1e-3, 3e-3, 5e-3]:
        config = get_baseline_config()
        config.exp_id = f"ablation_lambda_lap_{lambda_lap:.0e}"
        config.description = f"Laplacian正則化アブレーション: {lambda_lap}"
        config.loss.lambda_lap = lambda_lap
        configs[config.exp_id] = config

    # モデルサイズ比較
    for hidden_dim in [64, 128, 256]:
        config = get_baseline_config()
        config.exp_id = f"ablation_hidden_{hidden_dim}"
        config.description = f"隠れ層サイズアブレーション: {hidden_dim}"
        config.model.hidden_dim = hidden_dim
        configs[config.exp_id] = config

    # 特徴量アブレーション
    feature_configs = [
        ("no_pos", "座標特徴量なし", {"use_position_features": False}),
        ("no_phi_mean", "φ平均特徴量なし", {"use_phi_mean_features": False}),
        ("minimal", "最小特徴量", {"use_position_features": False, "use_phi_mean_features": False})
    ]

    for suffix, desc, feature_kwargs in feature_configs:
        config = get_baseline_config()
        config.exp_id = f"ablation_features_{suffix}"
        config.description = f"特徴量アブレーション: {desc}"
        for key, value in feature_kwargs.items():
            setattr(config.model, key, value)
        configs[config.exp_id] = config

    return configs


def get_cvae_ablation_configs() -> Dict[str, ExperimentConfig]:
    """CVAE関連のアブレーション設定"""
    configs = {}

    # 潜在次元比較
    for latent_dim in [4, 8, 16]:
        config = get_cvae_config()
        config.exp_id = f"cvae_latent_{latent_dim}"
        config.description = f"CVAE潜在次元: {latent_dim}"
        config.model.latent_dim = latent_dim
        configs[config.exp_id] = config

    # βファイナル値比較
    for beta_final in [0.5, 0.7, 1.0]:
        config = get_cvae_config()
        config.exp_id = f"cvae_beta_{beta_final:.1f}"
        config.description = f"CVAEβ最終値: {beta_final}"
        config.loss.beta_final = beta_final
        configs[config.exp_id] = config

    return configs


def get_production_config() -> ExperimentConfig:
    """本格実験用設定（長時間学習）"""
    config = get_cvae_config()
    config.exp_id = "production_cvae"
    config.description = "本格実験：CVAE長時間学習"
    config.training.max_epochs = 500
    config.training.early_stopping_patience = 50
    config.model.hidden_dim = 256
    config.model.num_layers = 4
    return config