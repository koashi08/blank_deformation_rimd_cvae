import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class BaseRIMDModel(nn.Module, ABC):
    """RIMD特徴量を使用するモデルのベースクラス"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.use_cvae = config.use_cvae

        # 特徴量次元（データローダから設定される）
        self.edge_dim = None
        self.node_dim = None

        # 出力次元
        self.output_dim = 2  # (Δx, Δy)

    def set_feature_dimensions(self, edge_dim: int, node_dim: int):
        """特徴量次元を設定"""
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        logger.info(f"Feature dimensions set: edge_dim={edge_dim}, node_dim={node_dim}")

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向き計算

        Args:
            batch: バッチデータ
                - edge_features: [total_edges, edge_dim]
                - node_features: [total_nodes, node_dim]
                - edge_index: [2, total_edges]
                - batch: [total_nodes] バッチインデックス

        Returns:
            出力辞書
                - output: [total_nodes, output_dim] 予測値
                - (CVAEの場合) mu, logvar, z
        """
        pass

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """推論用の前向き計算"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            return outputs['output']

    def get_model_summary(self) -> Dict[str, Any]:
        """モデルの要約情報を取得"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': self.model_type,
            'use_cvae': self.use_cvae,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'edge_dim': self.edge_dim,
            'node_dim': self.node_dim,
            'output_dim': self.output_dim
        }

    def initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform初期化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # 最終出力層はゼロ初期化
        self._initialize_output_layer()

    @abstractmethod
    def _initialize_output_layer(self):
        """最終出力層の初期化（サブクラスで実装）"""
        pass


class BaseCVAE(BaseRIMDModel):
    """CVAEの基底クラス"""

    def __init__(self, config):
        super().__init__(config)
        self.latent_dim = config.latent_dim

        # CVAEコンポーネント（サブクラスで実装）
        self.encoder = None
        self.decoder = None

    def encode(self, target: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """エンコーダ（学習時のみ使用）"""
        if self.encoder is None:
            raise NotImplementedError("Encoder not implemented")
        return self.encoder(target, condition)

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """デコーダ"""
        if self.decoder is None:
            raise NotImplementedError("Decoder not implemented")
        return self.decoder(z, condition)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """再パラメータ化トリック"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """潜在変数をサンプリング（推論時）"""
        return torch.randn(batch_size, self.latent_dim, device=device)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """CVAEの前向き計算"""
        # 条件特徴量を構築
        condition = self._build_condition(batch)

        if self.training and 'target' in batch:
            # 学習時：エンコーダとデコーダの両方を使用
            target = batch['target']
            mu, logvar = self.encode(target, condition)
            z = self.reparameterize(mu, logvar)
            output = self.decode(z, condition)

            return {
                'output': output,
                'mu': mu,
                'logvar': logvar,
                'z': z
            }
        else:
            # 推論時：デコーダのみ使用
            batch_size = self._get_batch_size(batch)
            device = next(self.parameters()).device
            z = self.sample_latent(batch_size, device)
            output = self.decode(z, condition)

            return {
                'output': output,
                'z': z
            }

    @abstractmethod
    def _build_condition(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """条件特徴量を構築（サブクラスで実装）"""
        pass

    @abstractmethod
    def _get_batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """バッチサイズを取得（サブクラスで実装）"""
        pass


class MLPBlock(nn.Module):
    """MLPブロック（LayerNorm + Dropout付き）"""

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.1,
                 use_layer_norm: bool = True, activation: str = "gelu"):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

        if use_layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()

        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class AttentionPooling(nn.Module):
    """アテンションベースのプーリング"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [N, D] 特徴量
            batch: [N] バッチインデックス

        Returns:
            [B, D] プールされた特徴量
        """
        if batch is None:
            # 単一グラフの場合
            attention_weights = torch.softmax(self.attention(x), dim=0)
            return torch.sum(attention_weights * x, dim=0, keepdim=True)
        else:
            # バッチ処理
            attention_scores = self.attention(x).squeeze(-1)
            output_list = []

            for batch_idx in torch.unique(batch):
                mask = (batch == batch_idx)
                masked_scores = attention_scores[mask]
                masked_features = x[mask]

                weights = torch.softmax(masked_scores, dim=0).unsqueeze(-1)
                pooled = torch.sum(weights * masked_features, dim=0)
                output_list.append(pooled)

            return torch.stack(output_list)


def create_model(config, edge_dim: int, node_dim: int) -> BaseRIMDModel:
    """設定に基づいてモデルを作成"""
    if config.model_type == "mlp":
        from .mlp_model import RIMDMLPModel
        model = RIMDMLPModel(config)
    elif config.model_type == "gnn":
        from .gnn_model import RIMDGNNModel
        model = RIMDGNNModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # 特徴量次元設定
    model.set_feature_dimensions(edge_dim, node_dim)

    # 重み初期化
    model.initialize_weights()

    logger.info(f"Created {config.model_type} model")
    logger.info(f"Model summary: {model.get_model_summary()}")

    return model