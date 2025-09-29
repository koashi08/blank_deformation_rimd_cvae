import torch
import torch.nn as nn
from typing import Dict, Tuple
import logging

from .base_model import BaseRIMDModel, BaseCVAE, MLPBlock, AttentionPooling

logger = logging.getLogger(__name__)


class RIMDMLPModel(BaseRIMDModel):
    """RIMD特徴量を使用するMLPベースモデル"""

    def __init__(self, config):
        super().__init__(config)

        # 設定
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate
        self.use_layer_norm = config.use_layer_norm

        # モデル構築は特徴量次元設定後に行う
        self.edge_encoder = None
        self.node_encoder = None
        self.global_mlp = None
        self.node_head = None

    def set_feature_dimensions(self, edge_dim: int, node_dim: int):
        """特徴量次元を設定してモデルを構築"""
        super().set_feature_dimensions(edge_dim, node_dim)
        self._build_model()

    def _build_model(self):
        """モデル構築"""
        # エッジエンコーダ（mean pooling用）
        self.edge_encoder = MLPBlock(
            self.edge_dim, self.hidden_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm
        )

        # ノードエンコーダ（mean pooling用）
        self.node_encoder = MLPBlock(
            self.node_dim, self.hidden_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm
        )

        # グローバル特徴量MLP
        self.global_mlp = MLPBlock(
            self.hidden_dim * 2,  # edge + node features
            self.hidden_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm
        )

        # ノード出力ヘッド（共有）
        head_layers = []
        current_dim = self.node_dim + self.hidden_dim  # node features + global context

        for i in range(self.num_layers - 1):
            head_layers.append(
                MLPBlock(
                    current_dim, self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    use_layer_norm=self.use_layer_norm
                )
            )
            current_dim = self.hidden_dim

        # 最終層
        head_layers.append(nn.Linear(current_dim, self.output_dim))

        self.node_head = nn.Sequential(*head_layers)

        # CVAEの場合、追加コンポーネントを構築
        if self.use_cvae:
            self._build_cvae_components()

    def _build_cvae_components(self):
        """CVAE用コンポーネントを構築"""
        # エンコーダ（target → latent）
        self.target_encoder = nn.Sequential(
            MLPBlock(self.output_dim, 32, self.dropout_rate, self.use_layer_norm),
            MLPBlock(32, 64, self.dropout_rate, self.use_layer_norm)
        )

        # μ, logvarを出力
        self.mu_head = nn.Linear(64 + self.hidden_dim, self.config.latent_dim)
        self.logvar_head = nn.Linear(64 + self.hidden_dim, self.config.latent_dim)

        # デコーダ（z + context → output）
        self.context_mlp = MLPBlock(
            self.config.latent_dim + self.hidden_dim,
            self.hidden_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向き計算"""
        if self.use_cvae:
            return self._forward_cvae(batch)
        else:
            return self._forward_mlp(batch)

    def _forward_mlp(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """非VAEの前向き計算"""
        # 特徴量エンコード
        edge_encoded = self.edge_encoder(batch['edge_features'])
        node_encoded = self.node_encoder(batch['node_features'])

        # グローバル特徴量（mean pooling）
        if 'batch' in batch:
            # バッチ処理
            edge_global = self._batch_mean_pool(edge_encoded, batch['batch'], batch['edge_index'])
            node_global = self._batch_mean_pool(node_encoded, batch['batch'])
        else:
            # 単一グラフ
            edge_global = edge_encoded.mean(dim=0, keepdim=True)
            node_global = node_encoded.mean(dim=0, keepdim=True)

        # グローバル特徴量を結合
        global_features = torch.cat([edge_global, node_global], dim=-1)
        global_context = self.global_mlp(global_features)

        # 各ノードにグローバル特徴量を結合
        if 'batch' in batch:
            # バッチインデックスに従ってグローバル特徴量を展開
            expanded_context = global_context[batch['batch']]
        else:
            expanded_context = global_context.expand(batch['node_features'].shape[0], -1)

        # ノード特徴量とグローバル特徴量を結合
        combined_features = torch.cat([batch['node_features'], expanded_context], dim=-1)

        # 出力計算
        output = self.node_head(combined_features)

        return {'output': output}

    def _forward_cvae(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """CVAEの前向き計算"""
        # グローバル特徴量計算（非VAEと同様）
        edge_encoded = self.edge_encoder(batch['edge_features'])
        node_encoded = self.node_encoder(batch['node_features'])

        if 'batch' in batch:
            edge_global = self._batch_mean_pool(edge_encoded, batch['batch'], batch['edge_index'])
            node_global = self._batch_mean_pool(node_encoded, batch['batch'])
        else:
            edge_global = edge_encoded.mean(dim=0, keepdim=True)
            node_global = node_encoded.mean(dim=0, keepdim=True)

        global_features = torch.cat([edge_global, node_global], dim=-1)
        global_context = self.global_mlp(global_features)

        if self.training and 'target' in batch:
            # 学習時：エンコーダとデコーダ
            target = batch['target']

            # ターゲットエンコード（グラフレベル）
            if 'batch' in batch:
                target_encoded = self._batch_mean_pool(
                    self.target_encoder(target), batch['batch']
                )
            else:
                target_encoded = self.target_encoder(target).mean(dim=0, keepdim=True)

            # μ, logvar計算
            encoder_input = torch.cat([target_encoded, global_context], dim=-1)
            mu = self.mu_head(encoder_input)
            logvar = self.logvar_head(encoder_input)

            # 再パラメータ化
            z = self.reparameterize(mu, logvar)

            # デコード
            context_with_z = torch.cat([z, global_context], dim=-1)
            enhanced_context = self.context_mlp(context_with_z)

            # 各ノードに展開
            if 'batch' in batch:
                expanded_context = enhanced_context[batch['batch']]
            else:
                expanded_context = enhanced_context.expand(batch['node_features'].shape[0], -1)

            combined_features = torch.cat([batch['node_features'], expanded_context], dim=-1)
            output = self.node_head(combined_features)

            return {
                'output': output,
                'mu': mu,
                'logvar': logvar,
                'z': z
            }
        else:
            # 推論時：デコーダのみ
            batch_size = global_context.shape[0]
            device = global_context.device
            z = torch.randn(batch_size, self.config.latent_dim, device=device)

            context_with_z = torch.cat([z, global_context], dim=-1)
            enhanced_context = self.context_mlp(context_with_z)

            if 'batch' in batch:
                expanded_context = enhanced_context[batch['batch']]
            else:
                expanded_context = enhanced_context.expand(batch['node_features'].shape[0], -1)

            combined_features = torch.cat([batch['node_features'], expanded_context], dim=-1)
            output = self.node_head(combined_features)

            return {
                'output': output,
                'z': z
            }

    def _batch_mean_pool(self, x: torch.Tensor, batch: torch.Tensor,
                        edge_index: torch.Tensor = None) -> torch.Tensor:
        """バッチ処理でのmean pooling"""
        if edge_index is not None:
            # エッジ特徴量の場合、エッジをノードにマップしてからプール
            edge_to_node_batch = batch[edge_index[0]]  # エッジの始点ノードのバッチインデックス
            unique_batches = torch.unique(edge_to_node_batch)
        else:
            # ノード特徴量の場合
            unique_batches = torch.unique(batch)

        pooled_list = []
        for batch_idx in unique_batches:
            if edge_index is not None:
                mask = (edge_to_node_batch == batch_idx)
            else:
                mask = (batch == batch_idx)

            pooled = x[mask].mean(dim=0)
            pooled_list.append(pooled)

        return torch.stack(pooled_list)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """再パラメータ化トリック"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def _initialize_output_layer(self):
        """最終出力層の初期化"""
        if hasattr(self, 'node_head') and self.node_head is not None:
            # 最終層をゼロ初期化
            final_layer = None
            for module in self.node_head.modules():
                if isinstance(module, nn.Linear):
                    final_layer = module

            if final_layer is not None:
                nn.init.zeros_(final_layer.weight)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)


class RIMDMLPModelWithAttention(RIMDMLPModel):
    """アテンション付きMLPモデル"""

    def _build_model(self):
        """アテンションプーリング付きモデル構築"""
        super()._build_model()

        # アテンションプーリングに置き換え
        self.edge_attention_pool = AttentionPooling(self.hidden_dim)
        self.node_attention_pool = AttentionPooling(self.hidden_dim)

    def _forward_mlp(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """アテンション付き前向き計算"""
        edge_encoded = self.edge_encoder(batch['edge_features'])
        node_encoded = self.node_encoder(batch['node_features'])

        # アテンションプーリング
        if 'batch' in batch:
            # バッチ処理のためのバッチインデックス調整が必要
            edge_batch = batch['batch'][batch['edge_index'][0]]
            edge_global = self.edge_attention_pool(edge_encoded, edge_batch)
            node_global = self.node_attention_pool(node_encoded, batch['batch'])
        else:
            edge_global = self.edge_attention_pool(edge_encoded)
            node_global = self.node_attention_pool(node_encoded)

        # 以降は基本MLPと同様
        global_features = torch.cat([edge_global, node_global], dim=-1)
        global_context = self.global_mlp(global_features)

        if 'batch' in batch:
            expanded_context = global_context[batch['batch']]
        else:
            expanded_context = global_context.expand(batch['node_features'].shape[0], -1)

        combined_features = torch.cat([batch['node_features'], expanded_context], dim=-1)
        output = self.node_head(combined_features)

        return {'output': output}