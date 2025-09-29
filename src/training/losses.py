import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RIMDLoss(nn.Module):
    """RIMD予測のための複合損失関数"""

    def __init__(self, config):
        super().__init__()
        self.config = config.loss

        # 基本損失関数
        if self.config.loss_type == "mse":
            self.base_loss = nn.MSELoss(reduction='none')
        elif self.config.loss_type == "huber":
            self.base_loss = nn.HuberLoss(
                delta=self.config.huber_delta,
                reduction='none'
            )
        elif self.config.loss_type == "smooth_l1":
            self.base_loss = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        # 損失係数
        self.lambda_lap = self.config.lambda_lap
        self.lambda_drift = self.config.lambda_drift
        self.lambda_arap = self.config.lambda_arap

        # CVAE関連
        self.beta_warmup_epochs = self.config.beta_warmup_epochs
        self.beta_final = self.config.beta_final

        logger.info(f"Initialized {self.config.loss_type} loss with "
                   f"λ_lap={self.lambda_lap}, λ_drift={self.lambda_drift}")

    def forward(self, predictions: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor],
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        複合損失を計算

        Args:
            predictions: モデルの予測結果
                - output: [total_nodes, 2] 予測変位
                - (CVAE) mu, logvar, z
            batch: バッチデータ
                - target: [total_nodes, 2] 真値変位
                - edge_index: [2, total_edges] エッジインデックス
                - batch: [total_nodes] バッチインデックス
            epoch: 現在のエポック（βウォームアップ用）

        Returns:
            損失辞書
        """
        losses = {}

        # 基本再構成損失
        pred_output = predictions['output']
        target = batch['target']

        reconstruction_loss = self.base_loss(pred_output, target)
        losses['reconstruction'] = reconstruction_loss.mean()

        # 幾何学的正則化損失
        geom_losses = self.compute_geometric_regularization(
            pred_output, batch['edge_index'], batch.get('batch')
        )
        losses.update(geom_losses)

        # CVAE損失
        if 'mu' in predictions and 'logvar' in predictions:
            kl_loss = self.compute_kl_loss(predictions['mu'], predictions['logvar'])
            beta = self.compute_beta(epoch)
            losses['kl'] = kl_loss
            losses['beta'] = torch.tensor(beta)

        # 総損失
        total_loss = losses['reconstruction']
        total_loss += self.lambda_lap * losses['laplacian']
        total_loss += self.lambda_drift * losses['drift']

        if self.lambda_arap > 0:
            total_loss += self.lambda_arap * losses.get('arap', 0.0)

        if 'kl' in losses:
            beta = losses['beta'].item()
            total_loss += beta * losses['kl']

        losses['total'] = total_loss

        return losses

    def compute_geometric_regularization(self, pred_output: torch.Tensor,
                                       edge_index: torch.Tensor,
                                       batch_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """幾何学的正則化損失を計算"""
        losses = {}

        # ラプラシアン平滑化
        laplacian_loss = self.compute_laplacian_smoothing(pred_output, edge_index)
        losses['laplacian'] = laplacian_loss

        # ドリフト抑制
        drift_loss = self.compute_drift_suppression(pred_output, batch_indices)
        losses['drift'] = drift_loss

        # 2D-ARAP損失（オプション）
        if self.lambda_arap > 0:
            arap_loss = self.compute_arap_loss(pred_output, edge_index)
            losses['arap'] = arap_loss

        return losses

    def compute_laplacian_smoothing(self, pred_output: torch.Tensor,
                                  edge_index: torch.Tensor) -> torch.Tensor:
        """ラプラシアン平滑化損失"""
        i, j = edge_index
        diff = pred_output[i] - pred_output[j]  # [num_edges, 2]
        laplacian_loss = torch.sum(diff ** 2)
        return laplacian_loss / edge_index.shape[1]  # エッジ数で正規化

    def compute_drift_suppression(self, pred_output: torch.Tensor,
                                batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ドリフト抑制損失（一様平行移動の抑制）"""
        if batch_indices is not None:
            # バッチ処理：各グラフごとに平均変位を計算
            drift_loss = 0.0
            for batch_idx in torch.unique(batch_indices):
                mask = (batch_indices == batch_idx)
                graph_output = pred_output[mask]
                mean_displacement = torch.mean(graph_output, dim=0)
                drift_loss += torch.sum(mean_displacement ** 2)
            return drift_loss / len(torch.unique(batch_indices))
        else:
            # 単一グラフ
            mean_displacement = torch.mean(pred_output, dim=0)
            return torch.sum(mean_displacement ** 2)

    def compute_arap_loss(self, pred_output: torch.Tensor,
                         edge_index: torch.Tensor) -> torch.Tensor:
        """2D-ARAP風損失（簡易版）"""
        # エッジベクトル
        i, j = edge_index
        edge_vectors = pred_output[j] - pred_output[i]  # [num_edges, 2]

        # エッジ長の変化を抑制（等方的変形を促進）
        edge_lengths = torch.norm(edge_vectors, dim=1)
        length_variance = torch.var(edge_lengths)

        return length_variance

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KLダイバージェンス損失"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

    def compute_beta(self, epoch: int) -> float:
        """βウォームアップスケジュール"""
        if epoch < self.beta_warmup_epochs:
            beta = (epoch / self.beta_warmup_epochs) * self.beta_final
        else:
            beta = self.beta_final
        return beta


class HeteroscedasticLoss(nn.Module):
    """ヘテロスケダスティック損失（平均と分散を同時予測）"""

    def __init__(self, config):
        super().__init__()
        self.config = config.loss

    def forward(self, predictions: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        ヘテロスケダスティック損失を計算

        Args:
            predictions: 予測結果
                - mu: [total_nodes, 2] 予測平均
                - logvar: [total_nodes, 2] 予測対数分散
            batch: バッチデータ
                - target: [total_nodes, 2] 真値

        Returns:
            損失辞書
        """
        mu = predictions['mu']
        logvar = predictions['logvar']
        target = batch['target']

        # 対数尤度損失（ガウシアン仮定）
        precision = torch.exp(-logvar)
        nll_loss = 0.5 * precision * (target - mu) ** 2 + 0.5 * logvar
        reconstruction_loss = nll_loss.mean()

        # 分散正則化（過度に小さい分散を防ぐ）
        var_reg = torch.mean(torch.exp(logvar))

        losses = {
            'reconstruction': reconstruction_loss,
            'variance_reg': var_reg,
            'total': reconstruction_loss + 0.01 * var_reg
        }

        return losses


class FocalLoss(nn.Module):
    """Focal Loss（困難サンプルに焦点を当てる）"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal Lossを計算

        Args:
            predictions: [N, D] 予測値
            targets: [N, D] 真値

        Returns:
            focal loss
        """
        mse_loss = F.mse_loss(predictions, targets, reduction='none')

        # 各サンプルの平均二乗誤差
        sample_mse = mse_loss.mean(dim=1)

        # 正規化された誤差（0-1の範囲に）
        normalized_error = torch.sigmoid(sample_mse)

        # Focal weight
        focal_weight = self.alpha * (normalized_error ** self.gamma)

        # 重み付き損失
        weighted_loss = focal_weight.unsqueeze(1) * mse_loss

        return weighted_loss.mean()


class MultiScaleLoss(nn.Module):
    """多解像度損失（粗い〜細かいスケールで評価）"""

    def __init__(self, scales: list = [1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.base_loss = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                coords: torch.Tensor) -> torch.Tensor:
        """
        多解像度で損失を計算

        Args:
            predictions: [N, 2] 予測変位
            targets: [N, 2] 真値変位
            coords: [N, 2] 座標（スケーリング用）

        Returns:
            multi-scale loss
        """
        total_loss = 0.0

        for scale in self.scales:
            # 座標をスケーリング
            scaled_coords = coords * scale

            # 予測と真値もスケールに応じて調整
            scaled_pred = predictions * scale
            scaled_target = targets * scale

            # スケール別損失
            scale_loss = self.base_loss(scaled_pred, scaled_target)
            total_loss += scale_loss

        return total_loss / len(self.scales)


def create_loss_function(config):
    """設定に基づいて損失関数を作成"""
    loss_type = getattr(config.loss, 'loss_type', 'huber')

    if loss_type == 'heteroscedastic':
        return HeteroscedasticLoss(config)
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'multiscale':
        return MultiScaleLoss()
    else:
        return RIMDLoss(config)


class LossTracker:
    """損失の履歴を追跡するヘルパークラス"""

    def __init__(self):
        self.history = {}

    def update(self, losses: Dict[str, torch.Tensor]):
        """損失を更新"""
        for key, value in losses.items():
            if key not in self.history:
                self.history[key] = []

            if isinstance(value, torch.Tensor):
                self.history[key].append(value.item())
            else:
                self.history[key].append(value)

    def get_averages(self, window: int = 10) -> Dict[str, float]:
        """直近の平均を取得"""
        averages = {}
        for key, values in self.history.items():
            if len(values) > 0:
                recent_values = values[-window:] if len(values) >= window else values
                averages[key] = sum(recent_values) / len(recent_values)
        return averages

    def reset(self):
        """履歴をリセット"""
        self.history = {}