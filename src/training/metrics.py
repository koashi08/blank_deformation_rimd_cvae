import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class RIMDMetrics:
    """RIMD予測の評価メトリクス"""

    def __init__(self, config):
        self.config = config

    def compute_metrics(self, predictions: torch.Tensor, batch: Dict[str, torch.Tensor],
                       compute_all: bool = False) -> Dict[str, float]:
        """
        メトリクスを計算

        Args:
            predictions: [N, 2] 予測変位（標準化済み）
            batch: バッチデータ
            compute_all: 全メトリクスを計算するか

        Returns:
            メトリクス辞書
        """
        # 標準化を元に戻す
        pred_mm = self._denormalize_predictions(predictions, batch)
        target_mm = self._denormalize_targets(batch['target'], batch)

        metrics = {}

        # 基本誤差メトリクス
        euclidean_errors = torch.norm(pred_mm - target_mm, dim=1)

        metrics['rmse_mm'] = torch.sqrt(torch.mean(euclidean_errors ** 2)).item()
        metrics['mae_mm'] = torch.mean(euclidean_errors).item()
        metrics['median_error_mm'] = torch.median(euclidean_errors).item()

        if compute_all:
            # パーセンタイル誤差
            metrics['p90_error_mm'] = torch.quantile(euclidean_errors, 0.9).item()
            metrics['p95_error_mm'] = torch.quantile(euclidean_errors, 0.95).item()
            metrics['max_error_mm'] = torch.max(euclidean_errors).item()

            # 座標別誤差
            x_errors = torch.abs(pred_mm[:, 0] - target_mm[:, 0])
            y_errors = torch.abs(pred_mm[:, 1] - target_mm[:, 1])

            metrics['mae_x_mm'] = torch.mean(x_errors).item()
            metrics['mae_y_mm'] = torch.mean(y_errors).item()
            metrics['max_x_mm'] = torch.max(x_errors).item()
            metrics['max_y_mm'] = torch.max(y_errors).item()

            # Gainの計算（改善率）
            if 'coords_1step' in batch and 'coords_nv' in batch:
                gain = self._compute_gain(pred_mm, target_mm, batch)
                metrics['gain_percent'] = gain

        return metrics

    def _denormalize_predictions(self, predictions: torch.Tensor,
                               batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """予測値の標準化を元に戻してmm単位に"""
        # まず標準化スケーラを適用（実装されている場合）
        if hasattr(self.config, 'scalers') and self.config.scalers.get('target_scaler'):
            # 標準化を逆変換（実際の実装では適用が必要）
            denorm_pred = predictions  # TODO: target_scaler.inverse_transform()
        else:
            denorm_pred = predictions

        # 代表寸法スケールを逆変換
        if 'representative_scale' in batch:
            denorm_pred = denorm_pred * batch['representative_scale'].unsqueeze(-1)

        return denorm_pred

    def _denormalize_targets(self, targets: torch.Tensor,
                           batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ターゲットの標準化を元に戻してmm単位に"""
        # まず標準化スケーラを適用（実装されている場合）
        if hasattr(self.config, 'scalers') and self.config.scalers.get('target_scaler'):
            # 標準化を逆変換（実際の実装では適用が必要）
            denorm_target = targets  # TODO: target_scaler.inverse_transform()
        else:
            denorm_target = targets

        # 代表寸法スケールを逆変換
        if 'representative_scale' in batch:
            denorm_target = denorm_target * batch['representative_scale'].unsqueeze(-1)

        return denorm_target

    def _compute_gain(self, pred_mm: torch.Tensor, target_mm: torch.Tensor,
                     batch: Dict[str, torch.Tensor]) -> float:
        """
        Gain（改善率）をBaseline-0基準で計算

        Baseline-0: 1step解析結果をそのまま使用（補正なし）
        Gain = (||X_Baseline-0 - X_nv|| - ||X_pred - X_nv||) / ||X_Baseline-0 - X_nv|| * 100%
             = (||X_1step - X_nv|| - ||X_pred - X_nv||) / ||X_1step - X_nv|| * 100%

        Args:
            pred_mm: 予測変位 [N, 2] (mm単位)
            target_mm: 真値変位 [N, 2] (mm単位)
            batch: バッチデータ（座標情報含む）

        Returns:
            float: Baseline-0からの改善率（%）
                正の値: 予測手法がBaseline-0より優れている
                負の値: 予測手法がBaseline-0より劣っている
                0: Baseline-0と同等
        """
        if 'coords_1step' not in batch or 'coords_nv' not in batch:
            return 0.0

        # 座標取得
        coords_1step = batch['coords_1step']
        coords_nv = batch['coords_nv']

        # 予測座標を計算
        coords_pred = coords_1step + pred_mm

        # 誤差計算
        error_1step = torch.norm(coords_1step - coords_nv, dim=1)
        error_pred = torch.norm(coords_pred - coords_nv, dim=1)

        # Gain計算
        improvement = error_1step - error_pred
        gain = torch.mean(improvement / (error_1step + 1e-8)) * 100

        return gain.item()

    def compute_detailed_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                               coords_1step: np.ndarray, coords_nv: np.ndarray,
                               case_ids: Optional[List[str]] = None) -> Dict[str, any]:
        """
        詳細メトリクスを計算（バッチ全体またはテストセット）

        Args:
            predictions: [N, 2] 予測変位
            targets: [N, 2] 真値変位
            coords_1step: [N, 2] 1step座標
            coords_nv: [N, 2] 逐次解析座標
            case_ids: ケースID（オプション）

        Returns:
            詳細メトリクス辞書
        """
        # 基本統計
        pred_coords = coords_1step + predictions
        euclidean_errors = np.linalg.norm(pred_coords - coords_nv, axis=1)

        metrics = {
            # 全体統計
            'rmse_mm': np.sqrt(np.mean(euclidean_errors ** 2)),
            'mae_mm': np.mean(euclidean_errors),
            'median_error_mm': np.median(euclidean_errors),
            'p90_error_mm': np.percentile(euclidean_errors, 90),
            'p95_error_mm': np.percentile(euclidean_errors, 95),
            'p99_error_mm': np.percentile(euclidean_errors, 99),
            'max_error_mm': np.max(euclidean_errors),

            # 座標別統計
            'mae_x_mm': np.mean(np.abs(predictions[:, 0] - targets[:, 0])),
            'mae_y_mm': np.mean(np.abs(predictions[:, 1] - targets[:, 1])),
            'rmse_x_mm': np.sqrt(np.mean((predictions[:, 0] - targets[:, 0]) ** 2)),
            'rmse_y_mm': np.sqrt(np.mean((predictions[:, 1] - targets[:, 1]) ** 2)),

            # 改善率
            'gain_percent': self._compute_gain_numpy(
                predictions, coords_1step, coords_nv
            ),

            # エラー分布
            'error_distribution': {
                'mean': np.mean(euclidean_errors),
                'std': np.std(euclidean_errors),
                'min': np.min(euclidean_errors),
                'max': np.max(euclidean_errors),
                'quartiles': np.percentile(euclidean_errors, [25, 50, 75])
            }
        }

        # ケース別統計（提供された場合）
        if case_ids is not None:
            metrics['case_statistics'] = self._compute_case_statistics(
                euclidean_errors, case_ids
            )

        return metrics

    def _compute_gain_numpy(self, predictions: np.ndarray, coords_1step: np.ndarray,
                          coords_nv: np.ndarray) -> float:
        """
        NumPy配列でのGain計算（Baseline-0基準）

        Baseline-0: 1step解析結果をそのまま使用
        Gain = (||X_Baseline-0 - X_nv|| - ||X_pred - X_nv||) / ||X_Baseline-0 - X_nv|| * 100%
        """
        pred_coords = coords_1step + predictions
        error_1step = np.linalg.norm(coords_1step - coords_nv, axis=1)
        error_pred = np.linalg.norm(pred_coords - coords_nv, axis=1)

        improvement = error_1step - error_pred
        gain = np.mean(improvement / (error_1step + 1e-8)) * 100

        return float(gain)

    def _compute_case_statistics(self, errors: np.ndarray,
                               case_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """ケース別統計を計算"""
        unique_cases = list(set(case_ids))
        case_stats = {}

        for case_id in unique_cases:
            case_mask = np.array([cid == case_id for cid in case_ids])
            case_errors = errors[case_mask]

            if len(case_errors) > 0:
                case_stats[case_id] = {
                    'mean_error': float(np.mean(case_errors)),
                    'max_error': float(np.max(case_errors)),
                    'median_error': float(np.median(case_errors)),
                    'p95_error': float(np.percentile(case_errors, 95)),
                    'num_nodes': len(case_errors)
                }

        return case_stats

    def create_error_analysis_report(self, metrics: Dict[str, any]) -> str:
        """エラー解析レポートを作成"""
        report = []
        report.append("=== RIMD Model Error Analysis Report ===\n")

        # 全体統計
        report.append("Overall Statistics:")
        report.append(f"  RMSE: {metrics['rmse_mm']:.2f} mm")
        report.append(f"  MAE: {metrics['mae_mm']:.2f} mm")
        report.append(f"  Median Error: {metrics['median_error_mm']:.2f} mm")
        report.append(f"  P90 Error: {metrics['p90_error_mm']:.2f} mm")
        report.append(f"  P95 Error: {metrics['p95_error_mm']:.2f} mm")
        report.append(f"  Max Error: {metrics['max_error_mm']:.2f} mm")
        if metrics.get('gain_percent'):
            report.append(f"  Gain (vs Baseline-0): {metrics['gain_percent']:.1f}%\n")
        else:
            report.append("")

        # 座標別統計
        report.append("Coordinate-wise Statistics:")
        report.append(f"  X-direction MAE: {metrics['mae_x_mm']:.2f} mm")
        report.append(f"  Y-direction MAE: {metrics['mae_y_mm']:.2f} mm")
        report.append(f"  X-direction RMSE: {metrics['rmse_x_mm']:.2f} mm")
        report.append(f"  Y-direction RMSE: {metrics['rmse_y_mm']:.2f} mm\n")

        # エラー分布
        if 'error_distribution' in metrics:
            dist = metrics['error_distribution']
            report.append("Error Distribution:")
            report.append(f"  Mean ± Std: {dist['mean']:.2f} ± {dist['std']:.2f} mm")
            report.append(f"  Range: {dist['min']:.2f} - {dist['max']:.2f} mm")
            q25, q50, q75 = dist['quartiles']
            report.append(f"  Quartiles (25%, 50%, 75%): {q25:.2f}, {q50:.2f}, {q75:.2f} mm\n")

        return '\n'.join(report)


class MetricsTracker:
    """メトリクスの追跡とログ記録"""

    def __init__(self):
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }

    def add_metrics(self, metrics: Dict[str, float], split: str, epoch: int):
        """メトリクスを追加"""
        metrics_with_epoch = {'epoch': epoch, **metrics}
        if split in self.metrics_history:
            self.metrics_history[split].append(metrics_with_epoch)

    def get_best_metrics(self, split: str, metric_key: str = 'rmse_mm',
                        minimize: bool = True) -> Dict[str, float]:
        """最良のメトリクスを取得"""
        if split not in self.metrics_history or not self.metrics_history[split]:
            return {}

        metrics_list = self.metrics_history[split]

        if minimize:
            best_metrics = min(metrics_list, key=lambda x: x.get(metric_key, float('inf')))
        else:
            best_metrics = max(metrics_list, key=lambda x: x.get(metric_key, float('-inf')))

        return best_metrics

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """メトリクスの要約を取得"""
        summary = {}

        for split in ['train', 'val', 'test']:
            if self.metrics_history[split]:
                last_metrics = self.metrics_history[split][-1]
                best_metrics = self.get_best_metrics(split, 'rmse_mm', minimize=True)

                summary[split] = {
                    'last_epoch': last_metrics.get('epoch', -1),
                    'last_rmse': last_metrics.get('rmse_mm', 0.0),
                    'best_epoch': best_metrics.get('epoch', -1),
                    'best_rmse': best_metrics.get('rmse_mm', 0.0),
                    'best_gain': best_metrics.get('gain_percent', 0.0)
                }

        return summary