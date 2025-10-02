import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# スタイル設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)


class RIMDVisualizer:
    """RIMD予測結果の可視化を行うクラス"""

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(exist_ok=True)

    def plot_prediction_accuracy(self, predictions: np.ndarray, targets: np.ndarray,
                               case_ids: Optional[List[str]] = None,
                               title: str = "Prediction Accuracy") -> plt.Figure:
        """予測精度の散布図を作成"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # X座標
        axes[0].scatter(targets[:, 0], predictions[:, 0], alpha=0.6, s=20)
        axes[0].plot([targets[:, 0].min(), targets[:, 0].max()],
                    [targets[:, 0].min(), targets[:, 0].max()], 'r--', linewidth=2)
        axes[0].set_xlabel('True X Displacement (mm)')
        axes[0].set_ylabel('Predicted X Displacement (mm)')
        axes[0].set_title('X-Direction Accuracy')
        axes[0].grid(True, alpha=0.3)

        # 相関係数表示
        r_x = np.corrcoef(targets[:, 0], predictions[:, 0])[0, 1]
        axes[0].text(0.05, 0.95, f'R = {r_x:.3f}', transform=axes[0].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Y座標
        axes[1].scatter(targets[:, 1], predictions[:, 1], alpha=0.6, s=20)
        axes[1].plot([targets[:, 1].min(), targets[:, 1].max()],
                    [targets[:, 1].min(), targets[:, 1].max()], 'r--', linewidth=2)
        axes[1].set_xlabel('True Y Displacement (mm)')
        axes[1].set_ylabel('Predicted Y Displacement (mm)')
        axes[1].set_title('Y-Direction Accuracy')
        axes[1].grid(True, alpha=0.3)

        r_y = np.corrcoef(targets[:, 1], predictions[:, 1])[0, 1]
        axes[1].text(0.05, 0.95, f'R = {r_y:.3f}', transform=axes[1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / "prediction_accuracy.png", dpi=300, bbox_inches='tight')

        return fig

    def plot_error_distribution(self, predictions: np.ndarray, targets: np.ndarray,
                              title: str = "Error Distribution") -> plt.Figure:
        """誤差分布をプロット"""
        errors = predictions - targets
        euclidean_errors = np.linalg.norm(errors, axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # X方向誤差ヒストグラム
        axes[0, 0].hist(errors[:, 0], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('X-Direction Error (mm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('X-Direction Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # 統計情報
        mean_x = np.mean(errors[:, 0])
        std_x = np.std(errors[:, 0])
        axes[0, 0].text(0.02, 0.98, f'Mean: {mean_x:.3f}\nStd: {std_x:.3f}',
                       transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Y方向誤差ヒストグラム
        axes[0, 1].hist(errors[:, 1], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Y-Direction Error (mm)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Y-Direction Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        mean_y = np.mean(errors[:, 1])
        std_y = np.std(errors[:, 1])
        axes[0, 1].text(0.02, 0.98, f'Mean: {mean_y:.3f}\nStd: {std_y:.3f}',
                       transform=axes[0, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # ユークリッド誤差ヒストグラム
        axes[1, 0].hist(euclidean_errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Euclidean Error (mm)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Euclidean Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        mean_euc = np.mean(euclidean_errors)
        std_euc = np.std(euclidean_errors)
        p95_euc = np.percentile(euclidean_errors, 95)
        axes[1, 0].axvline(p95_euc, color='orange', linestyle='--', linewidth=2, label=f'P95: {p95_euc:.2f}')
        axes[1, 0].text(0.02, 0.98, f'Mean: {mean_euc:.3f}\nStd: {std_euc:.3f}',
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        axes[1, 0].legend()

        # 誤差ベクトル散布図
        axes[1, 1].scatter(errors[:, 0], errors[:, 1], alpha=0.6, s=20)
        axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=1)
        axes[1, 1].set_xlabel('X Error (mm)')
        axes[1, 1].set_ylabel('Y Error (mm)')
        axes[1, 1].set_title('Error Vector Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / "error_distribution.png", dpi=300, bbox_inches='tight')

        return fig

    def plot_case_comparison(self, predictions: np.ndarray, targets: np.ndarray,
                           case_ids: List[str], max_cases: int = 10) -> plt.Figure:
        """ケース別性能比較"""
        if not case_ids:
            return None

        unique_cases = list(set(case_ids))[:max_cases]
        case_errors = []

        for case_id in unique_cases:
            case_mask = np.array([cid == case_id for cid in case_ids])
            case_pred = predictions[case_mask]
            case_target = targets[case_mask]
            errors = np.linalg.norm(case_pred - case_target, axis=1)
            case_errors.append(errors)

        fig, ax = plt.subplots(figsize=(12, 8))

        # ボックスプロット
        box_plot = ax.boxplot(case_errors, labels=unique_cases, patch_artist=True)

        # 色付け
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cases)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Case ID')
        ax.set_ylabel('Euclidean Error (mm)')
        ax.set_title('Error Distribution by Case')
        ax.grid(True, alpha=0.3)

        # X軸ラベルを回転
        plt.xticks(rotation=45)
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / "case_comparison.png", dpi=300, bbox_inches='tight')

        return fig

    def plot_learning_curves(self, metrics_history: Dict[str, List[Dict]],
                           title: str = "Learning Curves") -> plt.Figure:
        """学習曲線をプロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 損失曲線
        if 'train' in metrics_history and 'val' in metrics_history:
            train_epochs = [m['epoch'] for m in metrics_history['train']]
            train_loss = [m.get('total', m.get('reconstruction', 0)) for m in metrics_history['train']]
            val_epochs = [m['epoch'] for m in metrics_history['val']]
            val_loss = [m.get('total', m.get('reconstruction', 0)) for m in metrics_history['val']]

            axes[0, 0].plot(train_epochs, train_loss, label='Train', marker='o', markersize=3)
            axes[0, 0].plot(val_epochs, val_loss, label='Validation', marker='s', markersize=3)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training & Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # RMSE曲線
        if 'val' in metrics_history:
            val_rmse = [m.get('rmse_mm', 0) for m in metrics_history['val'] if 'rmse_mm' in m]
            if val_rmse:
                val_epochs_rmse = [m['epoch'] for m in metrics_history['val'] if 'rmse_mm' in m]
                axes[0, 1].plot(val_epochs_rmse, val_rmse, label='Validation RMSE',
                               marker='o', markersize=3, color='orange')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('RMSE (mm)')
                axes[0, 1].set_title('Validation RMSE')
                axes[0, 1].grid(True, alpha=0.3)

        # 損失成分（CVAEの場合）
        if 'train' in metrics_history:
            train_rec = [m.get('reconstruction', 0) for m in metrics_history['train']]
            train_kl = [m.get('kl', 0) for m in metrics_history['train']]

            axes[1, 0].plot(train_epochs, train_rec, label='Reconstruction', marker='o', markersize=3)
            if any(kl > 0 for kl in train_kl):
                axes[1, 0].plot(train_epochs, train_kl, label='KL Divergence', marker='s', markersize=3)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Component')
            axes[1, 0].set_title('Loss Components')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 学習率曲線
        train_lr = [m.get('learning_rate', 0) for m in metrics_history.get('train', [])]
        if any(lr > 0 for lr in train_lr):
            axes[1, 1].plot(train_epochs, train_lr, marker='o', markersize=3, color='green')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / "learning_curves.png", dpi=300, bbox_inches='tight')

        return fig

    def plot_model_comparison(self, comparison_results: Dict[str, Dict[str, float]],
                            title: str = "Model Comparison (vs Baseline-0)") -> plt.Figure:
        """モデル比較の可視化"""
        models = list(comparison_results.keys())
        metrics = ['rmse_mm', 'mae_mm', 'median_error_mm', 'p95_error_mm']

        # データ準備
        data = []
        for model in models:
            for metric in metrics:
                if metric in comparison_results[model]:
                    data.append({
                        'Model': model,
                        'Metric': metric.replace('_mm', '').upper(),
                        'Value': comparison_results[model][metric]
                    })

        df = pd.DataFrame(data)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 各メトリクスごとにバープロット
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            metric_data = df[df['Metric'] == metric.replace('_mm', '').upper()]

            if not metric_data.empty:
                sns.barplot(data=metric_data, x='Model', y='Value', ax=ax, palette='viridis')
                ax.set_title(f'{metric.replace("_mm", "").upper()} Comparison')
                ax.set_ylabel(f'{metric.replace("_mm", "")} (mm)')
                ax.tick_params(axis='x', rotation=45)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / "model_comparison.png", dpi=300, bbox_inches='tight')

        return fig

    def create_interactive_error_plot(self, predictions: np.ndarray, targets: np.ndarray,
                                    case_ids: Optional[List[str]] = None) -> go.Figure:
        """インタラクティブな誤差プロット（Plotly）"""
        errors = predictions - targets
        euclidean_errors = np.linalg.norm(errors, axis=1)

        # データフレーム作成
        df = pd.DataFrame({
            'x_error': errors[:, 0],
            'y_error': errors[:, 1],
            'euclidean_error': euclidean_errors,
            'pred_x': predictions[:, 0],
            'pred_y': predictions[:, 1],
            'target_x': targets[:, 0],
            'target_y': targets[:, 1],
            'case_id': case_ids if case_ids else ['Unknown'] * len(predictions)
        })

        # サブプロット作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('X vs Y Error', 'Error vs Target X', 'Error vs Target Y', 'Error Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )

        # 誤差ベクトルプロット
        fig.add_trace(
            go.Scatter(
                x=df['x_error'],
                y=df['y_error'],
                mode='markers',
                marker=dict(
                    color=df['euclidean_error'],
                    colorscale='Viridis',
                    colorbar=dict(title="Euclidean Error (mm)"),
                    size=5
                ),
                text=df['case_id'],
                hovertemplate='X Error: %{x:.3f}<br>Y Error: %{y:.3f}<br>Case: %{text}<extra></extra>',
                name='Error Vectors'
            ),
            row=1, col=1
        )

        # X座標 vs 誤差
        fig.add_trace(
            go.Scatter(
                x=df['target_x'],
                y=df['euclidean_error'],
                mode='markers',
                marker=dict(color='blue', size=4),
                text=df['case_id'],
                hovertemplate='Target X: %{x:.3f}<br>Error: %{y:.3f}<br>Case: %{text}<extra></extra>',
                name='Error vs Target X'
            ),
            row=1, col=2
        )

        # Y座標 vs 誤差
        fig.add_trace(
            go.Scatter(
                x=df['target_y'],
                y=df['euclidean_error'],
                mode='markers',
                marker=dict(color='red', size=4),
                text=df['case_id'],
                hovertemplate='Target Y: %{x:.3f}<br>Error: %{y:.3f}<br>Case: %{text}<extra></extra>',
                name='Error vs Target Y'
            ),
            row=2, col=1
        )

        # 誤差分布ヒストグラム
        fig.add_trace(
            go.Histogram(
                x=df['euclidean_error'],
                nbinsx=50,
                name='Error Distribution',
                marker=dict(color='green', opacity=0.7)
            ),
            row=2, col=2
        )

        # レイアウト更新
        fig.update_layout(
            title_text="Interactive Error Analysis",
            showlegend=False,
            height=800
        )

        # 軸ラベル更新
        fig.update_xaxes(title_text="X Error (mm)", row=1, col=1)
        fig.update_yaxes(title_text="Y Error (mm)", row=1, col=1)
        fig.update_xaxes(title_text="Target X (mm)", row=1, col=2)
        fig.update_yaxes(title_text="Euclidean Error (mm)", row=1, col=2)
        fig.update_xaxes(title_text="Target Y (mm)", row=2, col=1)
        fig.update_yaxes(title_text="Euclidean Error (mm)", row=2, col=1)
        fig.update_xaxes(title_text="Euclidean Error (mm)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        if self.save_dir:
            fig.write_html(str(self.save_dir / "interactive_error_plot.html"))

        return fig

    def create_dashboard(self, evaluation_results: Dict[str, Any],
                        predictions_data: Dict[str, np.ndarray]) -> go.Figure:
        """評価ダッシュボード作成"""
        # メトリクスサマリー表示用のデータ準備
        metrics = evaluation_results.get('metrics', {})

        # ダッシュボード作成
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Key Metrics', 'Error Distribution',
                'Prediction Accuracy (X)', 'Prediction Accuracy (Y)',
                'Case Performance', 'Improvement Gain'
            ),
            specs=[
                [{"type": "table"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )

        # メトリクステーブル
        if metrics:
            metric_names = ['RMSE (mm)', 'MAE (mm)', 'Median Error (mm)', 'P95 Error (mm)']
            metric_values = [
                f"{metrics.get('rmse_mm', 0):.2f}",
                f"{metrics.get('mae_mm', 0):.2f}",
                f"{metrics.get('median_error_mm', 0):.2f}",
                f"{metrics.get('p95_error_mm', 0):.2f}"
            ]

            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
                    cells=dict(values=[metric_names, metric_values], fill_color='white')
                ),
                row=1, col=1
            )

        # その他のプロットも追加...
        # (実装簡略化のため省略)

        fig.update_layout(
            title_text="RIMD Model Evaluation Dashboard (Gain vs Baseline-0)",
            height=1200
        )

        if self.save_dir:
            fig.write_html(str(self.save_dir / "evaluation_dashboard.html"))

        return fig


def create_visualization_report(eval_results: Dict[str, Any],
                               save_dir: Path) -> None:
    """包括的な可視化レポートを作成"""
    visualizer = RIMDVisualizer(save_dir)

    # 基本的な可視化
    for split, results in eval_results.items():
        if 'predictions' in results and 'targets' in results:
            predictions = results['predictions']
            targets = results['targets']
            case_ids = results.get('case_ids', [])

            # 予測精度プロット
            visualizer.plot_prediction_accuracy(
                predictions, targets, case_ids, f"{split.capitalize()} Prediction Accuracy"
            )

            # 誤差分布プロット
            visualizer.plot_error_distribution(
                predictions, targets, f"{split.capitalize()} Error Distribution"
            )

            # ケース比較（ケースIDがある場合）
            if case_ids:
                visualizer.plot_case_comparison(
                    predictions, targets, case_ids
                )

            # インタラクティブプロット
            interactive_fig = visualizer.create_interactive_error_plot(
                predictions, targets, case_ids
            )

    logger.info(f"Visualization report created in {save_dir}")