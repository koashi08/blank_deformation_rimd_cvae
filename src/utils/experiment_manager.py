import json
import joblib
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from ..utils.logger import setup_logger, MetricsLogger, log_config
from ..utils.reproducibility import set_seed, get_device, log_system_info


class ExperimentManager:
    """実験の設定、実行、結果管理"""

    def __init__(self, config, create_dirs: bool = True):
        self.config = config
        self.exp_dir = Path("experiments") / config.exp_id

        if create_dirs:
            self.setup_directories()

        self.logger = None
        self.metrics_logger = None
        self.device = None

    def setup_directories(self):
        """実験用ディレクトリを作成"""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)
        (self.exp_dir / "visualizations").mkdir(exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)

    def setup_experiment(self):
        """実験環境のセットアップ"""
        # タイムスタンプ付きの実験IDに更新
        self.config.update_exp_id_with_timestamp()
        self.exp_dir = Path("experiments") / self.config.exp_id
        self.setup_directories()

        # 設定保存
        self.config.save(self.exp_dir / "config.json")

        # ロギング設定
        self.setup_logging()

        # 再現性設定
        self.setup_reproducibility()

        # システム情報ログ
        log_system_info()
        log_config(self.config, self.logger)

    def setup_logging(self):
        """ロギングシステムの設定"""
        self.logger = setup_logger(
            name=f"exp_{self.config.exp_id}",
            log_dir=self.exp_dir / "logs",
            level=logging.INFO
        )
        self.metrics_logger = MetricsLogger(self.exp_dir / "logs")

    def setup_reproducibility(self):
        """再現性の設定"""
        set_seed(
            seed=self.config.training.seed,
            deterministic=self.config.training.deterministic
        )
        self.device = get_device()
        if self.logger:
            self.logger.info(f"Using device: {self.device}")

    def save_scalers(self, scalers: Dict[str, Any]):
        """スケーラを保存"""
        scaler_path = self.exp_dir / "scalers.pkl"
        joblib.dump(scalers, scaler_path)
        if self.logger:
            self.logger.info(f"Scalers saved to {scaler_path}")

    def load_scalers(self) -> Dict[str, Any]:
        """スケーラを読み込み"""
        scaler_path = self.exp_dir / "scalers.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scalers not found at {scaler_path}")
        return joblib.load(scaler_path)

    def save_model(self, model, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """モデルを保存"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

        # エポック別保存
        checkpoint_path = self.exp_dir / "checkpoints" / f"model_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)

        # ベストモデル保存
        if is_best:
            best_path = self.exp_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            if self.logger:
                self.logger.info(f"Best model saved at epoch {epoch}")

        # 最新モデル保存
        latest_path = self.exp_dir / "latest_model.pth"
        torch.save(checkpoint, latest_path)

    def load_model_checkpoint(self, checkpoint_name: str = "best_model.pth") -> Dict[str, Any]:
        """モデルチェックポイントを読み込み"""
        checkpoint_path = self.exp_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        return checkpoint

    def save_results(self, results: Dict[str, Any]):
        """最終結果を保存"""
        results_with_meta = {
            **results,
            'exp_id': self.config.exp_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict()
        }

        results_path = self.exp_dir / "final_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, indent=2, ensure_ascii=False)

        if self.logger:
            self.logger.info(f"Final results saved to {results_path}")

    def log_metrics(self, epoch: int, train_metrics: Dict[str, float],
                   val_metrics: Optional[Dict[str, float]] = None):
        """メトリクスをログ記録"""
        if self.metrics_logger:
            self.metrics_logger.log_metrics(epoch, train_metrics, prefix="train_")
            if val_metrics:
                self.metrics_logger.log_metrics(epoch, val_metrics, prefix="val_")

        if self.logger:
            train_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            self.logger.info(f"Epoch {epoch:03d} | Train | {train_str}")

            if val_metrics:
                val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                self.logger.info(f"Epoch {epoch:03d} | Val   | {val_str}")

    @classmethod
    def load_experiment(cls, exp_id: str) -> 'ExperimentManager':
        """保存済み実験を読み込み"""
        from config.base_config import ExperimentConfig

        exp_dir = Path("experiments") / exp_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        config_path = exp_dir / "config.json"
        config = ExperimentConfig.load(config_path)

        manager = cls(config, create_dirs=False)
        manager.exp_dir = exp_dir
        return manager

    def load_model_and_scalers(self, checkpoint_name: str = "best_model.pth") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """モデルとスケーラを同時に読み込み"""
        checkpoint = self.load_model_checkpoint(checkpoint_name)
        scalers = self.load_scalers()
        return checkpoint, scalers

    def get_experiment_summary(self) -> Dict[str, Any]:
        """実験の要約を取得"""
        summary = {
            'exp_id': self.config.exp_id,
            'description': self.config.description,
            'exp_dir': str(self.exp_dir),
            'config': self.config.to_dict()
        }

        # 結果ファイルが存在する場合は追加
        results_path = self.exp_dir / "final_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                summary['final_results'] = results

        return summary


# ログ設定を忘れずにimport
import logging