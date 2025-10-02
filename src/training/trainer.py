import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Any
import logging

from .losses import create_loss_function, LossTracker
from .metrics import RIMDMetrics
from ..models.base_model import create_model

logger = logging.getLogger(__name__)


class RIMDTrainer:
    """RIMD予測モデルの学習を管理するトレーナー"""

    def __init__(self, config, experiment_manager):
        self.config = config
        self.exp_manager = experiment_manager
        self.device = experiment_manager.device

        # モデル、損失、メトリクス（後で初期化）
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.metrics = None

        # 学習状態
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.loss_tracker = LossTracker()

        logger.info(f"Initialized trainer with device: {self.device}")

    def setup(self, datamodule):
        """データモジュールを使ってトレーナーをセットアップ"""
        # 特徴量次元を取得
        edge_dim, node_dim = datamodule.get_feature_dimensions()

        # モデル作成
        self.model = create_model(self.config, edge_dim, node_dim)
        self.model.to(self.device)

        # オプティマイザ作成
        self._create_optimizer()

        # スケジューラ作成
        self._create_scheduler()

        # 損失関数作成
        self.loss_fn = create_loss_function(self.config)

        # メトリクス作成
        self.metrics = RIMDMetrics(self.config)

        logger.info("Trainer setup completed")
        logger.info(f"Model: {self.model.get_model_summary()}")

    def _create_optimizer(self):
        """オプティマイザを作成"""
        optimizer_config = self.config.training

        if optimizer_config.optimizer.lower() == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps
            )
        elif optimizer_config.optimizer.lower() == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.optimizer}")

        logger.info(f"Created {optimizer_config.optimizer} optimizer")

    def _create_scheduler(self):
        """学習率スケジューラを作成"""
        scheduler_type = self.config.training.scheduler.lower()

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_epochs
            )
        elif scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.training.scheduler_patience,
                verbose=True
            )
        elif scheduler_type == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        logger.info(f"Created {scheduler_type} scheduler")

    def fit(self, datamodule):
        """モデルを学習"""
        self.setup(datamodule)

        # データローダー取得
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        logger.info(f"Starting training for {self.config.training.max_epochs} epochs")

        try:
            for epoch in range(self.config.training.max_epochs):
                self.current_epoch = epoch

                # 学習フェーズ
                train_metrics = self.train_epoch(train_loader)

                # 検証フェーズ
                val_metrics = self.validate_epoch(val_loader)

                # メトリクス記録
                self.exp_manager.log_metrics(epoch, train_metrics, val_metrics)

                # スケジューラ更新
                self._update_scheduler(val_metrics.get('total_loss', val_metrics.get('reconstruction_loss')))

                # 早期終了チェック
                if self._should_stop(val_metrics):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                # モデル保存
                self._save_checkpoint(epoch, val_metrics)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        logger.info("Training completed")
        return self.get_training_summary()

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """1エポックの学習"""
        self.model.train()
        self.loss_tracker.reset()

        epoch_pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch:03d} [Train]")

        for batch_idx, batch in enumerate(epoch_pbar):
            # バッチをデバイスに移動
            batch = self._move_batch_to_device(batch)

            # 勾配クリア
            self.optimizer.zero_grad()

            # 前向き計算
            predictions = self.model(batch)

            # 損失計算
            losses = self.loss_fn(predictions, batch, self.current_epoch)

            # 逆伝播
            losses['total'].backward()

            # 勾配クリッピング
            if self.config.model.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.model.gradient_clip_norm
                )

            # パラメータ更新
            self.optimizer.step()

            # 損失追跡
            self.loss_tracker.update(losses)

            # プログレスバー更新
            current_losses = self.loss_tracker.get_averages(window=10)
            epoch_pbar.set_postfix({
                'loss': f"{current_losses.get('total', 0.0):.4f}",
                'rec': f"{current_losses.get('reconstruction', 0.0):.4f}"
            })

        return self.loss_tracker.get_averages()

    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """1エポックの検証"""
        self.model.eval()
        val_loss_tracker = LossTracker()
        val_metrics_tracker = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {self.current_epoch:03d} [Val]"):
                batch = self._move_batch_to_device(batch)

                # 前向き計算
                predictions = self.model(batch)

                # 損失計算
                losses = self.loss_fn(predictions, batch, self.current_epoch)
                val_loss_tracker.update(losses)

                # メトリクス計算
                if hasattr(batch, 'coords_1step') and hasattr(batch, 'coords_nv'):
                    metrics = self.metrics.compute_metrics(
                        predictions['output'], batch, compute_all=True
                    )
                    val_metrics_tracker.append(metrics)

        # 平均メトリクス計算
        val_losses = val_loss_tracker.get_averages()

        if val_metrics_tracker:
            avg_metrics = self._average_metrics(val_metrics_tracker)
            val_losses.update(avg_metrics)

        return val_losses

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """バッチをデバイスに移動"""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _update_scheduler(self, val_loss: Optional[float]):
        """スケジューラを更新"""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _should_stop(self, val_metrics: Dict[str, float]) -> bool:
        """早期終了判定"""
        val_loss = val_metrics.get('total_loss', val_metrics.get('reconstruction_loss', float('inf')))

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.training.early_stopping_patience

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        """チェックポイント保存"""
        val_loss = val_metrics.get('total_loss', val_metrics.get('reconstruction_loss', float('inf')))
        is_best = val_loss <= self.best_val_loss

        self.exp_manager.save_model(
            self.model,
            epoch,
            {**val_metrics, 'train_metrics': self.loss_tracker.get_averages()},
            is_best=is_best
        )

    def _average_metrics(self, metrics_list: list) -> Dict[str, float]:
        """メトリクスのリストを平均化"""
        if not metrics_list:
            return {}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)

        return avg_metrics

    def predict(self, dataloader) -> Dict[str, np.ndarray]:
        """予測を実行"""
        self.model.eval()
        predictions = []
        targets = []
        case_ids = []
        coords_1step = []
        coords_nv = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = self._move_batch_to_device(batch)

                pred = self.model(batch)
                predictions.append(pred['output'].cpu().numpy())

                if 'target' in batch:
                    targets.append(batch['target'].cpu().numpy())

                if 'case_ids' in batch:
                    case_ids.extend(batch['case_ids'])

                # 評価用座標データも保存
                if 'coords_1step' in batch:
                    coords_1step.append(batch['coords_1step'].cpu().numpy())
                if 'coords_nv' in batch:
                    coords_nv.append(batch['coords_nv'].cpu().numpy())

        result = {
            'predictions': np.vstack(predictions),
            'case_ids': case_ids
        }

        if targets:
            result['targets'] = np.vstack(targets)

        if coords_1step:
            result['coords_1step'] = np.vstack(coords_1step)

        if coords_nv:
            result['coords_nv'] = np.vstack(coords_nv)

        return result

    def get_training_summary(self) -> Dict[str, Any]:
        """学習の要約を取得"""
        return {
            'epochs_trained': self.current_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'final_learning_rate': self.optimizer.param_groups[0]['lr'],
            'model_summary': self.model.get_model_summary(),
            'config': self.config.to_dict()
        }

    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイントを読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_model_for_inference(self, save_path: str):
        """推論用にモデルを保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config.model.to_dict(),
            'feature_dims': (self.model.edge_dim, self.model.node_dim)
        }, save_path)

        logger.info(f"Model saved for inference at {save_path}")