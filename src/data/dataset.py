import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class RIMDDataset(Dataset):
    """RIMD特徴量とターゲットのデータセット"""

    def __init__(
        self,
        case_files: List[str],
        processed_dir: Path,
        scalers: Dict[str, Any],
        config,
        split: str = "train"
    ):
        """
        Args:
            case_files: ケースファイル名のリスト
            processed_dir: 前処理済みデータディレクトリ
            scalers: 標準化器の辞書
            config: データ設定
            split: データ分割名 ("train", "val", "test")
        """
        self.case_files = case_files
        self.processed_dir = Path(processed_dir)
        self.scalers = scalers
        self.config = config
        self.split = split

        logger.info(f"Created {split} dataset with {len(case_files)} cases")

    def __len__(self) -> int:
        return len(self.case_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """1つのケースのデータを取得"""
        case_file = self.case_files[idx]
        case_path = self.processed_dir / f"{case_file}.pkl"

        # データ読み込み
        with open(case_path, 'rb') as f:
            data = pickle.load(f)

        # 特徴量構築
        edge_features, node_features = self._build_features(data)

        # ターゲット構築
        target = self._build_target(data)

        # PyTorchテンソルに変換
        result = {
            'edge_features': torch.tensor(edge_features, dtype=torch.float32),
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(data['edge_index'], dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float32),
            'case_id': case_file
        }

        # 評価用の追加データ
        if self.split in ["val", "test"]:
            result.update({
                'coords_1step': torch.tensor(data['xy_1step'], dtype=torch.float32),
                'coords_nv': torch.tensor(data['xy_nv'], dtype=torch.float32),
                'representative_scale': torch.tensor(self.scalers['representative_scale'], dtype=torch.float32)
            })

        return result

    def _build_features(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """特徴量を構築して標準化"""

        # エッジ特徴量 [φ, geometry]
        edge_features = np.concatenate([
            data['rimd_edges_1step'],  # [E, 3]
            data['edge_geometry']      # [E, 1]
        ], axis=1)

        # ノード特徴量 [σ, geometry, φ_mean, (coords)]
        node_feature_list = [
            data['rimd_nodes_1step'],  # [V, 6]
            data['geometry_features']   # [V, 2]
        ]

        # φ平均特徴量
        if self.config.model.use_phi_mean_features:
            node_feature_list.append(data['phi_mean'])  # [V, 3]

        # 座標特徴量
        if self.config.model.use_position_features:
            node_feature_list.append(data['xy_1step'])  # [V, 2]

        node_features = np.concatenate(node_feature_list, axis=1)

        # 標準化
        edge_features = self.scalers['edge_scaler'].transform(edge_features)
        node_features = self.scalers['node_scaler'].transform(node_features)

        return edge_features, node_features

    def _build_target(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """ターゲットを構築して標準化"""
        # 代表寸法で正規化
        delta_xy = data['delta_xy'] / self.scalers['representative_scale']

        # さらにZ標準化
        target = self.scalers['target_scaler'].transform(delta_xy)

        return target


class RIMDDataModule:
    """データローダを管理するクラス"""

    def __init__(self, config, scalers: Dict[str, Any]):
        self.config = config
        self.scalers = scalers
        self.processed_dir = Path("../data/processed")
        self.splits_dir = Path("../data/splits")

        # データ分割読み込み
        self._load_splits()

        # データセット作成
        self._create_datasets()

    def _load_splits(self):
        """データ分割を読み込み"""
        splits_path = self.splits_dir / "data_splits.pkl"
        with open(splits_path, 'rb') as f:
            self.splits = pickle.load(f)

        logger.info(f"Loaded splits: Train={len(self.splits['train'])}, "
                   f"Val={len(self.splits['val'])}, Test={len(self.splits['test'])}")

    def _create_datasets(self):
        """データセットを作成"""
        self.train_dataset = RIMDDataset(
            self.splits['train'], self.processed_dir, self.scalers, self.config, split="train"
        )
        self.val_dataset = RIMDDataset(
            self.splits['val'], self.processed_dir, self.scalers, self.config, split="val"
        )
        self.test_dataset = RIMDDataset(
            self.splits['test'], self.processed_dir, self.scalers, self.config, split="test"
        )

    def train_dataloader(self) -> DataLoader:
        """学習用データローダ"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self) -> DataLoader:
        """検証用データローダ"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def test_dataloader(self) -> DataLoader:
        """テスト用データローダ"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_sample_batch(self, split: str = "train") -> Dict[str, torch.Tensor]:
        """サンプルバッチを取得（デバッグ用）"""
        if split == "train":
            loader = self.train_dataloader()
        elif split == "val":
            loader = self.val_dataloader()
        else:
            loader = self.test_dataloader()

        return next(iter(loader))

    def get_feature_dimensions(self) -> Tuple[int, int]:
        """特徴量次元を取得"""
        sample_batch = self.get_sample_batch("train")
        edge_dim = sample_batch['edge_features'].shape[-1]
        node_dim = sample_batch['node_features'].shape[-1]

        logger.info(f"Feature dimensions: Edge={edge_dim}, Node={node_dim}")
        return edge_dim, node_dim


def collate_rimd_batch(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    RIMDデータのバッチ処理用collate関数

    Args:
        batch_list: バッチデータのリスト

    Returns:
        バッチ化されたデータ
    """
    # バッチサイズ
    batch_size = len(batch_list)

    # 各グラフの情報を取得
    edge_features_list = []
    node_features_list = []
    edge_index_list = []
    target_list = []
    case_ids = []

    # バッチインデックスとオフセット管理
    batch_indices = []
    node_offset = 0

    for i, data in enumerate(batch_list):
        edge_features_list.append(data['edge_features'])
        node_features_list.append(data['node_features'])

        # エッジインデックスにオフセットを適用
        edge_index = data['edge_index'] + node_offset
        edge_index_list.append(edge_index)

        target_list.append(data['target'])
        case_ids.append(data['case_id'])

        # バッチインデックス作成
        num_nodes = data['node_features'].shape[0]
        batch_indices.extend([i] * num_nodes)
        node_offset += num_nodes

    # テンソル結合
    result = {
        'edge_features': torch.cat(edge_features_list, dim=0),
        'node_features': torch.cat(node_features_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'target': torch.cat(target_list, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'case_ids': case_ids
    }

    # 評価用データも含める
    if 'coords_1step' in batch_list[0]:
        coords_1step_list = [data['coords_1step'] for data in batch_list]
        coords_nv_list = [data['coords_nv'] for data in batch_list]
        rep_scale_list = [data['representative_scale'] for data in batch_list]

        result.update({
            'coords_1step': torch.cat(coords_1step_list, dim=0),
            'coords_nv': torch.cat(coords_nv_list, dim=0),
            'representative_scale': torch.stack(rep_scale_list)
        })

    return result


class RIMDDataModuleWithCustomCollate(RIMDDataModule):
    """カスタムcollate関数を使用するデータモジュール"""

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_rimd_batch
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_rimd_batch
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_rimd_batch
        )