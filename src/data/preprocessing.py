import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class RIMDPreprocessor:
    """RIMD特徴量とターゲットの前処理"""

    def __init__(self, config):
        self.config = config
        self.data_root = Path(config.data_root)
        self.processed_dir = Path("../data/processed")
        self.splits_dir = Path("../data/splits")

        # ディレクトリ作成
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        self.scalers = {}

        # データファイルパス
        self.df_node_path = self.data_root / "df_node.pkl"
        self.df_element_path = self.data_root / "df_element.pkl"

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生データ（df_node.pkl、df_element.pkl）を読み込み

        Returns:
            df_node: ノードデータ
            df_element: 要素データ
        """
        logger.info("生データを読み込み中...")

        # ノードデータの読み込み
        if not self.df_node_path.exists():
            raise FileNotFoundError(f"Node data file not found: {self.df_node_path}")

        with open(self.df_node_path, 'rb') as f:
            df_node = pickle.load(f)

        # 要素データの読み込み
        if not self.df_element_path.exists():
            raise FileNotFoundError(f"Element data file not found: {self.df_element_path}")

        with open(self.df_element_path, 'rb') as f:
            df_element = pickle.load(f)

        logger.info(f"ノードデータ形状: {df_node.shape}")
        logger.info(f"要素データ形状: {df_element.shape}")
        logger.info(f"ケース数: {df_node['case'].nunique()}")

        return df_node, df_element

    def extract_case_data(self, df_node: pd.DataFrame, df_element: pd.DataFrame,
                         case_id: str) -> Dict[str, np.ndarray]:
        """
        特定ケースのデータを抽出

        Args:
            df_node: ノードデータ
            df_element: 要素データ
            case_id: ケースID（例：'No001'）

        Returns:
            ケースデータ辞書
        """
        # ノードデータの抽出
        node_mask = df_node['case'] == case_id
        case_nodes = df_node[node_mask].copy()

        # 要素データの抽出
        element_mask = df_element['case'] == case_id
        case_elements = df_element[element_mask].copy()

        if len(case_nodes) == 0:
            raise ValueError(f"No node data found for case {case_id}")
        if len(case_elements) == 0:
            raise ValueError(f"No element data found for case {case_id}")

        # NodeIDでソート（1~1071の順序を保証）
        case_nodes = case_nodes.sort_values('NodeID').reset_index(drop=True)

        # 座標データの抽出
        coords_1step_blk = case_nodes[['step_blk_coord_x', 'step_blk_coord_y']].values
        coords_1step_prod = case_nodes[['step_prod_coord_x', 'step_prod_coord_y']].values
        coords_nv_blk = case_nodes[['nv_blk_coord_x', 'nv_blk_coord_y']].values
        coords_nv_prod = case_nodes[['nv_prod_coord_x', 'nv_prod_coord_y']].values

        # 要素データの抽出（四角形要素）
        quad_elements = case_elements[['n1', 'n2', 'n3', 'n4']].values
        # NodeIDを0ベースのインデックスに変換
        quad_elements = quad_elements - 1  # 1~1071 -> 0~1070

        return {
            'coords_1step_blk': coords_1step_blk,
            'coords_1step_prod': coords_1step_prod,
            'coords_nv_blk': coords_nv_blk,
            'coords_nv_prod': coords_nv_prod,
            'quad_elements': quad_elements,
            'node_ids': case_nodes['NodeID'].values,
            'element_ids': case_elements['ElementID'].values
        }

    def triangulate_quad_mesh(self, quad_elements: np.ndarray) -> np.ndarray:
        """
        四角形メッシュを三角形に分割（一貫した規約）

        Args:
            quad_elements: [N, 4] 四角形要素の頂点インデックス

        Returns:
            [2*N, 3] 三角形要素の頂点インデックス
        """
        triangles = []
        for quad in quad_elements:
            # 対角線分割規約: (0,1,2) と (2,3,0)
            triangles.append([quad[0], quad[1], quad[2]])
            triangles.append([quad[2], quad[3], quad[0]])

        return np.array(triangles)

    def compute_deformation_gradient(self, coords_from: np.ndarray, coords_to: np.ndarray,
                                   triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        変形勾配テンソルを計算し、極分解でR, Sを取得

        Args:
            coords_from: [V, 2] 変形前座標
            coords_to: [V, 2] 変形後座標
            triangles: [T, 3] 三角形要素

        Returns:
            R: [V, 2, 2] 回転行列
            S: [V, 2, 2] ストレッチ行列
        """
        V = coords_from.shape[0]
        R = np.zeros((V, 2, 2))
        S = np.zeros((V, 2, 2))

        # cotan重み計算（簡易版）
        for i in range(V):
            # 頂点iを含む三角形を探す
            incident_triangles = triangles[np.any(triangles == i, axis=1)]

            if len(incident_triangles) == 0:
                # デフォルト値
                R[i] = np.eye(2)
                S[i] = np.eye(2)
                continue

            # 簡易変形勾配計算（最小二乗）
            A_list = []

            for tri in incident_triangles:
                v_indices = tri
                if i not in v_indices:
                    continue

                # ローカル座標系での計算
                p_from = coords_from[v_indices]
                p_to = coords_to[v_indices]

                # 重心を原点とする
                centroid_from = p_from.mean(axis=0)
                centroid_to = p_to.mean(axis=0)

                p_from_centered = p_from - centroid_from
                p_to_centered = p_to - centroid_to

                # 簡易変形勾配（重心からの相対座標の比）
                if np.linalg.norm(p_from_centered) > 1e-8:
                    F_local = np.outer(p_to_centered.flatten(), p_from_centered.flatten()) / np.linalg.norm(p_from_centered)**2
                    A_list.append(F_local)

            if len(A_list) == 0:
                R[i] = np.eye(2)
                S[i] = np.eye(2)
            else:
                # 平均化
                F = np.mean(A_list, axis=0).reshape(2, 2)

                # 極分解: F = RS
                U, sigma, Vt = np.linalg.svd(F)
                R[i] = U @ Vt
                S[i] = Vt.T @ np.diag(sigma) @ Vt

        return R, S

    def compute_rimd_features(self, coords_1step: np.ndarray, coords_nv: np.ndarray,
                            triangles: np.ndarray, edge_list: np.ndarray) -> Dict[str, np.ndarray]:
        """
        RIMD特徴量を計算

        Args:
            coords_1step: [V, 2] 1step座標
            coords_nv: [V, 2] 逐次解析座標
            triangles: [T, 3] 三角形要素
            edge_list: [E, 2] エッジリスト

        Returns:
            RIMD特徴量辞書
        """
        # 変形勾配の極分解
        R, S = self.compute_deformation_gradient(coords_1step, coords_nv, triangles)

        # エッジのRIMD特徴 (φ_ij = log(R_i^T @ R_j))
        edge_features = []
        for i, j in edge_list:
            R_diff = R[i].T @ R[j]
            # 回転行列のlog（簡易版：Rodrigues公式）
            theta = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            if np.abs(theta) < 1e-6:
                phi = np.zeros(3)
            else:
                axis = np.array([R_diff[2,1] - R_diff[1,2],
                               R_diff[0,2] - R_diff[2,0],
                               R_diff[1,0] - R_diff[0,1]]) / (2 * np.sin(theta))
                # 2Dの場合、z軸周りの回転のみ
                phi = np.array([0, 0, theta * axis[2]]) if len(axis) > 2 else np.array([0, 0, theta])

            edge_features.append(phi)

        # ノードのRIMD特徴 (σ_i = log(S_i))
        node_features = []
        for i in range(len(S)):
            # 対称行列のlog
            eigenvals, eigenvecs = np.linalg.eigh(S[i])
            eigenvals = np.maximum(eigenvals, 1e-8)  # 数値安定性
            log_S = eigenvecs @ np.diag(np.log(eigenvals)) @ eigenvecs.T

            # 2x2対称行列を6次元ベクトルに（上三角）
            sigma = np.array([log_S[0,0], log_S[1,1], log_S[0,1], 0, 0, 0])
            node_features.append(sigma)

        # 座標誤差（ターゲット）
        delta_xy = coords_nv - coords_1step

        return {
            'edge_features': np.array(edge_features),  # [E, 3]
            'node_features': np.array(node_features),  # [V, 6]
            'delta_xy': delta_xy,  # [V, 2]
            'coords_1step': coords_1step,  # [V, 2]
            'coords_nv': coords_nv,  # [V, 2]
            'edge_list': edge_list,  # [E, 2]
        }

    def create_edge_list_from_triangles(self, triangles: np.ndarray) -> np.ndarray:
        """三角形から重複のないエッジリストを作成"""
        edges = set()
        for tri in triangles:
            for i in range(3):
                v1, v2 = tri[i], tri[(i+1) % 3]
                edges.add((min(v1, v2), max(v1, v2)))
        return np.array(list(edges))

    def compute_geometric_features(self, coords: np.ndarray, triangles: np.ndarray,
                                 edge_list: np.ndarray) -> Dict[str, np.ndarray]:
        """
        補助幾何特徴量を計算

        Args:
            coords: [V, 2] 座標
            triangles: [T, 3] 三角形
            edge_list: [E, 2] エッジリスト

        Returns:
            幾何特徴量辞書
        """
        V = coords.shape[0]

        # ノード特徴量
        node_areas = np.zeros(V)  # 頂点周りの面積
        node_angles = np.zeros(V)  # 頂点での平均角度

        for tri in triangles:
            v1, v2, v3 = coords[tri]
            area = 0.5 * np.abs(np.cross(v2 - v1, v3 - v1))

            for i, vi in enumerate(tri):
                node_areas[vi] += area / 3  # 面積を3等分

                # 角度計算
                vj, vk = tri[(i+1) % 3], tri[(i+2) % 3]
                vec1 = coords[vj] - coords[vi]
                vec2 = coords[vk] - coords[vi]

                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                node_angles[vi] += angle

        # エッジ特徴量
        edge_lengths = []
        for i, j in edge_list:
            length = np.linalg.norm(coords[i] - coords[j])
            edge_lengths.append(length)

        return {
            'node_features': np.column_stack([node_areas, node_angles]),  # [V, 2]
            'edge_features': np.array(edge_lengths).reshape(-1, 1)  # [E, 1]
        }


    def process_all_cases(self, case_ids: Optional[List[str]] = None):
        """全ケースを処理（実データ対応）"""
        # 生データを読み込み
        df_node, df_element = self.load_raw_data()

        if case_ids is None:
            # 実データから利用可能なケースIDを取得
            case_ids = sorted(df_node['case'].unique())

        logger.info(f"Processing {len(case_ids)} cases from real data")

        for case_id in tqdm(case_ids, desc="Processing cases"):
            try:
                processed_data = self.process_single_case_real(df_node, df_element, case_id)

                # 保存
                case_path = self.processed_dir / f"{case_id}.pkl"
                with open(case_path, 'wb') as f:
                    pickle.dump(processed_data, f)

            except Exception as e:
                logger.error(f"Error processing case {case_id}: {e}")
                continue

        logger.info("All cases processed successfully")

    def process_single_case_real(self, df_node: pd.DataFrame, df_element: pd.DataFrame,
                                case_id: str) -> Dict[str, np.ndarray]:
        """
        実データから単一ケースを処理

        Args:
            df_node: ノードデータ
            df_element: 要素データ
            case_id: ケースID

        Returns:
            処理済みデータ辞書
        """
        # ケースデータを抽出
        case_data = self.extract_case_data(df_node, df_element, case_id)

        # 座標の取得（ブランク→製品の変形を解析）
        coords_1step = case_data['coords_1step_prod']  # 1step解析の製品座標
        coords_nv = case_data['coords_nv_prod']        # 逐次解析の製品座標
        coords_1step_blk = case_data['coords_1step_blk']  # 1step解析のブランク座標（参考用）

        # 四角形要素を三角形に分割
        triangles = self.triangulate_quad_mesh(case_data['quad_elements'])

        # エッジリスト作成
        edge_list = self.create_edge_list_from_triangles(triangles)

        # RIMD特徴量計算（ブランク→製品の変形から）
        rimd_features = self.compute_rimd_features(
            coords_1step_blk, coords_1step, triangles, edge_list
        )

        # 補助幾何特徴量計算
        geom_features_1step = self.compute_geometric_features(coords_1step, triangles, edge_list)

        # phi_mean計算（各頂点の一環辺φの平均）
        phi_mean = self.compute_phi_mean(rimd_features['edge_features'], edge_list, len(coords_1step))

        # ターゲット（座標誤差）計算
        delta_xy = coords_nv - coords_1step  # [V, 2]

        # エッジインデックス（PyTorch Geometric形式）
        edge_index = edge_list.T  # [2, E]

        return {
            'case_id': case_id,
            'rimd_edges_1step': rimd_features['edge_features'],  # [E, 3]
            'rimd_nodes_1step': rimd_features['node_features'],  # [V, 6]
            'edge_index': edge_index,  # [2, E]
            'delta_xy': delta_xy,  # [V, 2]
            'xy_1step': coords_1step,  # [V, 2]
            'xy_nv': coords_nv,  # [V, 2]
            'geometry_features': geom_features_1step['node_features'],  # [V, Gn]
            'edge_geometry': geom_features_1step['edge_features'],  # [E, Ge]
            'phi_mean': phi_mean,  # [V, 3]
            'node_ids': case_data['node_ids'],  # [V]
        }

    def compute_phi_mean(self, edge_features: np.ndarray, edge_list: np.ndarray,
                        num_nodes: int) -> np.ndarray:
        """
        各頂点の一環辺φの平均を計算

        Args:
            edge_features: [E, 3] エッジのRIMD特徴量（φ）
            edge_list: [E, 2] エッジリスト
            num_nodes: ノード数

        Returns:
            [V, 3] 各頂点のφ平均
        """
        phi_sum = np.zeros((num_nodes, 3))
        phi_count = np.zeros(num_nodes)

        for edge_idx, (i, j) in enumerate(edge_list):
            phi = edge_features[edge_idx]
            phi_sum[i] += phi
            phi_sum[j] += phi
            phi_count[i] += 1
            phi_count[j] += 1

        # 平均計算（0除算回避）
        phi_mean = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            if phi_count[i] > 0:
                phi_mean[i] = phi_sum[i] / phi_count[i]

        return phi_mean

    def create_splits(self):
        """データをtrain/val/testに分割"""
        processed_files = list(self.processed_dir.glob("*.pkl"))
        total_files = len(processed_files)

        if total_files == 0:
            raise ValueError("No processed files found. Run process_all_cases first.")

        # シャッフル
        np.random.seed(self.config.training.seed)
        indices = np.random.permutation(total_files)

        # 分割
        n_train = int(total_files * self.config.train_ratio)
        n_val = int(total_files * self.config.val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        # ファイル名取得
        train_files = [processed_files[i].stem for i in train_indices]
        val_files = [processed_files[i].stem for i in val_indices]
        test_files = [processed_files[i].stem for i in test_indices]

        # 分割情報保存
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        splits_path = self.splits_dir / "data_splits.pkl"
        with open(splits_path, 'wb') as f:
            pickle.dump(splits, f)

        logger.info(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        logger.info(f"Splits saved to {splits_path}")

    def compute_representative_scale(self, case_files: List[str]) -> float:
        """代表寸法Sを計算（BBの対角線長の中央値）"""
        scales = []

        for case_file in case_files:
            case_path = self.processed_dir / f"{case_file}.pkl"
            with open(case_path, 'rb') as f:
                data = pickle.load(f)

            coords = data['xy_1step']
            bb_min = coords.min(axis=0)
            bb_max = coords.max(axis=0)
            diagonal = np.linalg.norm(bb_max - bb_min)
            scales.append(diagonal)

        return np.median(scales)

    def fit_scalers(self) -> Dict[str, any]:
        """標準化器をfitして保存"""
        # データ分割読み込み
        splits_path = self.splits_dir / "data_splits.pkl"
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)

        train_files = splits['train']

        # 代表寸法計算
        representative_scale = self.compute_representative_scale(train_files)

        logger.info(f"Representative scale: {representative_scale:.2f}")

        # 特徴量を収集してスケーラをfit
        edge_features = []
        node_features = []
        targets = []

        for case_file in tqdm(train_files, desc="Collecting features for scaling"):
            case_path = self.processed_dir / f"{case_file}.pkl"
            with open(case_path, 'rb') as f:
                data = pickle.load(f)

            # エッジ特徴量 [φ, geometry]
            edge_feat = np.concatenate([
                data['rimd_edges_1step'],
                data['edge_geometry']
            ], axis=1)
            edge_features.append(edge_feat)

            # ノード特徴量 [σ, geometry, φ_mean, coords]
            node_feat_list = [data['rimd_nodes_1step'], data['geometry_features'], data['phi_mean']]

            if self.config.use_position_features:
                node_feat_list.append(data['xy_1step'])

            node_feat = np.concatenate(node_feat_list, axis=1)
            node_features.append(node_feat)

            # ターゲット（代表寸法で正規化）
            target_normalized = data['delta_xy'] / representative_scale
            targets.append(target_normalized)

        # 特徴量を結合
        edge_features_combined = np.vstack(edge_features)
        node_features_combined = np.vstack(node_features)
        targets_combined = np.vstack(targets)

        # スケーラをfit
        edge_scaler = StandardScaler()
        node_scaler = StandardScaler()
        target_scaler = StandardScaler()

        edge_scaler.fit(edge_features_combined)
        node_scaler.fit(node_features_combined)
        target_scaler.fit(targets_combined)

        self.scalers = {
            'edge_scaler': edge_scaler,
            'node_scaler': node_scaler,
            'target_scaler': target_scaler,
            'representative_scale': representative_scale
        }

        logger.info("Scalers fitted successfully")
        return self.scalers

    def load_splits(self) -> Dict[str, List[str]]:
        """データ分割を読み込み"""
        splits_path = self.splits_dir / "data_splits.pkl"
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        return splits