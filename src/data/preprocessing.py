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
        # 展開形状（2D）
        required_blk_cols = ['step_blk_coord_x', 'step_blk_coord_y', 'nv_blk_coord_x', 'nv_blk_coord_y']
        missing_blk_cols = [col for col in required_blk_cols if col not in case_nodes.columns]
        if missing_blk_cols:
            raise ValueError(f"Missing blank coordinate columns for case {case_id}: {missing_blk_cols}")

        coords_1step_blk = case_nodes[['step_blk_coord_x', 'step_blk_coord_y']].values
        coords_nv_blk = case_nodes[['nv_blk_coord_x', 'nv_blk_coord_y']].values

        # 製品形状（3D）
        required_prod_cols = ['step_prod_coord_x', 'step_prod_coord_y', 'step_prod_coord_z',
                             'nv_prod_coord_x', 'nv_prod_coord_y', 'nv_prod_coord_z']
        missing_prod_cols = [col for col in required_prod_cols if col not in case_nodes.columns]
        if missing_prod_cols:
            raise ValueError(f"Missing product coordinate columns for case {case_id}: {missing_prod_cols}")

        coords_1step_prod = case_nodes[['step_prod_coord_x', 'step_prod_coord_y', 'step_prod_coord_z']].values
        coords_nv_prod = case_nodes[['nv_prod_coord_x', 'nv_prod_coord_y', 'nv_prod_coord_z']].values

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

    def compute_neighbors_from_triangles(self, triangles: np.ndarray) -> Dict[int, List[int]]:
        """
        三角形から隣接情報を計算

        Args:
            triangles: [T, 3] 三角形要素

        Returns:
            各頂点の隣接頂点辞書
        """
        from collections import defaultdict
        neighbors = defaultdict(set)

        for face in triangles:
            for i in range(3):
                vi = face[i]
                vj = face[(i + 1) % 3]
                neighbors[vi].add(vj)
                neighbors[vj].add(vi)

        return {key: list(value) for key, value in neighbors.items()}

    def compute_cotangent_weights(self, vertices: np.ndarray, triangles: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        コタンジェント重みを計算

        Args:
            vertices: [V, 3] 頂点座標
            triangles: [T, 3] 三角形要素

        Returns:
            各エッジのコタンジェント重み
        """
        edge_weights = {}

        for face in triangles:
            i, j, k = face
            vi, vj, vk = vertices[i], vertices[j], vertices[k]

            # 各辺のベクトル
            e_ij = vj - vi
            e_jk = vk - vj
            e_ki = vi - vk

            def cotangent(v1, v2):
                cos_theta = np.dot(v1, v2)
                sin_theta = np.linalg.norm(np.cross(v1, v2))
                return cos_theta / sin_theta if sin_theta > 1e-8 else 0.0

            # 各エッジの対角のコタンジェント値
            cot_alpha = cotangent(e_ki, -e_jk)  # 辺(i,j)の対角
            cot_beta = cotangent(e_ij, -e_ki)   # 辺(j,k)の対角
            cot_gamma = cotangent(e_jk, -e_ij)  # 辺(k,i)の対角

            def add_edge_weight(edge_weights, i, j, weight):
                edge = tuple(sorted((i, j)))
                if edge not in edge_weights:
                    edge_weights[edge] = 0.0
                edge_weights[edge] += weight

            add_edge_weight(edge_weights, i, j, cot_alpha)
            add_edge_weight(edge_weights, j, k, cot_beta)
            add_edge_weight(edge_weights, k, i, cot_gamma)

        return edge_weights

    def compute_deformation_gradient_3d_to_2d(self, coords_from_3d: np.ndarray, coords_to_2d: np.ndarray,
                                            triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        3D→2D変形勾配テンソルを計算し、極分解でR, Sを取得

        Args:
            coords_from_3d: [V, 3] 変形前3D座標（製品形状）
            coords_to_2d: [V, 2] 変形後2D座標（展開形状）
            triangles: [T, 3] 三角形要素

        Returns:
            R: [V, 3, 3] 回転行列
            S: [V, 3, 3] ストレッチ行列
        """
        # 2D座標を3D座標に拡張（z=0）
        coords_to_3d = np.column_stack([coords_to_2d, np.zeros(coords_from_3d.shape[0])])

        # 隣接情報とコタンジェント重みを計算
        neighbors = self.compute_neighbors_from_triangles(triangles)
        cotangent_weights = self.compute_cotangent_weights(coords_from_3d, triangles)

        # 変形勾配計算
        deformation_gradients = self._compute_deformation_gradients(
            coords_from_3d, coords_to_3d, neighbors, cotangent_weights
        )

        # 極分解
        R, S = self._decompose_deformation_gradients(deformation_gradients)

        return R, S

    def _compute_deformation_gradients(self, vertices_ref: np.ndarray, vertices_def: np.ndarray,
                                     neighbors: Dict[int, List[int]],
                                     cotangent_weights: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        コタンジェント重みを使った変形勾配計算
        """
        num_vertices = vertices_ref.shape[0]
        deformation_gradients = np.zeros((num_vertices, 3, 3))

        for i in range(num_vertices):
            if i not in neighbors:
                deformation_gradients[i] = np.eye(3)
                continue

            Ni = neighbors[i]

            # 最小二乗法で変形勾配を計算
            M = np.zeros((3, 3))
            N_mat = np.zeros((3, 3))

            for j in Ni:
                weight = cotangent_weights.get(tuple(sorted((i, j))), 1.0)
                E = vertices_ref[i] - vertices_ref[j]
                E_def = vertices_def[i] - vertices_def[j]
                M += weight * np.outer(E, E)
                N_mat += weight * np.outer(E_def, E)

            # M が特異の場合は疑似逆行列を使用
            if np.linalg.cond(M) < 1e12:
                T_i = N_mat @ np.linalg.inv(M)
            else:
                T_i = N_mat @ np.linalg.pinv(M)

            deformation_gradients[i] = T_i

        return deformation_gradients

    def _decompose_deformation_gradients(self, deformation_gradients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        変形勾配を回転部分とスケール/シアー部分に分解
        """
        num_vertices = len(deformation_gradients)
        rotations = np.zeros((num_vertices, 3, 3))
        scalings_shears = np.zeros((num_vertices, 3, 3))

        for i in range(num_vertices):
            T = deformation_gradients[i]

            # 特異値分解 (SVD)
            U, Sigma, Vt = np.linalg.svd(T)

            # 回転行列 R
            R = U @ Vt
            if np.linalg.det(R) < 0:  # 反転が生じた場合の修正
                Vt[2, :] *= -1
                R = U @ Vt

            # スケール/シアー行列 S
            S = Vt.T @ np.diag(Sigma) @ Vt

            rotations[i] = R
            scalings_shears[i] = S

        return rotations, scalings_shears

    def _compute_rotation_differences_features(self, rotations: np.ndarray,
                                             neighbors: Dict[int, List[int]]) -> List[np.ndarray]:
        """
        隣接頂点間の回転差の対数を計算
        """
        from scipy.linalg import logm
        edge_features = []

        for i, Ri in enumerate(rotations):
            if i not in neighbors:
                continue

            for j in neighbors[i]:
                if i < j:  # エッジ (i, j) の片方向のみ計算
                    Rj = rotations[j]
                    dRij = Ri.T @ Rj  # 回転差 dR_ij = R_i^T * R_j

                    try:
                        log_dRij = logm(dRij)  # 行列対数を計算
                        # 軸角表現に変換（反対称行列の成分を抽出）
                        phi = np.array([log_dRij[2,1], log_dRij[0,2], log_dRij[1,0]])
                        edge_features.append(phi)
                    except Exception:
                        # フォールバック：軸角表現で直接計算
                        theta = np.arccos(np.clip((np.trace(dRij) - 1) / 2, -1, 1))
                        if np.abs(theta) < 1e-6:
                            phi = np.zeros(3)
                        else:
                            axis = np.array([dRij[2,1] - dRij[1,2],
                                           dRij[0,2] - dRij[2,0],
                                           dRij[1,0] - dRij[0,1]]) / (2 * np.sin(theta))
                            phi = theta * axis
                        edge_features.append(phi)

        return edge_features

    def _compute_scaling_shear_features(self, scalings_shears: np.ndarray) -> List[np.ndarray]:
        """
        スケール/シアー行列の対数を計算
        """
        from scipy.linalg import logm
        node_features = []

        for i in range(len(scalings_shears)):
            S = scalings_shears[i]

            try:
                # 対称行列の対数を直接計算
                log_S = logm(S)
                # 3x3対称行列を6次元ベクトルに（上三角）
                sigma = np.array([log_S[0,0], log_S[1,1], log_S[2,2],
                                log_S[0,1], log_S[0,2], log_S[1,2]])
            except Exception:
                # フォールバック：固有値分解を使用
                eigenvals, eigenvecs = np.linalg.eigh(S)
                eigenvals = np.maximum(eigenvals, 1e-8)  # 数値安定性
                log_S = eigenvecs @ np.diag(np.log(eigenvals)) @ eigenvecs.T
                sigma = np.array([log_S[0,0], log_S[1,1], log_S[2,2],
                                log_S[0,1], log_S[0,2], log_S[1,2]])

            node_features.append(sigma)

        return node_features

    def compute_rimd_features_3d_to_2d(self, coords_prod_3d: np.ndarray, coords_blk_2d: np.ndarray,
                                     triangles: np.ndarray, edge_list: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        3D→2D変形のRIMD特徴量を計算

        Args:
            coords_prod_3d: [V, 3] 製品3D座標
            coords_blk_2d: [V, 2] 展開2D座標
            triangles: [T, 3] 三角形要素
            edge_list: [E, 2] エッジリスト

        Returns:
            RIMD特徴量辞書
        """
        # 3D→2D変形勾配の極分解
        R, S = self.compute_deformation_gradient_3d_to_2d(coords_prod_3d, coords_blk_2d, triangles)

        # 隣接情報を再計算（RIMD特徴量計算用）
        neighbors = self.compute_neighbors_from_triangles(triangles)

        # RIMD特徴量計算（サンプルコードのcompute_RIMD_featureを参考）
        edge_features = self._compute_rotation_differences_features(R, neighbors)
        node_features = self._compute_scaling_shear_features(S)

        # エッジリストを隣接情報から再構築（順序保証）
        edge_list_from_neighbors = []
        for i in range(len(R)):
            if i not in neighbors:
                continue
            for j in neighbors[i]:
                if i < j:  # 片方向のみ
                    edge_list_from_neighbors.append([i, j])

        edge_list_from_neighbors = np.array(edge_list_from_neighbors)

        return {
            'edge_features': np.array(edge_features),  # [E, 3]
            'node_features': np.array(node_features),  # [V, 6]
            'coords_prod_3d': coords_prod_3d,          # [V, 3]
            'coords_blk_2d': coords_blk_2d,            # [V, 2]
            'edge_list': edge_list_from_neighbors,     # [E, 2]
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

        # 座標の取得（製品→展開の変形を解析）
        coords_1step_prod_3d = case_data['coords_1step_prod']  # 1step解析の製品座標（3D）
        coords_1step_blk_2d = case_data['coords_1step_blk']    # 1step解析の展開座標（2D）
        coords_nv_blk_2d = case_data['coords_nv_blk']          # 逐次解析の展開座標（2D）

        # 四角形要素を三角形に分割
        triangles = self.triangulate_quad_mesh(case_data['quad_elements'])

        # エッジリスト作成
        edge_list = self.create_edge_list_from_triangles(triangles)

        # RIMD特徴量計算（製品3D→展開2Dの変形から）
        rimd_features = self.compute_rimd_features_3d_to_2d(
            coords_1step_prod_3d, coords_1step_blk_2d, triangles, edge_list
        )

        # RIMD計算で生成されたエッジリストを使用
        edge_list_rimd = rimd_features['edge_list']

        # # 補助幾何特徴量計算（展開形状ベース）
        # geom_features_1step = self.compute_geometric_features(coords_1step_blk_2d, triangles, edge_list_rimd)

        # phi_mean計算（各頂点の一環辺φの平均）
        phi_mean = self.compute_phi_mean(rimd_features['edge_features'], edge_list_rimd, len(coords_1step_blk_2d))

        # ターゲット（展開形状での座標誤差）計算
        delta_xy = coords_nv_blk_2d - coords_1step_blk_2d  # [V, 2]

        # エッジインデックス（PyTorch Geometric形式）
        edge_index = edge_list_rimd.T  # [2, E]

        return {
            'case_id': case_id,
            'rimd_edges_1step': rimd_features['edge_features'],  # [E, 3] - 製品→展開のRIMD
            'rimd_nodes_1step': rimd_features['node_features'],  # [V, 6] - 製品→展開のRIMD
            'edge_index': edge_index,  # [2, E]
            'delta_xy': delta_xy,  # [V, 2] - 展開形状での座標誤差
            'xy_1step': coords_1step_blk_2d,  # [V, 2] - 展開1step座標
            'xy_nv': coords_nv_blk_2d,  # [V, 2] - 展開逐次解析座標
            # 'geometry_features': geom_features_1step['node_features'],  # [V, Gn]
            # 'edge_geometry': geom_features_1step['edge_features'],  # [E, Ge]
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