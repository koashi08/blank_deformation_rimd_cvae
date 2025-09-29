import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import json

from ..training.metrics import RIMDMetrics
from ..training.trainer import RIMDTrainer
from ..data.dataset import RIMDDataModule

logger = logging.getLogger(__name__)


class RIMDEvaluator:
    """RIMD予測モデルの包括的評価を行うクラス"""

    def __init__(self, config, experiment_manager):
        self.config = config
        self.exp_manager = experiment_manager
        self.device = experiment_manager.device
        self.metrics_calculator = RIMDMetrics(config)

        # 評価結果格納
        self.evaluation_results = {}

    def evaluate_model(self, model_path: str, datamodule: RIMDDataModule,
                      splits: List[str] = ['test'], save_predictions: bool = True) -> Dict[str, Any]:
        """
        モデルの包括的評価を実行

        Args:
            model_path: 学習済みモデルパス
            datamodule: データモジュール
            splits: 評価対象データ分割 ['train', 'val', 'test']
            save_predictions: 予測結果を保存するか

        Returns:
            評価結果辞書
        """
        # モデル読み込み
        trainer = RIMDTrainer(self.config, self.exp_manager)
        trainer.setup(datamodule)
        trainer.load_checkpoint(model_path)

        logger.info(f"Evaluating model: {model_path}")

        results = {}

        for split in splits:
            logger.info(f"Evaluating on {split} set...")

            # データローダー取得
            dataloader = getattr(datamodule, f"{split}_dataloader")()

            # 予測実行
            predictions = trainer.predict(dataloader)

            # メトリクス計算
            split_results = self._compute_comprehensive_metrics(
                predictions, split, save_predictions
            )

            results[split] = split_results

            logger.info(f"{split.upper()} Results:")
            logger.info(f"  RMSE: {split_results['metrics']['rmse_mm']:.2f} mm")
            logger.info(f"  MAE: {split_results['metrics']['mae_mm']:.2f} mm")
            logger.info(f"  Gain: {split_results['metrics']['gain_percent']:.1f}%")

        # 統合結果
        evaluation_summary = self._create_evaluation_summary(results)

        # 結果保存
        self._save_evaluation_results(results, evaluation_summary)

        return {
            'detailed_results': results,
            'summary': evaluation_summary
        }

    def _compute_comprehensive_metrics(self, predictions: Dict[str, np.ndarray],
                                     split: str, save_predictions: bool) -> Dict[str, Any]:
        """包括的メトリクスを計算"""
        pred_array = predictions['predictions']
        target_array = predictions.get('targets')
        case_ids = predictions.get('case_ids', [])

        if target_array is None:
            logger.warning(f"No targets available for {split} split")
            return {'predictions': pred_array, 'case_ids': case_ids}

        # 基本メトリクス
        detailed_metrics = self.metrics_calculator.compute_detailed_metrics(
            pred_array, target_array,
            coords_1step=None,  # TODO: データから取得
            coords_nv=None,     # TODO: データから取得
            case_ids=case_ids if case_ids else None
        )

        # 統計的分析
        statistical_analysis = self._compute_statistical_analysis(pred_array, target_array)

        # 予測品質分析
        prediction_quality = self._analyze_prediction_quality(pred_array, target_array)

        # ケース別分析
        case_analysis = self._analyze_by_case(pred_array, target_array, case_ids)

        results = {
            'metrics': detailed_metrics,
            'statistical_analysis': statistical_analysis,
            'prediction_quality': prediction_quality,
            'case_analysis': case_analysis,
            'predictions': pred_array,
            'targets': target_array,
            'case_ids': case_ids
        }

        # 予測結果保存
        if save_predictions:
            self._save_predictions(results, split)

        return results

    def _compute_statistical_analysis(self, predictions: np.ndarray,
                                    targets: np.ndarray) -> Dict[str, Any]:
        """統計的分析を実行"""
        residuals = predictions - targets

        # 基本統計
        residual_stats = {
            'mean': np.mean(residuals, axis=0).tolist(),
            'std': np.std(residuals, axis=0).tolist(),
            'skewness': [float(stats.skew(residuals[:, i])) for i in range(residuals.shape[1])],
            'kurtosis': [float(stats.kurtosis(residuals[:, i])) for i in range(residuals.shape[1])]
        }

        # 正規性検定
        normality_tests = {}
        for i, coord in enumerate(['x', 'y']):
            _, p_value = stats.shapiro(residuals[:, i])
            normality_tests[f'{coord}_normal'] = {
                'p_value': float(p_value),
                'is_normal': bool(p_value > 0.05)
            }

        # 相関分析
        correlations = {}
        for i, coord in enumerate(['x', 'y']):
            r, p = stats.pearsonr(predictions[:, i], targets[:, i])
            correlations[f'{coord}_correlation'] = {
                'r': float(r),
                'r_squared': float(r**2),
                'p_value': float(p)
            }

        return {
            'residual_statistics': residual_stats,
            'normality_tests': normality_tests,
            'correlations': correlations
        }

    def _analyze_prediction_quality(self, predictions: np.ndarray,
                                  targets: np.ndarray) -> Dict[str, Any]:
        """予測品質の詳細分析"""
        errors = np.linalg.norm(predictions - targets, axis=1)

        # エラー分布分析
        error_percentiles = np.percentile(errors, [10, 25, 50, 75, 90, 95, 99])

        # 高精度・低精度サンプル特定
        high_accuracy_threshold = np.percentile(errors, 25)
        low_accuracy_threshold = np.percentile(errors, 75)

        quality_analysis = {
            'error_percentiles': {
                'p10': float(error_percentiles[0]),
                'p25': float(error_percentiles[1]),
                'p50': float(error_percentiles[2]),
                'p75': float(error_percentiles[3]),
                'p90': float(error_percentiles[4]),
                'p95': float(error_percentiles[5]),
                'p99': float(error_percentiles[6])
            },
            'accuracy_categories': {
                'high_accuracy_count': int(np.sum(errors <= high_accuracy_threshold)),
                'medium_accuracy_count': int(np.sum((errors > high_accuracy_threshold) & (errors <= low_accuracy_threshold))),
                'low_accuracy_count': int(np.sum(errors > low_accuracy_threshold)),
                'high_accuracy_threshold': float(high_accuracy_threshold),
                'low_accuracy_threshold': float(low_accuracy_threshold)
            }
        }

        return quality_analysis

    def _analyze_by_case(self, predictions: np.ndarray, targets: np.ndarray,
                        case_ids: List[str]) -> Dict[str, Any]:
        """ケース別分析"""
        if not case_ids:
            return {}

        case_analysis = {}
        unique_cases = list(set(case_ids))

        for case_id in unique_cases:
            case_mask = np.array([cid == case_id for cid in case_ids])
            case_pred = predictions[case_mask]
            case_target = targets[case_mask]

            if len(case_pred) > 0:
                case_errors = np.linalg.norm(case_pred - case_target, axis=1)

                case_analysis[case_id] = {
                    'num_samples': len(case_pred),
                    'mean_error': float(np.mean(case_errors)),
                    'std_error': float(np.std(case_errors)),
                    'max_error': float(np.max(case_errors)),
                    'median_error': float(np.median(case_errors)),
                    'rmse': float(np.sqrt(np.mean(case_errors**2))),
                    'mae': float(np.mean(case_errors))
                }

        # ケース間比較
        if len(case_analysis) > 1:
            case_means = [stats['mean_error'] for stats in case_analysis.values()]
            case_comparison = {
                'best_case': min(case_analysis.items(), key=lambda x: x[1]['mean_error'])[0],
                'worst_case': max(case_analysis.items(), key=lambda x: x[1]['mean_error'])[0],
                'case_variability': {
                    'mean_error_std': float(np.std(case_means)),
                    'mean_error_range': float(np.max(case_means) - np.min(case_means))
                }
            }
            case_analysis['_summary'] = case_comparison

        return case_analysis

    def _create_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """評価結果のサマリーを作成"""
        summary = {
            'overall_performance': {},
            'split_comparison': {},
            'key_findings': []
        }

        # 全体性能
        if 'test' in results:
            test_metrics = results['test']['metrics']
            summary['overall_performance'] = {
                'rmse_mm': test_metrics['rmse_mm'],
                'mae_mm': test_metrics['mae_mm'],
                'median_error_mm': test_metrics['median_error_mm'],
                'p95_error_mm': test_metrics['p95_error_mm'],
                'gain_percent': test_metrics.get('gain_percent', 0.0)
            }

        # 分割間比較
        if len(results) > 1:
            for split, split_results in results.items():
                if 'metrics' in split_results:
                    summary['split_comparison'][split] = {
                        'rmse': split_results['metrics']['rmse_mm'],
                        'mae': split_results['metrics']['mae_mm']
                    }

        # 主要な発見事項
        findings = []

        # 性能レベル判定
        if 'test' in results:
            test_rmse = results['test']['metrics']['rmse_mm']
            if test_rmse < 1.0:
                findings.append("Excellent prediction accuracy (RMSE < 1.0mm)")
            elif test_rmse < 2.0:
                findings.append("Good prediction accuracy (RMSE < 2.0mm)")
            elif test_rmse < 5.0:
                findings.append("Moderate prediction accuracy (RMSE < 5.0mm)")
            else:
                findings.append("Needs improvement (RMSE >= 5.0mm)")

        # オーバーフィッティング検出
        if 'train' in results and 'test' in results:
            train_rmse = results['train']['metrics']['rmse_mm']
            test_rmse = results['test']['metrics']['rmse_mm']
            if test_rmse / train_rmse > 1.5:
                findings.append("Potential overfitting detected")

        summary['key_findings'] = findings

        return summary

    def _save_predictions(self, results: Dict[str, Any], split: str):
        """予測結果を保存"""
        predictions_dir = self.exp_manager.exp_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)

        # NumPy形式で保存
        np.save(predictions_dir / f"{split}_predictions.npy", results['predictions'])
        np.save(predictions_dir / f"{split}_targets.npy", results['targets'])

        # CSVでも保存（可視化用）
        if results['case_ids']:
            df = pd.DataFrame({
                'case_id': results['case_ids'],
                'pred_x': results['predictions'][:, 0],
                'pred_y': results['predictions'][:, 1],
                'target_x': results['targets'][:, 0],
                'target_y': results['targets'][:, 1],
                'error_x': results['predictions'][:, 0] - results['targets'][:, 0],
                'error_y': results['predictions'][:, 1] - results['targets'][:, 1],
                'euclidean_error': np.linalg.norm(results['predictions'] - results['targets'], axis=1)
            })
            df.to_csv(predictions_dir / f"{split}_predictions.csv", index=False)

    def _save_evaluation_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """評価結果を保存"""
        eval_dir = self.exp_manager.exp_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        # 詳細結果（NumPy配列は除く）
        results_to_save = {}
        for split, split_results in results.items():
            results_to_save[split] = {
                'metrics': split_results['metrics'],
                'statistical_analysis': split_results['statistical_analysis'],
                'prediction_quality': split_results['prediction_quality'],
                'case_analysis': split_results['case_analysis']
            }

        with open(eval_dir / "detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)

        # サマリー
        with open(eval_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # レポート生成
        report = self._generate_evaluation_report(summary, results_to_save)
        with open(eval_dir / "evaluation_report.md", 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Evaluation results saved to {eval_dir}")

    def _generate_evaluation_report(self, summary: Dict[str, Any],
                                   results: Dict[str, Any]) -> str:
        """評価レポートを生成"""
        report = []
        report.append("# RIMD Model Evaluation Report\n")

        # 全体性能
        if summary['overall_performance']:
            perf = summary['overall_performance']
            report.append("## Overall Performance")
            report.append(f"- **RMSE**: {perf['rmse_mm']:.2f} mm")
            report.append(f"- **MAE**: {perf['mae_mm']:.2f} mm")
            report.append(f"- **Median Error**: {perf['median_error_mm']:.2f} mm")
            report.append(f"- **P95 Error**: {perf['p95_error_mm']:.2f} mm")
            if perf.get('gain_percent'):
                report.append(f"- **Improvement Gain**: {perf['gain_percent']:.1f}%")
            report.append("")

        # 主要な発見事項
        if summary['key_findings']:
            report.append("## Key Findings")
            for finding in summary['key_findings']:
                report.append(f"- {finding}")
            report.append("")

        # 分割間比較
        if summary['split_comparison']:
            report.append("## Performance by Data Split")
            report.append("| Split | RMSE (mm) | MAE (mm) |")
            report.append("|-------|-----------|----------|")
            for split, metrics in summary['split_comparison'].items():
                report.append(f"| {split.capitalize()} | {metrics['rmse']:.2f} | {metrics['mae']:.2f} |")
            report.append("")

        # 詳細分析（テストセットの場合）
        if 'test' in results:
            test_results = results['test']

            # 統計的分析
            if 'statistical_analysis' in test_results:
                stat_analysis = test_results['statistical_analysis']
                report.append("## Statistical Analysis")

                # 残差統計
                residual_stats = stat_analysis['residual_statistics']
                report.append("### Residual Statistics")
                report.append("| Coordinate | Mean | Std | Skewness | Kurtosis |")
                report.append("|------------|------|-----|----------|----------|")
                for i, coord in enumerate(['X', 'Y']):
                    report.append(f"| {coord} | {residual_stats['mean'][i]:.3f} | "
                                f"{residual_stats['std'][i]:.3f} | "
                                f"{residual_stats['skewness'][i]:.3f} | "
                                f"{residual_stats['kurtosis'][i]:.3f} |")
                report.append("")

        return '\n'.join(report)


class ModelComparator:
    """複数モデルの比較分析を行うクラス"""

    def __init__(self, experiment_manager):
        self.exp_manager = experiment_manager
        self.comparison_results = {}

    def compare_models(self, model_configs: List[Dict[str, Any]],
                      datamodule: RIMDDataModule) -> Dict[str, Any]:
        """
        複数モデルの比較評価

        Args:
            model_configs: モデル設定リスト [{'name': str, 'config': Config, 'model_path': str}]
            datamodule: データモジュール

        Returns:
            比較結果
        """
        results = {}

        for model_config in model_configs:
            model_name = model_config['name']
            config = model_config['config']
            model_path = model_config['model_path']

            logger.info(f"Evaluating model: {model_name}")

            # 評価実行
            evaluator = RIMDEvaluator(config, self.exp_manager)
            eval_results = evaluator.evaluate_model(
                model_path, datamodule, splits=['test'], save_predictions=False
            )

            results[model_name] = eval_results['detailed_results']['test']['metrics']

        # 比較分析
        comparison = self._analyze_model_comparison(results)

        # 結果保存
        self._save_comparison_results(results, comparison)

        return {
            'individual_results': results,
            'comparison_analysis': comparison
        }

    def _analyze_model_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """モデル比較分析"""
        metrics_to_compare = ['rmse_mm', 'mae_mm', 'median_error_mm', 'p95_error_mm']

        comparison = {
            'best_models': {},
            'performance_ranking': {},
            'statistical_significance': {}
        }

        # 最良モデル特定
        for metric in metrics_to_compare:
            values = [(name, res[metric]) for name, res in results.items() if metric in res]
            if values:
                best_model = min(values, key=lambda x: x[1])
                comparison['best_models'][metric] = {
                    'model': best_model[0],
                    'value': best_model[1]
                }

        # ランキング作成
        for metric in metrics_to_compare:
            ranking = sorted(results.items(), key=lambda x: x[1].get(metric, float('inf')))
            comparison['performance_ranking'][metric] = [
                {'model': name, 'value': res.get(metric, None)}
                for name, res in ranking
            ]

        return comparison

    def _save_comparison_results(self, results: Dict[str, Any],
                               comparison: Dict[str, Any]):
        """比較結果を保存"""
        comparison_dir = self.exp_manager.exp_dir / "model_comparison"
        comparison_dir.mkdir(exist_ok=True)

        # 結果保存
        with open(comparison_dir / "comparison_results.json", 'w', encoding='utf-8') as f:
            json.dump({'results': results, 'comparison': comparison},
                     f, indent=2, ensure_ascii=False)

        # 比較レポート生成
        report = self._generate_comparison_report(results, comparison)
        with open(comparison_dir / "comparison_report.md", 'w', encoding='utf-8') as f:
            f.write(report)

    def _generate_comparison_report(self, results: Dict[str, Any],
                                  comparison: Dict[str, Any]) -> str:
        """比較レポートを生成"""
        report = []
        report.append("# Model Comparison Report\n")

        # 最良モデル
        report.append("## Best Models by Metric")
        for metric, best in comparison['best_models'].items():
            report.append(f"- **{metric}**: {best['model']} ({best['value']:.3f})")
        report.append("")

        # 性能ランキング
        report.append("## Performance Ranking")
        for metric in ['rmse_mm', 'mae_mm']:
            report.append(f"### {metric.upper()}")
            report.append("| Rank | Model | Value |")
            report.append("|------|-------|-------|")
            for i, entry in enumerate(comparison['performance_ranking'][metric], 1):
                if entry['value'] is not None:
                    report.append(f"| {i} | {entry['model']} | {entry['value']:.3f} |")
            report.append("")

        return '\n'.join(report)