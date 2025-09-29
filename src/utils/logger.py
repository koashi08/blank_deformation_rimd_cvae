import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "rimd_cvae",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    ロガーを設定

    Args:
        name: ロガー名
        log_dir: ログファイル出力ディレクトリ
        level: ログレベル
        console: コンソール出力するか
        file_logging: ファイル出力するか

    Returns:
        設定されたロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 既存のハンドラをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # コンソールハンドラ
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ファイルハンドラ
    if file_logging and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 日時付きログファイル
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """メトリクス記録用クラス"""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"

    def log_metrics(self, epoch: int, metrics: dict, prefix: str = ""):
        """
        メトリクスをJSONL形式で記録

        Args:
            epoch: エポック番号
            metrics: メトリクス辞書
            prefix: メトリクス名のプレフィックス
        """
        import json

        log_entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **{f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        }

        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def load_metrics(self) -> list:
        """記録されたメトリクスを読み込み"""
        import json

        if not self.metrics_file.exists():
            return []

        metrics = []
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                metrics.append(json.loads(line.strip()))

        return metrics


def log_config(config, logger: Optional[logging.Logger] = None):
    """設定をログ出力"""
    if logger is None:
        logger = logging.getLogger("rimd_cvae")

    logger.info("=== Experiment Configuration ===")
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = vars(config)

    import json
    config_str = json.dumps(config_dict, indent=2, ensure_ascii=False)
    logger.info(f"\n{config_str}")
    logger.info("================================\n")