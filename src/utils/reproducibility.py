import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    再現性のためのシード設定

    Args:
        seed: ランダムシード
        deterministic: 決定論的動作を有効にするか
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.12以降
        torch.use_deterministic_algorithms(True, warn_only=True)

    # 環境変数も設定
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    """
    利用可能なデバイスを取得

    Args:
        force_cpu: CPUを強制使用するか

    Returns:
        torch.device: 使用デバイス
    """
    if force_cpu:
        return torch.device('cpu')

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def log_system_info():
    """システム情報をログ出力"""
    import platform
    import sys

    print("=== System Information ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")

    print(f"NumPy: {np.__version__}")
    print("==========================\n")