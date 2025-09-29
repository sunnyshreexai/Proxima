"""Utility functions for Proxima package."""

import numpy as np
import random
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    logger.info(f"Random seed set to {seed}")


def get_device() -> str:
    """Get available device (CPU/GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            logger.info("Using CPU")
    except ImportError:
        device = 'cpu'
        logger.info("PyTorch not available, using CPU")

    return device


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        logger.debug(f"{self.name} took {self.elapsed:.2f} seconds")


def compute_training_loss(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128
) -> np.ndarray:
    """
    Compute training loss for each instance.

    Args:
        model: Trained ML model
        X: Training data
        y: Training labels
        batch_size: Batch size for processing

    Returns:
        Array of training losses
    """
    n_samples = len(X)
    losses = np.zeros(n_samples)

    for i in range(0, n_samples, batch_size):
        batch_X = X[i:min(i + batch_size, n_samples)]
        batch_y = y[i:min(i + batch_size, n_samples)]

        # Get predictions
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(batch_X)
            batch_losses = -np.log(np.clip(
                [probs[j, batch_y[j]] for j in range(len(batch_y))],
                1e-10, 1 - 1e-10
            ))
        else:
            preds = model.predict(batch_X)
            batch_losses = np.abs(preds - batch_y)

        losses[i:min(i + batch_size, n_samples)] = batch_losses

    return losses


def evaluate_influence_accuracy(
    predicted: Union[List, np.ndarray],
    ground_truth: Union[List, np.ndarray]
) -> float:
    """
    Evaluate accuracy of influence predictions.

    Args:
        predicted: Predicted influential instances
        ground_truth: Ground truth influential instances

    Returns:
        Accuracy score (0-1)
    """
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)

    # Count matches
    matches = len(set(predicted) & set(ground_truth))

    # Calculate accuracy
    accuracy = matches / len(ground_truth) if len(ground_truth) > 0 else 0

    return accuracy


def visualize_influence_scores(
    influence_scores: np.ndarray,
    top_k: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize influence scores.

    Args:
        influence_scores: Array of influence scores
        top_k: Number of top instances to highlight
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Distribution of all scores
        ax1.hist(influence_scores, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Influence Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Influence Scores')
        ax1.grid(True, alpha=0.3)

        # Top-k scores
        top_k_indices = np.argsort(np.abs(influence_scores))[-top_k:][::-1]
        top_k_scores = influence_scores[top_k_indices]

        ax2.barh(range(top_k), top_k_scores, alpha=0.7)
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels([f'Instance {idx}' for idx in top_k_indices])
        ax2.set_xlabel('Influence Score')
        ax2.set_title(f'Top-{top_k} Influential Instances')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    except ImportError:
        logger.warning("Matplotlib not available for visualization")


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from file."""
    path = Path(path)

    if path.suffix == '.npz':
        data = np.load(path)
        return data['X'], data['y']
    elif path.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y']
    elif path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_results(
    results: Dict[str, Any],
    path: str,
    format: str = 'json'
) -> None:
    """Save results to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'csv':
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Results saved to {path}")


def compute_pairwise_distances(
    X1: np.ndarray,
    X2: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """Compute pairwise distances between two sets of points."""
    from scipy.spatial.distance import cdist
    return cdist(X1, X2, metric=metric)


def normalize_data(
    X: np.ndarray,
    method: str = 'standard'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data.

    Args:
        X: Data to normalize
        method: Normalization method

    Returns:
        Normalized data and normalization parameters
    """
    if method == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-10
        X_norm = (X - mean) / std
        params = {'mean': mean, 'std': std}
    elif method == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        range_val = max_val - min_val + 1e-10
        X_norm = (X - min_val) / range_val
        params = {'min': min_val, 'max': max_val}
    else:
        X_norm = X
        params = {}

    return X_norm, params


def check_model_compatibility(model: Any) -> Dict[str, bool]:
    """Check model compatibility with Proxima."""
    compatibility = {}

    # Check basic requirements
    compatibility['has_fit'] = hasattr(model, 'fit')
    compatibility['has_predict'] = hasattr(model, 'predict')
    compatibility['has_predict_proba'] = hasattr(model, 'predict_proba')

    # Check for parameter access
    compatibility['has_coef'] = hasattr(model, 'coef_')
    compatibility['has_params'] = hasattr(model, 'get_params')

    # Check model type
    model_type = type(model).__name__
    compatibility['model_type'] = model_type
    compatibility['is_sklearn'] = 'sklearn' in str(type(model).__module__)
    compatibility['is_torch'] = 'torch' in str(type(model).__module__)
    compatibility['is_tf'] = 'tensorflow' in str(type(model).__module__)

    return compatibility


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic dataset for testing."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        flip_y=noise,
        random_state=seed
    )

    return X, y


__all__ = [
    'set_seed',
    'get_device',
    'Timer',
    'compute_training_loss',
    'evaluate_influence_accuracy',
    'visualize_influence_scores',
    'load_dataset',
    'save_results',
    'compute_pairwise_distances',
    'normalize_data',
    'check_model_compatibility',
    'create_synthetic_dataset'
]