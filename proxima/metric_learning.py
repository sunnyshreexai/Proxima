"""Metric learning module for computing distances between instances."""

import numpy as np
from typing import Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class MetricLearner(ABC):
    """Abstract base class for metric learning."""

    def __init__(self, config: Any):
        """Initialize metric learner with configuration."""
        self.config = config
        self.is_fitted = False
        self.transform_matrix = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MetricLearner':
        """Fit the metric learner."""
        pass

    @abstractmethod
    def compute_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between X1 and X2."""
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform the data."""
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to learned metric space."""
        if not self.is_fitted:
            raise ValueError("Metric learner must be fitted first")
        if self.transform_matrix is not None:
            return X @ self.transform_matrix
        return X


class NCALearner(MetricLearner):
    """Neighborhood Components Analysis metric learner."""

    def __init__(self, config: Any):
        """Initialize NCA learner."""
        super().__init__(config)

        n_components = config.metric_embedding_dim
        if n_components is None:
            n_components = 'auto'

        self.nca = NeighborhoodComponentsAnalysis(
            n_components=n_components,
            init='auto',
            warm_start=False,
            max_iter=config.metric_learning_epochs,
            tol=1e-5,
            verbose=config.verbose
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NCALearner':
        """Fit NCA metric learner."""
        logger.debug(f"Fitting NCA with {len(X)} samples")

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit NCA
        self.nca.fit(X_scaled, y)
        self.transform_matrix = self.nca.components_.T
        self.is_fitted = True

        logger.debug(f"NCA fitted with embedding dimension: {self.nca.components_.shape[0]}")

        return self

    def compute_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute distances using learned NCA metric."""
        if not self.is_fitted:
            raise ValueError("NCA must be fitted first")

        # Scale inputs
        X1_scaled = self.scaler.transform(X1)
        X2_scaled = self.scaler.transform(X2)

        # Transform to NCA space
        X1_transformed = self.nca.transform(X1_scaled)
        X2_transformed = self.nca.transform(X2_scaled)

        # Compute Euclidean distance in transformed space
        distances = cdist(X1_transformed, X2_transformed, metric='euclidean')

        return distances


class SiameseLearner(MetricLearner):
    """Siamese network-based deep metric learner."""

    def __init__(self, config: Any):
        """Initialize Siamese learner."""
        super().__init__(config)
        self.embedding_dim = config.metric_embedding_dim or 128
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SiameseLearner':
        """Fit Siamese network for metric learning."""
        logger.debug(f"Fitting Siamese network with {len(X)} samples")

        # Try to use PyTorch if available
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            self._fit_pytorch(X, y)
        except ImportError:
            logger.warning("PyTorch not available, falling back to simple embedding")
            self._fit_simple(X, y)

        self.is_fitted = True
        return self

    def _fit_pytorch(self, X: np.ndarray, y: np.ndarray):
        """Fit using PyTorch Siamese network."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        X_scaled = self.scaler.fit_transform(X)

        class SiameseNetwork(nn.Module):
            def __init__(self, input_dim, embedding_dim):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, embedding_dim)
                )

            def forward(self, x):
                return self.fc(x)

        # Initialize model
        device = 'cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu'
        self.model = SiameseNetwork(X.shape[1], self.embedding_dim).to(device)

        # Create pairs for training
        pairs, pair_labels = self._create_pairs(X_scaled, y)

        # Convert to tensors
        pairs_tensor = torch.FloatTensor(pairs).to(device)
        labels_tensor = torch.FloatTensor(pair_labels).to(device)

        # Create dataloader
        dataset = TensorDataset(pairs_tensor, labels_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.metric_batch_size,
            shuffle=True
        )

        # Training
        criterion = nn.CosineEmbeddingLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.metric_learning_rate
        )

        self.model.train()
        for epoch in range(self.config.metric_learning_epochs):
            total_loss = 0
            for batch_pairs, batch_labels in dataloader:
                optimizer.zero_grad()

                # Get embeddings
                emb1 = self.model(batch_pairs[:, 0])
                emb2 = self.model(batch_pairs[:, 1])

                # Compute loss
                loss = criterion(emb1, emb2, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.config.verbose >= 2 and (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch + 1}/{self.config.metric_learning_epochs}, "
                           f"Loss: {total_loss / len(dataloader):.4f}")

        self.model.eval()

    def _fit_simple(self, X: np.ndarray, y: np.ndarray):
        """Simple fallback fitting without deep learning."""
        X_scaled = self.scaler.fit_transform(X)

        # Use PCA for embedding
        from sklearn.decomposition import PCA

        self.model = PCA(n_components=min(self.embedding_dim, X.shape[1]))
        self.model.fit(X_scaled)
        self.transform_matrix = self.model.components_.T

    def _create_pairs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create training pairs for Siamese network."""
        n_samples = len(X)
        n_pairs = min(n_samples * 10, 10000)  # Limit number of pairs

        pairs = []
        labels = []

        for _ in range(n_pairs):
            idx1 = np.random.randint(0, n_samples)

            if np.random.rand() < 0.5:
                # Same class pair
                same_class_indices = np.where(y == y[idx1])[0]
                if len(same_class_indices) > 1:
                    idx2 = np.random.choice(same_class_indices)
                    pairs.append([X[idx1], X[idx2]])
                    labels.append(1)  # Similar
            else:
                # Different class pair
                diff_class_indices = np.where(y != y[idx1])[0]
                if len(diff_class_indices) > 0:
                    idx2 = np.random.choice(diff_class_indices)
                    pairs.append([X[idx1], X[idx2]])
                    labels.append(-1)  # Dissimilar

        return np.array(pairs), np.array(labels)

    def compute_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute distances using learned Siamese metric."""
        if not self.is_fitted:
            raise ValueError("Siamese learner must be fitted first")

        X1_scaled = self.scaler.transform(X1)
        X2_scaled = self.scaler.transform(X2)

        if hasattr(self.model, 'forward'):
            # PyTorch model
            import torch
            device = next(self.model.parameters()).device

            with torch.no_grad():
                X1_tensor = torch.FloatTensor(X1_scaled).to(device)
                X2_tensor = torch.FloatTensor(X2_scaled).to(device)

                emb1 = self.model(X1_tensor).cpu().numpy()
                emb2 = self.model(X2_tensor).cpu().numpy()
        else:
            # Fallback model
            emb1 = self.model.transform(X1_scaled)
            emb2 = self.model.transform(X2_scaled)

        # Compute cosine distance
        distances = cdist(emb1, emb2, metric='cosine')

        return distances


class EuclideanLearner(MetricLearner):
    """Simple Euclidean distance metric."""

    def __init__(self, config: Any):
        """Initialize Euclidean learner."""
        super().__init__(config)
        self.scaler = StandardScaler() if config.normalize_features else None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EuclideanLearner':
        """Fit scaler if normalization is enabled."""
        if self.scaler is not None:
            self.scaler.fit(X)
        self.is_fitted = True
        return self

    def compute_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances."""
        if self.scaler is not None:
            X1 = self.scaler.transform(X1)
            X2 = self.scaler.transform(X2)

        return cdist(X1, X2, metric='euclidean')


class CosineLearner(MetricLearner):
    """Cosine distance metric."""

    def __init__(self, config: Any):
        """Initialize Cosine learner."""
        super().__init__(config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CosineLearner':
        """No fitting needed for cosine distance."""
        self.is_fitted = True
        return self

    def compute_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute cosine distances."""
        return cdist(X1, X2, metric='cosine')


class ManhattanLearner(MetricLearner):
    """Manhattan (L1) distance metric."""

    def __init__(self, config: Any):
        """Initialize Manhattan learner."""
        super().__init__(config)
        self.scaler = StandardScaler() if config.normalize_features else None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ManhattanLearner':
        """Fit scaler if normalization is enabled."""
        if self.scaler is not None:
            self.scaler.fit(X)
        self.is_fitted = True
        return self

    def compute_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute Manhattan distances."""
        if self.scaler is not None:
            X1 = self.scaler.transform(X1)
            X2 = self.scaler.transform(X2)

        return cdist(X1, X2, metric='cityblock')


def create_metric_learner(metric_type: str, config: Any) -> MetricLearner:
    """
    Factory function to create appropriate metric learner.

    Args:
        metric_type: Type of metric ('nca', 'siamese', 'euclidean', 'cosine', 'manhattan')
        config: Configuration object

    Returns:
        MetricLearner instance
    """
    learners = {
        'nca': NCALearner,
        'siamese': SiameseLearner,
        'euclidean': EuclideanLearner,
        'cosine': CosineLearner,
        'manhattan': ManhattanLearner
    }

    if metric_type not in learners:
        raise ValueError(f"Unknown metric type: {metric_type}")

    return learners[metric_type](config)


__all__ = [
    'MetricLearner',
    'NCALearner',
    'SiameseLearner',
    'EuclideanLearner',
    'CosineLearner',
    'ManhattanLearner',
    'create_metric_learner'
]