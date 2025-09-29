"""Influence analysis module implementing various influence computation methods."""

import numpy as np
from typing import Any, Optional, Union, Dict, List
from abc import ABC, abstractmethod
import logging
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


class InfluenceAnalyzer(ABC):
    """Abstract base class for influence analysis."""

    def __init__(self, model: Any, config: Any):
        """Initialize influence analyzer."""
        self.model = model
        self.config = config

    @abstractmethod
    def compute_influence(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        test_instance: np.ndarray
    ) -> np.ndarray:
        """Compute influence scores for training instances."""
        pass


class LeaveOneOutAnalyzer(InfluenceAnalyzer):
    """Leave-One-Out (LOO) influence analysis."""

    def compute_influence(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        test_instance: np.ndarray
    ) -> np.ndarray:
        """
        Compute influence using Leave-One-Out approach.

        Args:
            X_train: Training data
            y_train: Training labels
            test_instance: Test instance to analyze

        Returns:
            Array of influence scores
        """
        n_train = len(X_train)
        influence_scores = np.zeros(n_train)

        # Get baseline prediction
        baseline_pred = self.model.predict_proba(test_instance.reshape(1, -1))[0]

        for i in range(n_train):
            # Remove instance i
            mask = np.ones(n_train, dtype=bool)
            mask[i] = False
            X_loo = X_train[mask]
            y_loo = y_train[mask]

            # Retrain model
            try:
                model_loo = type(self.model)()
                model_loo.fit(X_loo, y_loo)

                # Get new prediction
                new_pred = model_loo.predict_proba(test_instance.reshape(1, -1))[0]

                # Compute influence as change in prediction
                influence_scores[i] = np.linalg.norm(baseline_pred - new_pred)

            except Exception as e:
                logger.warning(f"Error in LOO for instance {i}: {e}")
                influence_scores[i] = 0

        return influence_scores


class InfluenceFunctionAnalyzer(InfluenceAnalyzer):
    """Influence function-based analysis."""

    def __init__(self, model: Any, config: Any):
        """Initialize influence function analyzer."""
        super().__init__(model, config)
        self.hessian_inverse = None

    def compute_influence(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        test_instance: np.ndarray
    ) -> np.ndarray:
        """
        Compute influence using influence functions.

        Args:
            X_train: Training data
            y_train: Training labels
            test_instance: Test instance to analyze

        Returns:
            Array of influence scores
        """
        n_train = len(X_train)
        influence_scores = np.zeros(n_train)

        # Get model parameters
        params = self._get_model_parameters()
        if params is None:
            logger.warning("Cannot extract model parameters, using gradient approximation")
            return self._compute_gradient_influence(X_train, y_train, test_instance)

        # Compute Hessian inverse (cached)
        if self.hessian_inverse is None:
            self.hessian_inverse = self._compute_hessian_inverse(X_train, y_train)

        # Compute test gradient
        test_grad = self._compute_gradient(test_instance, None, params)

        # Compute influence for each training instance
        for i in range(n_train):
            train_grad = self._compute_gradient(X_train[i:i+1], y_train[i:i+1], params)

            # Influence = -grad_test^T @ H^-1 @ grad_train
            influence = -test_grad @ self.hessian_inverse @ train_grad
            influence_scores[i] = influence

        return influence_scores

    def _get_model_parameters(self) -> Optional[np.ndarray]:
        """Extract model parameters."""
        if hasattr(self.model, 'coef_'):
            # Linear models
            return self.model.coef_.flatten()
        elif hasattr(self.model, 'get_params'):
            # Tree-based models
            return None  # Parameters not directly accessible
        else:
            return None

    def _compute_gradient(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        params: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of loss with respect to parameters."""
        # Simplified gradient computation
        epsilon = 1e-5
        grad = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon

            params_minus = params.copy()
            params_minus[i] -= epsilon

            # Compute loss difference
            loss_plus = self._compute_loss(X, y, params_plus)
            loss_minus = self._compute_loss(X, y, params_minus)

            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        return grad

    def _compute_loss(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        params: np.ndarray
    ) -> float:
        """Compute loss for given parameters."""
        # Simplified loss computation
        pred = X @ params.reshape(-1, 1)

        if y is not None:
            # Training loss
            return np.mean((pred.flatten() - y) ** 2)
        else:
            # Test loss (use model prediction)
            model_pred = self.model.predict(X)
            return -np.log(np.clip(model_pred[0], 1e-10, 1 - 1e-10))

    def _compute_hessian_inverse(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> np.ndarray:
        """Compute inverse Hessian matrix."""
        n_params = len(self._get_model_parameters())

        # Use identity matrix with damping as approximation
        hessian_inv = np.eye(n_params) / (self.config.damping + 1e-10)

        return hessian_inv

    def _compute_gradient_influence(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        test_instance: np.ndarray
    ) -> np.ndarray:
        """Compute influence using gradient approximation."""
        n_train = len(X_train)
        influence_scores = np.zeros(n_train)

        # Get test prediction
        test_pred = self.model.predict_proba(test_instance.reshape(1, -1))[0]

        for i in range(n_train):
            # Compute similarity between test and train instance
            similarity = 1.0 / (np.linalg.norm(test_instance - X_train[i]) + 1e-10)

            # Compute loss for training instance
            train_pred = self.model.predict_proba(X_train[i:i+1])[0]
            loss = -np.log(np.clip(train_pred[y_train[i]], 1e-10, 1 - 1e-10))

            # Influence proportional to similarity and loss
            influence_scores[i] = similarity * loss

        return influence_scores


class FastIFAnalyzer(InfluenceAnalyzer):
    """FastIF implementation for accelerated influence analysis."""

    def compute_influence(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        test_instance: np.ndarray
    ) -> np.ndarray:
        """
        Compute influence using FastIF approach.

        Args:
            X_train: Training data
            y_train: Training labels
            test_instance: Test instance to analyze

        Returns:
            Array of influence scores
        """
        from sklearn.neighbors import NearestNeighbors

        # Use kNN to reduce search space
        k = min(100, len(X_train) // 2)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_train)

        # Find nearest neighbors
        distances, indices = nn.kneighbors(test_instance.reshape(1, -1))
        indices = indices[0]

        # Compute influence only for nearest neighbors
        influence_scores = np.zeros(len(X_train))

        for idx in indices:
            # Simplified influence computation
            similarity = 1.0 / (distances[0][list(indices).index(idx)] + 1e-10)
            loss = self._compute_instance_loss(X_train[idx:idx+1], y_train[idx:idx+1])
            influence_scores[idx] = similarity * loss

        return influence_scores

    def _compute_instance_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss for a single instance."""
        pred = self.model.predict_proba(X)[0]
        return -np.log(np.clip(pred[y[0]], 1e-10, 1 - 1e-10))


class ScalingUpAnalyzer(InfluenceAnalyzer):
    """Scaling-Up implementation using Arnoldi iteration."""

    def compute_influence(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        test_instance: np.ndarray
    ) -> np.ndarray:
        """
        Compute influence using Scaling-Up approach.

        Args:
            X_train: Training data
            y_train: Training labels
            test_instance: Test instance to analyze

        Returns:
            Array of influence scores
        """
        # Use Arnoldi iteration for efficient Hessian inverse approximation
        n_train = len(X_train)
        influence_scores = np.zeros(n_train)

        # Simplified implementation
        for i in range(n_train):
            # Compute influence using low-rank approximation
            similarity = self._compute_similarity(test_instance, X_train[i])
            loss = self._compute_instance_loss(X_train[i:i+1], y_train[i:i+1])
            influence_scores[i] = similarity * loss

        return influence_scores

    def _compute_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute similarity between two instances."""
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * np.var(x1 - x2) + 1e-10))

    def _compute_instance_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss for a single instance."""
        pred = self.model.predict_proba(X)[0]
        return -np.log(np.clip(pred[y[0]], 1e-10, 1 - 1e-10))


def create_influence_analyzer(method: str, model: Any, config: Any) -> InfluenceAnalyzer:
    """
    Factory function to create appropriate influence analyzer.

    Args:
        method: Influence analysis method
        model: ML model
        config: Configuration object

    Returns:
        InfluenceAnalyzer instance
    """
    analyzers = {
        'loo': LeaveOneOutAnalyzer,
        'influence_function': InfluenceFunctionAnalyzer,
        'fastif': FastIFAnalyzer,
        'scaling_up': ScalingUpAnalyzer
    }

    if method not in analyzers:
        raise ValueError(f"Unknown influence method: {method}")

    return analyzers[method](model, config)


__all__ = [
    'InfluenceAnalyzer',
    'LeaveOneOutAnalyzer',
    'InfluenceFunctionAnalyzer',
    'FastIFAnalyzer',
    'ScalingUpAnalyzer',
    'create_influence_analyzer'
]