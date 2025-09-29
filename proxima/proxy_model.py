"""Proxy model building module."""

import numpy as np
from typing import Any, Optional, Union
import logging
from sklearn.base import clone
import copy

logger = logging.getLogger(__name__)


class ProxyModelBuilder:
    """Builder for creating proxy models."""

    def __init__(self, original_model: Any, config: Any):
        """
        Initialize proxy model builder.

        Args:
            original_model: Original ML model
            config: Configuration object
        """
        self.original_model = original_model
        self.config = config
        self.proxy_model = None

    def build(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Build proxy model.

        Args:
            X_train: Training data (proxy subset)
            y_train: Training labels (relabeled by original model)

        Returns:
            Trained proxy model
        """
        logger.debug(f"Building proxy model with {len(X_train)} instances")

        # Create proxy model architecture
        if self.config.proxy_model_type == "same":
            self.proxy_model = self._create_same_architecture()
        elif self.config.proxy_model_type == "simpler":
            self.proxy_model = self._create_simpler_architecture()
        elif self.config.proxy_model_type == "linear":
            self.proxy_model = self._create_linear_model()
        else:
            raise ValueError(f"Unknown proxy model type: {self.config.proxy_model_type}")

        # Train proxy model
        self._train_proxy_model(X_train, y_train)

        return self.proxy_model

    def _create_same_architecture(self) -> Any:
        """Create proxy model with same architecture as original."""
        try:
            # Try sklearn clone
            return clone(self.original_model)
        except:
            # Try deep copy
            try:
                return copy.deepcopy(self.original_model)
            except:
                # Create new instance if available
                return type(self.original_model)()

    def _create_simpler_architecture(self) -> Any:
        """Create simpler proxy model."""
        # Determine model type and create simpler version
        model_type = type(self.original_model).__name__.lower()

        if 'neural' in model_type or 'mlp' in model_type:
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=(50,),  # Simpler architecture
                max_iter=self.config.proxy_training_epochs,
                early_stopping=self.config.proxy_early_stopping,
                validation_fraction=0.1,
                n_iter_no_change=self.config.proxy_patience,
                learning_rate_init=self.config.proxy_learning_rate,
                alpha=self.config.proxy_regularization
            )
        elif 'forest' in model_type:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=50,  # Fewer trees
                max_depth=10,  # Shallower trees
                n_jobs=self.config.parallel_workers
            )
        elif 'gradient' in model_type or 'boost' in model_type:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=self.config.proxy_learning_rate
            )
        else:
            # Default to logistic regression
            return self._create_linear_model()

    def _create_linear_model(self) -> Any:
        """Create linear proxy model."""
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=1.0 / (self.config.proxy_regularization + 1e-10),
            max_iter=self.config.proxy_training_epochs * 10,
            solver='lbfgs',
            n_jobs=self.config.parallel_workers
        )

    def _train_proxy_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the proxy model."""
        if self.config.ensemble_proxies:
            self._train_ensemble(X_train, y_train)
        else:
            self._train_single(X_train, y_train)

    def _train_single(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train single proxy model."""
        try:
            if hasattr(self.proxy_model, 'fit'):
                self.proxy_model.fit(X_train, y_train)
            else:
                # For PyTorch/TensorFlow models
                self._train_deep_model(X_train, y_train)

            logger.debug("Proxy model training completed")

        except Exception as e:
            logger.error(f"Error training proxy model: {e}")
            # Fallback to simple model
            self.proxy_model = self._create_linear_model()
            self.proxy_model.fit(X_train, y_train)

    def _train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train ensemble of proxy models."""
        from sklearn.ensemble import VotingClassifier

        models = []
        for i in range(self.config.ensemble_size):
            if i == 0:
                model = self._create_same_architecture()
            elif i == 1:
                model = self._create_simpler_architecture()
            else:
                model = self._create_linear_model()

            models.append((f'model_{i}', model))

        self.proxy_model = VotingClassifier(estimators=models, voting='soft')
        self.proxy_model.fit(X_train, y_train)

    def _train_deep_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train deep learning proxy model."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            # Convert to tensors
            device = 'cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu'
            X_tensor = torch.FloatTensor(X_train).to(device)
            y_tensor = torch.LongTensor(y_train).to(device)

            # Create dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.proxy_model.parameters(),
                lr=self.config.proxy_learning_rate,
                weight_decay=self.config.proxy_regularization
            )

            # Training loop
            self.proxy_model.train()
            for epoch in range(self.config.proxy_training_epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.proxy_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            self.proxy_model.eval()

        except ImportError:
            logger.warning("PyTorch not available for deep model training")
            raise


__all__ = ['ProxyModelBuilder']