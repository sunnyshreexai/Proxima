"""Core Proxima implementation for proxy model-based influence analysis."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import time
import warnings
from pathlib import Path
import pickle
import json

from .config import ProximaConfig, load_config
from .metric_learning import MetricLearner, create_metric_learner
from .proxy_model import ProxyModelBuilder
from .influence_analysis import InfluenceAnalyzer, create_influence_analyzer
from .utils import (
    compute_training_loss,
    set_seed,
    get_device,
    Timer,
    evaluate_influence_accuracy
)

logger = logging.getLogger(__name__)


class Proxima:
    """
    Main class for proxy model-based influence analysis.

    This implements the Proxima approach from the paper, using a subset of
    training instances identified via loss-over-distance ratio to create
    a simpler proxy model for efficient influence analysis.
    """

    def __init__(
        self,
        model: Any,
        dataset: np.ndarray,
        labels: np.ndarray,
        config: Optional[Union[ProximaConfig, Dict, str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Proxima.

        Args:
            model: Original ML model (sklearn, torch, or tf compatible)
            dataset: Training dataset (n_samples, n_features)
            labels: Training labels
            config: Configuration object, dict, path, or preset name
            seed: Random seed for reproducibility
        """
        self.original_model = model
        self.dataset = dataset
        self.labels = labels
        self.config = load_config(config) if config else ProximaConfig()
        self.config.validate()

        if seed is not None:
            set_seed(seed)

        # Components
        self.metric_learner: Optional[MetricLearner] = None
        self.proxy_model_builder: Optional[ProxyModelBuilder] = None
        self.influence_analyzer: Optional[InfluenceAnalyzer] = None

        # Preprocessing results
        self.training_losses: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None

        # Proxy model components
        self.proxy_subset: Optional[np.ndarray] = None
        self.proxy_subset_indices: Optional[np.ndarray] = None
        self.proxy_labels: Optional[np.ndarray] = None
        self.proxy_model: Optional[Any] = None

        # Results storage
        self.influence_scores: Dict[int, np.ndarray] = {}
        self.top_k_influential: Dict[int, np.ndarray] = {}

        # Cache
        self.cache: Dict[str, Any] = {}

        if self.config.verbose >= 1:
            self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging based on verbosity level."""
        level = logging.INFO if self.config.verbose == 1 else logging.DEBUG
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def preprocess(self) -> None:
        """
        Preprocessing phase: metric learning and training loss computation.

        This phase needs to be performed only once per dataset.
        """
        logger.info("Starting preprocessing phase")

        # Step 1: Metric Learning
        logger.info("Learning distance metric")
        self.metric_learner = create_metric_learner(
            metric_type=self.config.metric_type,
            config=self.config
        )

        with Timer("Metric learning"):
            self.metric_learner.fit(self.dataset, self.labels)

        # Step 2: Compute training losses
        logger.info("Computing training losses")
        with Timer("Training loss computation"):
            self.training_losses = compute_training_loss(
                model=self.original_model,
                X=self.dataset,
                y=self.labels,
                batch_size=self.config.batch_size
            )

        logger.info(f"Mean training loss: {np.mean(self.training_losses):.4f}")

        # Cache results if enabled
        if self.config.save_intermediate:
            self._save_preprocessing_results()

    def build_proxy_model(
        self,
        test_instance: np.ndarray,
        test_index: Optional[int] = None
    ) -> Tuple[np.ndarray, Any]:
        """
        Build proxy model for a specific test instance.

        Args:
            test_instance: Test instance for influence analysis
            test_index: Optional index for caching

        Returns:
            Tuple of (proxy_subset_indices, proxy_model)
        """
        logger.info("Building proxy model")

        # Step 1: Identify proxy subset
        proxy_subset_indices = self._identify_proxy_subset(test_instance)

        # Step 2: Train proxy model
        proxy_model = self._train_proxy_model(proxy_subset_indices)

        # Cache if index provided
        if test_index is not None:
            cache_key = f"proxy_{test_index}"
            self.cache[cache_key] = {
                'indices': proxy_subset_indices,
                'model': proxy_model
            }

        return proxy_subset_indices, proxy_model

    def _identify_proxy_subset(self, test_instance: np.ndarray) -> np.ndarray:
        """
        Identify proxy subset using loss-over-distance ratio.

        Args:
            test_instance: Test instance

        Returns:
            Indices of instances in proxy subset
        """
        # Compute distances from test instance to all training instances
        distances = self.metric_learner.compute_distance(
            test_instance.reshape(1, -1),
            self.dataset
        ).flatten()

        # Compute loss-over-distance ratio
        ratios = self.training_losses / (self.config.lambda_ratio * distances + 1e-10)

        # Find threshold using adaptive approach
        if self.config.adaptive_threshold:
            threshold = self._find_adaptive_threshold(ratios)
        else:
            threshold = self.config.threshold_initial

        # Select instances below threshold
        proxy_indices = np.where(ratios < threshold)[0]

        # Ensure minimum subset size
        min_size = self.config.top_k * self.config.min_subset_size_factor
        if len(proxy_indices) < min_size:
            # Add more instances based on ratio ranking
            sorted_indices = np.argsort(ratios)
            proxy_indices = sorted_indices[:min_size]

        logger.info(f"Proxy subset size: {len(proxy_indices)} "
                   f"({100 * len(proxy_indices) / len(self.dataset):.1f}% of dataset)")

        return proxy_indices

    def _find_adaptive_threshold(self, ratios: np.ndarray) -> float:
        """
        Find adaptive threshold for subset selection.

        Args:
            ratios: Loss-over-distance ratios

        Returns:
            Optimal threshold value
        """
        threshold = self.config.threshold_initial
        previous_indices = None
        convergence_threshold = int(self.config.top_k * self.config.convergence_ratio)

        for iteration in range(self.config.max_iterations):
            current_indices = np.where(ratios < threshold)[0]

            # Check minimum size
            min_size = self.config.top_k * self.config.min_subset_size_factor
            if len(current_indices) < min_size:
                threshold += self.config.threshold_increment
                continue

            # Check convergence
            if previous_indices is not None:
                # Get top-k from each set based on ratios
                prev_top_k = previous_indices[np.argsort(ratios[previous_indices])[:self.config.top_k]]
                curr_top_k = current_indices[np.argsort(ratios[current_indices])[:self.config.top_k]]

                overlap = len(np.intersect1d(prev_top_k, curr_top_k))

                if overlap >= convergence_threshold:
                    logger.debug(f"Converged at threshold {threshold:.3f} "
                               f"with {overlap}/{self.config.top_k} overlap")
                    break

            previous_indices = current_indices
            threshold += self.config.threshold_increment

        return threshold

    def _train_proxy_model(self, proxy_subset_indices: np.ndarray) -> Any:
        """
        Train proxy model on the proxy subset.

        Args:
            proxy_subset_indices: Indices of proxy subset

        Returns:
            Trained proxy model
        """
        # Get proxy subset
        proxy_subset = self.dataset[proxy_subset_indices]

        # Relabel using original model predictions
        proxy_labels = self.original_model.predict(proxy_subset)

        # Build proxy model
        self.proxy_model_builder = ProxyModelBuilder(
            original_model=self.original_model,
            config=self.config
        )

        proxy_model = self.proxy_model_builder.build(
            X_train=proxy_subset,
            y_train=proxy_labels
        )

        # Store for later use
        self.proxy_subset = proxy_subset
        self.proxy_subset_indices = proxy_subset_indices
        self.proxy_labels = proxy_labels
        self.proxy_model = proxy_model

        return proxy_model

    def analyze_influence(
        self,
        test_instance: np.ndarray,
        test_label: Optional[int] = None,
        proxy_model: Optional[Any] = None,
        proxy_subset_indices: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform influence analysis for a test instance.

        Args:
            test_instance: Test instance to analyze
            test_label: True label of test instance (optional)
            proxy_model: Pre-built proxy model (optional)
            proxy_subset_indices: Pre-computed proxy subset indices (optional)

        Returns:
            Dictionary containing influence analysis results
        """
        start_time = time.time()

        # Build proxy model if not provided
        if proxy_model is None or proxy_subset_indices is None:
            proxy_subset_indices, proxy_model = self.build_proxy_model(test_instance)

        # Get proxy subset data
        proxy_subset = self.dataset[proxy_subset_indices]
        proxy_labels = self.original_model.predict(proxy_subset)

        # Create influence analyzer
        self.influence_analyzer = create_influence_analyzer(
            method=self.config.influence_method,
            model=proxy_model,
            config=self.config
        )

        # Compute influence scores
        influence_scores = self.influence_analyzer.compute_influence(
            X_train=proxy_subset,
            y_train=proxy_labels,
            test_instance=test_instance
        )

        # Get top-k influential instances
        top_k_indices_local = np.argsort(np.abs(influence_scores))[-self.config.top_k:][::-1]
        top_k_indices_global = proxy_subset_indices[top_k_indices_local]

        # Compile results
        results = {
            'top_k_indices': top_k_indices_global.tolist(),
            'top_k_scores': influence_scores[top_k_indices_local].tolist(),
            'proxy_subset_size': len(proxy_subset_indices),
            'proxy_subset_ratio': len(proxy_subset_indices) / len(self.dataset),
            'time_taken': time.time() - start_time,
            'test_prediction': int(proxy_model.predict(test_instance.reshape(1, -1))[0])
        }

        if test_label is not None:
            results['test_label'] = int(test_label)
            results['correct_prediction'] = results['test_prediction'] == test_label

        # Add detailed scores if verbose
        if self.config.verbose >= 2:
            results['all_proxy_indices'] = proxy_subset_indices.tolist()
            results['all_influence_scores'] = influence_scores.tolist()

        return results

    def batch_analyze(
        self,
        test_instances: np.ndarray,
        test_labels: Optional[np.ndarray] = None,
        parallel: bool = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze influence for multiple test instances.

        Args:
            test_instances: Array of test instances
            test_labels: Optional array of test labels
            parallel: Whether to use parallel processing

        Returns:
            List of influence analysis results
        """
        if parallel is None:
            parallel = self.config.batch_processing

        results = []

        if parallel and self.config.parallel_workers > 1:
            from multiprocessing import Pool

            with Pool(self.config.parallel_workers) as pool:
                args = [(inst, label if test_labels is not None else None, i)
                       for i, inst in enumerate(test_instances)]
                results = pool.starmap(self._analyze_single, args)
        else:
            for i, test_instance in enumerate(test_instances):
                label = test_labels[i] if test_labels is not None else None
                result = self._analyze_single(test_instance, label, i)
                results.append(result)

                if self.config.verbose >= 1 and (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_instances)} instances")

        return results

    def _analyze_single(
        self,
        test_instance: np.ndarray,
        test_label: Optional[int],
        index: int
    ) -> Dict[str, Any]:
        """Helper method for single instance analysis."""
        result = self.analyze_influence(test_instance, test_label)
        result['instance_index'] = index
        return result

    def evaluate(
        self,
        ground_truth_influences: Dict[int, np.ndarray],
        test_instances: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate Proxima against ground truth influence scores.

        Args:
            ground_truth_influences: Ground truth influence rankings
            test_instances: Test instances to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        accuracies = []

        for i, test_instance in enumerate(test_instances):
            result = self.analyze_influence(test_instance)
            predicted_top_k = result['top_k_indices']

            if i in ground_truth_influences:
                ground_truth_top_k = ground_truth_influences[i][:self.config.top_k]
                accuracy = evaluate_influence_accuracy(
                    predicted_top_k,
                    ground_truth_top_k
                )
                accuracies.append(accuracy)

        metrics['mean_accuracy'] = np.mean(accuracies)
        metrics['std_accuracy'] = np.std(accuracies)
        metrics['min_accuracy'] = np.min(accuracies)
        metrics['max_accuracy'] = np.max(accuracies)

        return metrics

    def save(self, path: str) -> None:
        """Save Proxima state to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'config': self.config.to_dict(),
            'training_losses': self.training_losses,
            'metric_learner': self.metric_learner,
            'cache': self.cache
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved Proxima state to {save_path}")

    @classmethod
    def load(cls, path: str, model: Any, dataset: np.ndarray, labels: np.ndarray) -> 'Proxima':
        """Load Proxima state from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        instance = cls(
            model=model,
            dataset=dataset,
            labels=labels,
            config=state['config']
        )

        instance.training_losses = state['training_losses']
        instance.metric_learner = state['metric_learner']
        instance.cache = state.get('cache', {})

        return instance

    def _save_preprocessing_results(self) -> None:
        """Save preprocessing results to disk."""
        if not self.config.output_dir:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save training losses
        np.save(output_dir / "training_losses.npy", self.training_losses)

        # Save metric learner
        with open(output_dir / "metric_learner.pkl", 'wb') as f:
            pickle.dump(self.metric_learner, f)

        logger.info(f"Saved preprocessing results to {output_dir}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the influence analysis."""
        stats = {
            'dataset_size': len(self.dataset),
            'n_features': self.dataset.shape[1],
            'n_classes': len(np.unique(self.labels)),
            'config': {
                'top_k': self.config.top_k,
                'lambda_ratio': self.config.lambda_ratio,
                'metric_type': self.config.metric_type,
                'influence_method': self.config.influence_method
            }
        }

        if self.training_losses is not None:
            stats['training_loss_stats'] = {
                'mean': float(np.mean(self.training_losses)),
                'std': float(np.std(self.training_losses)),
                'min': float(np.min(self.training_losses)),
                'max': float(np.max(self.training_losses))
            }

        if self.proxy_subset_indices is not None:
            stats['proxy_subset_stats'] = {
                'size': len(self.proxy_subset_indices),
                'ratio': len(self.proxy_subset_indices) / len(self.dataset)
            }

        stats['cache_size'] = len(self.cache)

        return stats


__all__ = ['Proxima']