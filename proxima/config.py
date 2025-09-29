"""Configuration module for Proxima with all parameters optional and configurable."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Union
import yaml
import json
from pathlib import Path


@dataclass
class ProximaConfig:
    """
    Comprehensive configuration for Proxima influence analysis.
    All parameters have sensible defaults and are fully configurable.
    """

    # ============ Core Parameters ============
    top_k: int = 10
    """Number of top influential instances to identify (default: 10)"""

    lambda_ratio: float = 1.0
    """Lambda parameter for loss-over-distance ratio (default: 1.0)"""

    threshold_initial: float = 0.5
    """Initial threshold for ratio-based subset selection (default: 0.5)"""

    threshold_increment: float = 0.25
    """Increment for threshold adaptation (default: 0.25)"""

    min_subset_size_factor: int = 10
    """Minimum subset size as factor of top_k (default: 10)"""

    convergence_ratio: float = 0.8
    """Ratio of instances that must match for convergence (default: 0.8)"""

    # ============ Metric Learning Parameters ============
    metric_type: str = "nca"
    """Type of metric learning: 'nca', 'siamese', 'euclidean', 'cosine' (default: 'nca')"""

    metric_embedding_dim: Optional[int] = None
    """Dimension for metric embedding (None for auto) (default: None)"""

    metric_learning_epochs: int = 50
    """Number of epochs for metric learning (default: 50)"""

    metric_learning_rate: float = 0.001
    """Learning rate for metric learning (default: 0.001)"""

    metric_batch_size: int = 32
    """Batch size for metric learning (default: 32)"""

    # ============ Proxy Model Parameters ============
    proxy_model_type: str = "same"
    """Type of proxy model: 'same', 'simpler', 'linear' (default: 'same')"""

    proxy_training_epochs: int = 100
    """Number of epochs for proxy model training (default: 100)"""

    proxy_early_stopping: bool = True
    """Enable early stopping for proxy training (default: True)"""

    proxy_patience: int = 10
    """Patience for early stopping (default: 10)"""

    proxy_learning_rate: float = 0.001
    """Learning rate for proxy model (default: 0.001)"""

    proxy_regularization: float = 0.0001
    """L2 regularization strength for proxy model (default: 0.0001)"""

    # ============ Influence Analysis Parameters ============
    influence_method: str = "influence_function"
    """Method for influence analysis: 'influence_function', 'loo', 'fastif', 'scaling_up' (default: 'influence_function')"""

    damping: float = 0.001
    """Damping factor for influence function (default: 0.001)"""

    recursion_depth: int = 5
    """Recursion depth for influence function approximation (default: 5)"""

    r_averaging: int = 1
    """Number of samples for stochastic estimation (default: 1)"""

    # ============ Performance Optimization ============
    use_gpu: bool = True
    """Use GPU if available (default: True)"""

    cache_distances: bool = True
    """Cache computed distances (default: True)"""

    cache_influences: bool = True
    """Cache influence computations (default: True)"""

    parallel_workers: int = 4
    """Number of parallel workers (default: 4)"""

    batch_processing: bool = True
    """Process instances in batches (default: True)"""

    batch_size: int = 128
    """Batch size for processing (default: 128)"""

    # ============ Memory Management ============
    memory_efficient: bool = False
    """Use memory-efficient mode (default: False)"""

    max_cache_size_mb: int = 1000
    """Maximum cache size in MB (default: 1000)"""

    sparse_computation: bool = False
    """Use sparse matrix computations where possible (default: False)"""

    # ============ Preprocessing Parameters ============
    normalize_features: bool = True
    """Normalize features before processing (default: True)"""

    normalization_method: str = "standard"
    """Normalization method: 'standard', 'minmax', 'robust' (default: 'standard')"""

    handle_missing: bool = True
    """Handle missing values automatically (default: True)"""

    missing_strategy: str = "mean"
    """Strategy for missing values: 'mean', 'median', 'drop' (default: 'mean')"""

    # ============ Logging and Output ============
    verbose: int = 1
    """Verbosity level: 0=silent, 1=progress, 2=detailed (default: 1)"""

    log_file: Optional[str] = None
    """Path to log file (default: None)"""

    save_intermediate: bool = False
    """Save intermediate results (default: False)"""

    output_dir: Optional[str] = None
    """Directory for outputs (default: None)"""

    results_format: str = "json"
    """Format for results: 'json', 'pickle', 'csv' (default: 'json')"""

    # ============ Validation Parameters ============
    validate_inputs: bool = True
    """Validate input data (default: True)"""

    check_convergence: bool = True
    """Check for convergence during iterations (default: True)"""

    max_iterations: int = 20
    """Maximum iterations for threshold search (default: 20)"""

    # ============ Visualization Parameters ============
    visualize_results: bool = False
    """Generate visualizations (default: False)"""

    plot_format: str = "png"
    """Format for plots: 'png', 'pdf', 'svg' (default: 'png')"""

    plot_dpi: int = 100
    """DPI for plots (default: 100)"""

    # ============ Advanced Parameters ============
    adaptive_threshold: bool = True
    """Use adaptive threshold selection (default: True)"""

    weighted_sampling: bool = False
    """Use weighted sampling based on loss (default: False)"""

    ensemble_proxies: bool = False
    """Use ensemble of proxy models (default: False)"""

    ensemble_size: int = 3
    """Number of models in ensemble (default: 3)"""

    cross_validation: bool = False
    """Use cross-validation for proxy model (default: False)"""

    cv_folds: int = 5
    """Number of cross-validation folds (default: 5)"""

    # ============ Dataset-Specific Parameters ============
    tabular_specific: Dict[str, Any] = field(default_factory=dict)
    """Dataset-specific parameters for tabular data"""

    image_specific: Dict[str, Any] = field(default_factory=dict)
    """Dataset-specific parameters for image data"""

    text_specific: Dict[str, Any] = field(default_factory=dict)
    """Dataset-specific parameters for text data"""

    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []

        # Validate ranges
        if self.top_k < 1:
            errors.append(f"top_k must be positive, got {self.top_k}")

        if self.lambda_ratio <= 0:
            errors.append(f"lambda_ratio must be positive, got {self.lambda_ratio}")

        if not 0 < self.threshold_initial < 10:
            errors.append(f"threshold_initial must be in (0, 10), got {self.threshold_initial}")

        if self.convergence_ratio <= 0 or self.convergence_ratio > 1:
            errors.append(f"convergence_ratio must be in (0, 1], got {self.convergence_ratio}")

        if self.damping <= 0:
            errors.append(f"damping must be positive, got {self.damping}")

        # Validate string options
        valid_metrics = ['nca', 'siamese', 'euclidean', 'cosine', 'manhattan']
        if self.metric_type not in valid_metrics:
            errors.append(f"Invalid metric_type: {self.metric_type}")

        valid_influence = ['influence_function', 'loo', 'fastif', 'scaling_up']
        if self.influence_method not in valid_influence:
            errors.append(f"Invalid influence_method: {self.influence_method}")

        valid_proxy = ['same', 'simpler', 'linear']
        if self.proxy_model_type not in valid_proxy:
            errors.append(f"Invalid proxy_model_type: {self.proxy_model_type}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Optional[str] = None) -> str:
        """Export configuration to YAML format."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=True)

        if path:
            Path(path).write_text(yaml_str)

        return yaml_str

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Export configuration to JSON format."""
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)

        if path:
            Path(path).write_text(json_str)

        return json_str

    @classmethod
    def from_yaml(cls, path: str) -> "ProximaConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "ProximaConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProximaConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def update(self, **kwargs) -> "ProximaConfig":
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Unknown configuration parameter: {key}")

        self.validate()
        return self

    def __str__(self) -> str:
        """String representation of configuration."""
        params = []
        for key, value in self.to_dict().items():
            if not key.endswith('_specific'):
                params.append(f"  {key}: {value}")
        return "ProximaConfig(\n" + "\n".join(params) + "\n)"

    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return f"ProximaConfig({self.to_dict()})"


class ConfigPresets:
    """Preset configurations for common scenarios."""

    @staticmethod
    def fast() -> ProximaConfig:
        """Fast configuration for quick analysis."""
        return ProximaConfig(
            top_k=5,
            threshold_initial=0.3,
            threshold_increment=0.5,
            min_subset_size_factor=5,
            metric_learning_epochs=20,
            proxy_training_epochs=50,
            cache_distances=True,
            cache_influences=True,
            memory_efficient=True
        )

    @staticmethod
    def accurate() -> ProximaConfig:
        """High-accuracy configuration for thorough analysis."""
        return ProximaConfig(
            top_k=20,
            threshold_initial=0.1,
            threshold_increment=0.1,
            convergence_ratio=0.9,
            metric_learning_epochs=100,
            proxy_training_epochs=200,
            influence_method="loo",
            recursion_depth=10,
            ensemble_proxies=True,
            ensemble_size=5,
            cross_validation=True
        )

    @staticmethod
    def balanced() -> ProximaConfig:
        """Balanced configuration for general use."""
        return ProximaConfig(
            top_k=10,
            threshold_initial=0.5,
            threshold_increment=0.25,
            metric_learning_epochs=50,
            proxy_training_epochs=100
        )

    @staticmethod
    def memory_efficient() -> ProximaConfig:
        """Memory-efficient configuration for large datasets."""
        return ProximaConfig(
            memory_efficient=True,
            batch_size=32,
            cache_distances=False,
            cache_influences=False,
            sparse_computation=True,
            max_cache_size_mb=500,
            parallel_workers=2
        )

    @staticmethod
    def gpu_optimized() -> ProximaConfig:
        """GPU-optimized configuration."""
        return ProximaConfig(
            use_gpu=True,
            batch_size=256,
            parallel_workers=8,
            cache_distances=True,
            cache_influences=True
        )

    @staticmethod
    def tabular_data() -> ProximaConfig:
        """Configuration optimized for tabular datasets."""
        return ProximaConfig(
            metric_type="nca",
            proxy_model_type="same",
            normalize_features=True,
            handle_missing=True,
            tabular_specific={
                "categorical_encoding": "onehot",
                "feature_selection": True
            }
        )

    @staticmethod
    def image_data() -> ProximaConfig:
        """Configuration optimized for image datasets."""
        return ProximaConfig(
            metric_type="siamese",
            proxy_model_type="simpler",
            batch_size=128,
            image_specific={
                "augmentation": False,
                "pretrained_features": True,
                "feature_extractor": "resnet"
            }
        )


def load_config(source: Any) -> ProximaConfig:
    """
    Load configuration from various sources.

    Args:
        source: Can be:
            - ProximaConfig instance
            - Dictionary of parameters
            - Path to YAML/JSON file
            - String preset name

    Returns:
        ProximaConfig instance
    """
    if isinstance(source, ProximaConfig):
        return source

    if isinstance(source, dict):
        return ProximaConfig.from_dict(source)

    if isinstance(source, str):
        # Check if it's a preset
        preset_names = ['fast', 'accurate', 'balanced', 'memory_efficient',
                       'gpu_optimized', 'tabular_data', 'image_data']
        if source.lower() in preset_names:
            return getattr(ConfigPresets, source.lower())()

        # Check if it's a file path
        path = Path(source)
        if path.exists():
            if path.suffix in ['.yaml', '.yml']:
                return ProximaConfig.from_yaml(source)
            elif path.suffix == '.json':
                return ProximaConfig.from_json(source)

    raise ValueError(f"Cannot load configuration from: {source}")


__all__ = [
    "ProximaConfig",
    "ConfigPresets",
    "load_config"
]