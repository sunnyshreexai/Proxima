"""
Proxima: A Proxy Model-Based Approach to Influence Analysis

A production-ready framework for efficient influence analysis of machine learning models
using proxy models and loss-over-distance ratio sampling.
"""

__version__ = "1.0.0"
__author__ = "Sunny Shree, Yu Lei, Raghu N. Kacker, D. Richard Kuhn"
__email__ = "sunny.shree@mavs.uta.edu"

# Core imports
from .config import ProximaConfig, ConfigPresets
from .core import Proxima
from .metric_learning import MetricLearner, NCALearner, SiameseLearner
from .proxy_model import ProxyModelBuilder
from .influence_analysis import (
    InfluenceAnalyzer,
    LeaveOneOutAnalyzer,
    InfluenceFunctionAnalyzer
)
from .utils import (
    compute_training_loss,
    evaluate_influence_accuracy,
    visualize_influence_scores,
    set_seed,
    get_device,
    Timer
)

__all__ = [
    # Core classes
    "Proxima",
    "ProximaConfig",
    "ConfigPresets",

    # Metric learning
    "MetricLearner",
    "NCALearner",
    "SiameseLearner",

    # Proxy model
    "ProxyModelBuilder",

    # Influence analysis
    "InfluenceAnalyzer",
    "LeaveOneOutAnalyzer",
    "InfluenceFunctionAnalyzer",

    # Utilities
    "compute_training_loss",
    "evaluate_influence_accuracy",
    "visualize_influence_scores",
    "set_seed",
    "get_device",
    "Timer",

    # Version info
    "__version__",
]


def get_version():
    """Return the version string."""
    return __version__


def show_config():
    """Display default configuration."""
    from .config import ProximaConfig

    config = ProximaConfig()
    print("Default Proxima Configuration:")
    print(config)


# Package metadata
__metadata__ = {
    "name": "Proxima",
    "version": __version__,
    "description": "A Proxy Model-Based Approach to Influence Analysis",
    "url": "https://github.com/sunnyshreexai/Proxima",
    "license": "MIT",
    "python_requires": ">=3.8",
}