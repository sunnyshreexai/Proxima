# Proxima: A Proxy Model-Based Approach to Influence Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Proxima is a production-ready Python package for efficient influence analysis of machine learning models. It implements a novel proxy model-based approach that significantly speeds up the computation of training data influence on model predictions.

## Key Innovation

Proxima introduces a **loss-over-distance ratio** metric to identify a subset of training instances that preserves the most influential data points. By building a simpler proxy model on this subset, Proxima achieves:

- **95% average accuracy** in identifying top-10 influential instances
- **3-11x speedup** compared to state-of-the-art methods
- **Compatibility** with any ML model (scikit-learn, PyTorch, TensorFlow)

## Features

- **Two-Phase Framework**: Preprocessing (metric learning + loss computation) and proxy model construction
- **Multiple Influence Methods**: Influence Functions, Leave-One-Out, FastIF, Scaling-Up
- **Flexible Metric Learning**: NCA, Siamese networks, and standard distance metrics
- **Fully Configurable**: 50+ optional parameters with sensible defaults
- **Production Ready**: Comprehensive error handling, logging, and caching
- **Framework Agnostic**: Works with any ML model that has fit/predict methods
- **Performance Optimized**: GPU support, parallel processing, and intelligent caching

## Installation

### From PyPI (recommended)

```bash
pip install proxima-influence
```

### From Source

```bash
git clone https://github.com/sunnyshreexai/Proxima.git
cd Proxima
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/sunnyshreexai/Proxima.git
cd Proxima
pip install -e ".[dev]"
```

## Quick Start

```python
from proxima import Proxima
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize Proxima
proxima = Proxima(
    model=model,
    dataset=X_train,
    labels=y_train
)

# Preprocessing (run once)
proxima.preprocess()

# Analyze influence for a test instance
results = proxima.analyze_influence(
    test_instance=X_test[0],
    test_label=y_test[0]
)

print(f"Top-10 influential instances: {results['top_k_indices']}")
print(f"Influence scores: {results['top_k_scores']}")
print(f"Time taken: {results['time_taken']:.2f}s")
```

## Architecture

Proxima employs a two-level framework:

### Phase 1: Preprocessing
- **Metric Learning**: Learn task-specific distance metric (NCA/Siamese)
- **Loss Computation**: Calculate training loss for each instance

### Phase 2: Proxy Model Construction
- **Subset Identification**: Use loss-over-distance ratio to find proxy subset
- **Model Training**: Train proxy model on relabeled subset
- **Influence Analysis**: Compute influence using proxy model

```
┌─────────────────────────────────────────┐
│         Original ML Model M              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│        Preprocessing Phase              │
│   • Metric Learning (NCA/Siamese)       │
│   • Training Loss Computation           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      Proxy Model Construction           │
│   • Compute Loss/Distance Ratio         │
│   • Identify Proxy Subset D'            │
│   • Relabel Using Model M               │
│   • Train Proxy Model M'                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│       Influence Analysis                │
│   • Use M' and D' for Fast Computation  │
│   • Rank Top-K Influential Instances    │
└─────────────────────────────────────────┘
```

## Configuration

All parameters are optional and configurable:

```python
from proxima import ProximaConfig, ConfigPresets

# Use preset configuration
config = ConfigPresets.fast()  # Quick analysis

# Or customize parameters
config = ProximaConfig(
    # Core parameters
    top_k=10,                        # Number of top influential instances
    lambda_ratio=1.0,                # Loss-over-distance ratio parameter
    threshold_initial=0.5,           # Initial threshold for subset selection

    # Metric learning
    metric_type="nca",               # NCA, siamese, euclidean, cosine
    metric_learning_epochs=50,       # Training epochs for metric learning

    # Proxy model
    proxy_model_type="same",         # same, simpler, or linear
    proxy_training_epochs=100,       # Training epochs for proxy model

    # Influence analysis
    influence_method="influence_function",  # loo, fastif, scaling_up

    # Performance
    use_gpu=True,                    # Use GPU if available
    cache_distances=True,            # Cache distance computations
    parallel_workers=4,              # Parallel processing

    # Output
    verbose=1,                       # Logging verbosity
    save_intermediate=True           # Save intermediate results
)

proxima = Proxima(model, X_train, y_train, config)
```

### Configuration Presets

```python
# Fast configuration for quick analysis
config = ConfigPresets.fast()

# High-accuracy configuration
config = ConfigPresets.accurate()

# Balanced configuration
config = ConfigPresets.balanced()

# Memory-efficient for large datasets
config = ConfigPresets.memory_efficient()

# GPU-optimized configuration
config = ConfigPresets.gpu_optimized()
```

## Advanced Usage

### Batch Analysis

```python
# Analyze multiple test instances
results = proxima.batch_analyze(
    test_instances=X_test[:100],
    test_labels=y_test[:100],
    parallel=True
)
```

### Custom Metric Learning

```python
from proxima.metric_learning import MetricLearner

class CustomMetric(MetricLearner):
    def compute_distance(self, X1, X2):
        # Your custom distance computation
        return custom_distances

proxima.metric_learner = CustomMetric(config)
```

### Evaluation Against Ground Truth

```python
# Evaluate against known influential instances
metrics = proxima.evaluate(
    ground_truth_influences=ground_truth,
    test_instances=X_test
)

print(f"Mean accuracy: {metrics['mean_accuracy']:.2%}")
```

### Visualization

```python
from proxima.utils import visualize_influence_scores

# Visualize influence scores
visualize_influence_scores(
    influence_scores=results['all_influence_scores'],
    top_k=10,
    save_path="influence_plot.png"
)
```

## Benchmark Results

Based on experiments with multiple datasets:

| Dataset | Proxima | FastIF | Scaling-Up |
|---------|---------|--------|------------|
| Adult-Income | **98%** | 91% | 84% |
| Lending Club | **99%** | 93% | 88% |
| COMPAS | **95%** | 90% | 86% |
| German-Credit | 93% | **97%** | 81% |
| MNIST | **93%** | 86% | 78% |
| Fashion-MNIST | **94%** | 92% | 83% |
| CIFAR-10 | 87% | 89% | **91%** |

**Average Time (seconds):**
- Proxima: **2.1s**
- FastIF: 6.6s
- Scaling-Up: 6.8s

## API Reference

### Core Classes

- `Proxima`: Main class for influence analysis
- `ProximaConfig`: Configuration management
- `MetricLearner`: Base class for metric learning
- `ProxyModelBuilder`: Proxy model construction
- `InfluenceAnalyzer`: Influence computation

### Key Methods

```python
# Initialize
proxima = Proxima(model, dataset, labels, config)

# Preprocessing (run once)
proxima.preprocess()

# Analyze single instance
results = proxima.analyze_influence(test_instance, test_label)

# Batch analysis
results = proxima.batch_analyze(test_instances, test_labels)

# Evaluate accuracy
metrics = proxima.evaluate(ground_truth, test_instances)

# Save/Load state
proxima.save("proxima_state.pkl")
proxima = Proxima.load("proxima_state.pkl", model, dataset, labels)
```

## Citation

If you use Proxima in your research, please cite:

```bibtex
@inproceedings{shree2024proxima,
  title={Proxima: A Proxy Model-Based Approach to Influence Analysis},
  author={Shree, Sunny and Lei, Yu and Kacker, Raghu N and Kuhn, D Richard},
  booktitle={Proceedings of the International Conference on Software Testing},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Support

For issues, questions, or suggestions, please open an issue on [GitHub](https://github.com/sunnyshreexai/Proxima/issues).

## Acknowledgments

This work is supported by research grant (70NANB21H092) from the Information Technology Laboratory of the National Institute of Standards and Technology (NIST).