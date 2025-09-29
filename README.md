# Proxima: A Proxy Model-Based Approach to Influence Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Proxima is a production-ready Python package for efficient influence analysis of machine learning models. It implements a novel proxy model-based approach that significantly speeds up the computation of training data influence on model predictions.

## Key Innovation

Proxima introduces a **loss-over-distance ratio** metric to identify a subset of training instances that preserves the most influential data points. By building a simpler proxy model on this subset, Proxima achieves:

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


## Support

For issues, questions, or suggestions, please open an issue on [GitHub](https://github.com/sunnyshreexai/Proxima/issues).

