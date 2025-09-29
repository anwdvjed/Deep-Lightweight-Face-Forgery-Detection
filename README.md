# Deep-Lightweight-Face-Forgery-Detection

## Overview
This repository tracks the work required to rewrite the `main` branch of **dinvo3**, a deep-learning based lightweight face forgery detection system. The rewrite focuses on improving model robustness, reproducibility, and maintainability while preserving the light footprint that makes dinvo3 deployable on edge devices.

## Rewrite Goals
- **Modular architecture** – Split data processing, model definition, and evaluation utilities into dedicated Python packages for clarity.
- **Reproducible experiments** – Provide deterministic training scripts and configuration files that can be reused across experiments.
- **Efficient deployment** – Supply ONNX and TensorRT export scripts alongside lightweight inference wrappers for CPU and GPU targets.
- **Comprehensive documentation** – Ensure every component exposes clear API documentation and usage examples.

## Proposed Project Structure
```
dinvo3/
├── configs/              # Hydra/YAML experiment definitions
├── data/                 # Data preparation scripts and metadata
├── docs/                 # API references, tutorials, and changelog
├── scripts/              # Training, evaluation, and export entrypoints
├── src/
│   ├── dinvo3/
│   │   ├── data/         # Dataset loaders and augmentations
│   │   ├── models/       # Backbone networks and classifiers
│   │   ├── modules/      # Losses, metrics, and schedulers
│   │   └── utils/        # Logging, checkpoints, and helpers
│   └── tests/            # Unit and integration tests
└── tools/                # Benchmarking and profiling utilities
```

## Implementation Roadmap
1. **Scaffold modules** with stub implementations and type hints to define public APIs.
2. **Port legacy code** from the current main branch into the new module layout, applying linting and formatting standards.
3. **Introduce configuration-driven training** using Hydra or argparse to simplify experiment management.
4. **Add automated testing** with pytest and continuous integration to guarantee stability.
5. **Prepare deployment artifacts**, including quantized weights and runtime benchmarks.

## Testing Strategy
- Unit tests targeting dataset loaders, preprocessing pipelines, and model components.
- Integration tests running end-to-end training on a lightweight subset of the dataset.
- Regression tests comparing metrics against known baselines to prevent performance drops.

## Contribution Guidelines
- Use feature branches derived from `rewrite/main` until the rewrite stabilizes.
- Follow PEP 8 and adopt `black` + `isort` for formatting.
- Document any new feature in `docs/` and update the changelog.
- Open pull requests with accompanying test evidence and benchmarking results.

## Next Steps
- Initialize the scaffolding described above.
- Import the latest stable dinvo3 checkpoints for benchmarking parity.
- Coordinate review sessions to validate architectural choices before merging to `main`.
