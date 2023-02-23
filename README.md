# [AAMAS 2023] Don’t Simulate Twice: One-Shot Sensitivity Analyses via Automatic Differentiation

Code to reproduce the results of the paper "Don’t Simulate Twice: One-Shot Sensitivity Analyses via Automatic Differentiation" presented at AAMAS 2023.

# Setup

First, install [PyTorch](pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) according to your system specifications (CUDA / CPU) version.

Second, we need to install the [GradABM-JUNE](https://github.com/arnauqb/GradABM-JUNE) package. To accurately reproduce the results in the AAMAS paper, we will specifically install the AAMAS release version.

```bash
git clone git@github.com:arnauqb/GradABM-JUNE.git
cd GradABM-JUNE
python setup.py install
```

We can now clone this repository

```bash
git clone git@github.com:arnauqb/one_shot_sensitivity.git
```

The calibration script can be found in `scripts/calibration.py`. The plotting of the results can be found in `notebooks/plot_results.ipynb`.
