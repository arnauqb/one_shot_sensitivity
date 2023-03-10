# [AAMAS 2023] Don’t Simulate Twice: One-Shot Sensitivity Analyses via Automatic Differentiation

Code to reproduce the results of the paper "Don’t Simulate Twice: One-Shot Sensitivity Analyses via Automatic Differentiation" presented at AAMAS 2023. This code has been adapted from the original [GradABM](https://github.com/AdityaLab/GradABM) repository. It only contains the code related to the calibration of the model. The GradABM-JUNE model can be found [here](https://github.com/arnauqb/GradABM-JUNE).

# Setup

First, install [PyTorch](pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) according to your system specifications (CUDA / CPU) version.

Second, we need to install the [GradABM-JUNE](https://github.com/arnauqb/GradABM-JUNE) package. To accurately reproduce the results in the AAMAS paper, we will specifically install the AAMAS release version.

```bash
git clone git@github.com:arnauqb/GradABM-JUNE.git
cd GradABM-JUNE
git checkout aamas_release
python setup.py install
```

We can now clone this repository

```bash
git clone git@github.com:arnauqb/one_shot_sensitivity.git
```
unzip the data files

```bash
unzip Data.zip
```
and then run the default calibration with 
```
bash run.sh
```

The notebook containing the paper plots can be found in `notebooks/plots.ipynb`.
