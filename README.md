[//]: # (![ChemProp Logo]&#40;docs/source/_static/images/chemprop_logo.svg&#41;)
# thermo_GNN
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/chemprop.svg)](https://badge.fury.io/py/chemprop)
[![Downloads](https://pepy.tech/badge/your-package-name)](https://github.com/chimie-paristech-CTM/thermo_GNN)

This is the repository containing the code associated with the paper "Graph-based deep learning models for thermodynamic property prediction: The interplay between target definition, data distribution, featurization, and model architecture". Code is provided "as-is". Minor edits may be required to tailor the scripts for different computational systems. 
## Table of Contents

- [Features](#Features)
- [Requirements](#Requirements)
- [Installation](#installation)
- [Updates](#Updates)
- [Quick Start](#quick-start)
  - [Folder Structure](#Folder-Structure)
  - [Train](#Train)
- [Script](#Script)
- [Dataset](#Dataset)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgement)


## Features

- **Atom fingerprint**:
- **Mol-feature**:
- **Ringcount feature**:
- **MLP_Trigonometric**:
- **KAN_Trigonometric**:

More information on the meaning of the individual features, we refer to the associated manuscript.

## Requirements
To use CPUs, Suitable for x86 and ARM platforms. 
To use GPUs, you will need:
 * cuda >= 8.0
 * cuDNN

## Installation
To download the code
```
git clone https://github.com/chimie-paristech-CTM/xxxx
cd thermo_GNN
```
To set up the ts-tools conda environment:
```
conda env create -f environment.yml
```
To install the TS-tools package, activate the ts-tools environment and run the following command within the TS-tools directory:
```
conda activate thermo_GNN
pip install -e .
```
## Updates
- &#9745;  add Atom fingerprint.
- &#9745;  add Mol-feature.
- &#9745;  add Ringcount feature.
- &#9745;  add Mlp_Trigonometric.
- &#9745;  add KAN_Trigonometric.


## Quick Start
### Folder Structure
```
.
├── README.md
├── dataset/smalldataset
│   ├── data/
│   │   ├── *data*.csv
│   ...    
└── chemprop/
```
```
.
├── *data*.csv
column
|smiles | target_label|
data:
│smiles1| target_label1|  
│smiles2| target_label2|     
│smiles3| target_label3|    
...   
└── 
```

### Train
To train a model, run:

```
python train.py --data_path <path> --dataset_type <type> --save_dir <dir>  --epochs <epoch> --input_features_type <input_features_type> --aggregation <norm> --output_fingerprint <output_fingerprint> --model <model>
```
where:
1. `<path>` is the csv file path not the dir path.
2. `<aggregation>` containing [sum, mean, norm] controlling the output type of output_head.
3. `<input_features_type>` containing [chemprop, jpca, molecule_level_feature] controling the type of input feature.
4. `<output_fingerprint>` containing [atom, mol] controlling the type of output fingerprint.
5. `<model>` containing [dpmnn, kantrigonometric, mlptrigonometric] controlling the type of output fingerprint.

For example:
```
python train.py --data_path ./dataset/singledata/lipo_train.csv  --dataset_type regression  --output_fingerprint atom  --save_dir ./lipo/checkpoint --epochs 2  --input_features_type molecule_level_feature --aggregation norm 
```

A full list of available command-line arguments can be found in [chemprop/args.py](https://github.com/xxxx).

If installed from source, `python train.py` can be replaced with `chemprop_train`.

Notes:
* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with `--metric <metric>`.
* `--save_dir` may be left out if you don't want to save model checkpoints.
* `--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the `save_dir`.

## Script
the folder 'dataset_preparation' contains all scripts to process the original datasets.
## Dataset
This link [datasets](https://doi.org/10.6084/m9.figshare.27262947) contains qm9, paton, qmugs, pc9, and qmugs1.1 datasets.
## Citation
If (parts of) this work are used as part of a publication, please cite the paper:
```
@article{***,
  title={Graph-based deep learning models for thermodynamic property prediction: The interplay between target definition, data distribution, featurization, and model architecture},
  author={Bowen Deng ,Thijs Stuyver},
  journal={ChemRxiv},
  year={2024}
}
```
Furthermore, since the work is based on chemprop, please also cite the paper in which this code was originally presented:
```
@article{***,
  title={Chemprop: A Machine Learning Package for Chemical Property Prediction},
  author={Esther Heid ,Kevin P. Greenman ,Yunsie Chung ,Shih-Cheng Li ,David E. Graff ,Florence H. Vermeire ,Haoyang Wu ,William H. Green ,Charles J. McGill},
  journal={ChemRxiv},
  year={2023}
}
```
## Acknowledgement

- PyTorch implementation of chemprop: [https://github.com/chemprop/chemprop](https://github.com/chemprop/chemprop)
