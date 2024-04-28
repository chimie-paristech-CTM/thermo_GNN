[//]: # (![ChemProp Logo]&#40;docs/source/_static/images/chemprop_logo.svg&#41;)
# Universal Datasets

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/chemprop)](https://badge.fury.io/py/chemprop)
[![PyPI version](https://badge.fury.io/py/chemprop.svg)](https://badge.fury.io/py/chemprop)

This is the repository corresponding to Graph-based deep learning models for thermodynamic property prediction: The interplay between featurization, model architecture and data distribution
## Requirements
To use CPUs, Suitable for x86 and ARM platforms. 
To use GPUs, you will need:
 * cuda >= 8.0
 * cuDNN

## Installation
To download the code
```
git clone https://github.com/chimie-paristech-CTM/xxxx
cd universal-dataset
```
To set up the ts-tools conda environment:
```
conda env create -f environment.yml
```
To install the TS-tools package, activate the ts-tools environment and run the following command within the TS-tools directory:
```
conda activate universaldataset
pip install -e .
```
## Updates.
- &#9745;  add multidata train(sample datapoint by tempereture and loss weight).
- &#9745;  add another multidata train(sample datapoint by tempereture before trainning, traing the model by one input csv file and loss weight).
- &#9745;  divide the smiles feature as atom_feature, bond_feature and mol_feature.
- &#9745;  add the aggregation of atom level feature(output of atom_feature and bond_feature by message passing) and molecue feature.
- &#9745;  add a script to caculate the atomization_energy in script folder.
- &#9745;  add sum, mean and norm to atom version readout.
## how to train multidata model
Folder Structure:
```
.
├── README.md
├── dataset/smalldataset
│   ├── data1/
│   │   ├── *data1*.csv
│   ├── data2/
│   │   ├── *data2*.csv
│   ├── data3/
│   │   ├── *data3*.csv
│   └── data4/
│   │    ── *data4*.csv
│   ...    
└── chemprop/
```

```
python train.py --data_path <path>  --dataset_type <type>  --fingerprint <fingerprint>  --save_dir <dir> --epochs <epochs> --output_head <output_head> --message_type <message_type> --multidata <multidata>  --smiles_label_values <smiles_label_values> --input_features_type <input_features_type>
```
where 
1. `<path>` is the path to contain multi dataset directory and every have a dataset CSV file. 
2. `<type>` is one of [classification, regression, multiclass, spectra] depending on the type of the dataset.
3. `<dir>` is the directory where model checkpoints will be saved. 
4. `<fingerprint>` containing [atom, mol, hyper] depending on the type of dataset you want. 
5. `<epochs>` control traing model epoch. 
6. `<output_head>`  containing [FNN, Transformer, MLP] controling the output model.
7. `<message_type>` containing ['message', 'multiheadattention', 'mlpsin','spiking'] controling the train model.
8. `<multidata>` containing [multi, multi_single_input] controling two ways to read input csv file.
9. `<aggregation>` containing [sum, mean, norm] controling the output type of output_head.
10. `<input_features_type>` containing [chemprop, molecule_level_feature] controling to use defualt chemprop input feature or atom and molecula level feature as input.
11. `<smiles_label_values>` "xxdata1xx,0.5" "fxxdata2xx,0.3" "xxdata3xx,0.7" "xxdata4xx,1.2" "xxdata5xx,1.3" controling training loss in different dataset csv file. If do not use this, it will be 1 as defuat.
For example:
Supporting multiple datasets, please place the data in the dataset/smalldataset directory. The file format is displayed above, where * data1 *. csv's data1 can be replaced with any character. Use -- data_path to specify the path to its "dataset/smalldataset", where "--multidata t" is equal to having multi data mode enabled, and "--smiles_label_values" is the declared error factor for each dataset. If -- smiles_label_values is not used, the default error factor is 1.
run:
```
python train.py --data_path ./dataset/singledata/lipo_train.csv  --dataset_type regression  --output_fingerprint atom  --save_dir ./lipo/checkpoint --epochs 2 --output_head FFN --message_type mlptrigonometric  --smiles_label_values "esol_train,0.5" "freesolv_train,0.3" "pdbbind_full_train,0.7" "lipo_train,1.2" "lipo,1.3"  --multidata 'multi' --input_features_type molecule_level_feature --aggregation norm
```

## how to train multidata model with single input csv file
Folder Structure:
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
|smiles |  target|  target_label|
data:
│smiles1| target1| target_label1|  
│smiles2| target2| target_label2|     
│smiles3| target3| target_label3|    
...   
└── 
```
```
python train.py --data_path <path>  --dataset_type <type>  --fingerprint <fingerprint>  --save_dir <dir> --epochs <epochs> --output_head <output_head> --message_type <message_type> --multidata <multidata>  --smiles_label_values <smiles_label_values> --input_features_type <input_features_type> --ignore_columns <ignore_columns>
```
same as above. special attention to follow.
1. `<ignore_column>` represent the target_label name in the csv file.
2. `<path>` is the csv file path not the dir path.
Supporting multiple datasets with a sum csv file.
run:
```
python train.py --data_path ./dataset/singledata/lipo_train.csv  --dataset_type regression  --output_fingerprint atom  --save_dir ./lipo/checkpoint --epochs 2 --output_head FFN --message_type mlptrigonometric  --smiles_label_values "esol_train,0.5" "freesolv_train,0.3" "pdbbind_full_train,0.7" "lipo_train,1.2" "lipo,1.3" --ignore_columns 'dataset_columns' --multidata 'multi_single_input' --input_features_type molecule_level_feature --aggregation norm
```
## To train a model in Regular method
To train a model, run:
```
python train.py --data_path <path> --dataset_type <type> --save_dir <dir> --output_head <output_head> --epochs <epoch> --input_features_type <input_features_type> --aggregation <norm> --message_type <message_type> --output_fingerprint <output_fingerprint>
```
<>
For example:
```
python train.py --data_path ./dataset/singledata/lipo_train.csv  --dataset_type regression  --output_fingerprint atom  --save_dir ./lipo/checkpoint --epochs 2 --output_head FFN --message_type mlptrigonometric --input_features_type molecule_level_feature --aggregation norm
```

A full list of available command-line arguments can be found in [chemprop/args.py](https://github.com/xxxx).

If installed from source, `python train.py` can be replaced with `chemprop_train`.

Notes:
* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with `--metric <metric>`.
* `--save_dir` may be left out if you don't want to save model checkpoints.
* `--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the `save_dir`.

## script
the script 'run_chemprop' will run chemprop to train every model in directry automatily.
## dataset
the dataset.zip contain the qm9 enthalpy and paton atomization enrgy dataset.
## Citation
If (parts of) this work are used as part of a publication, please cite the paper:
```
@article{***,
  title={Graph-based deep learning models for thermodynamic property prediction: The interplay between featurization, model architecture and data distribution},
  author={Bowen Deng ,Thijs Stuyver},
  journal={ChemRxiv},
  year={2024}
}
```
Furthermore, since the work based on chemprop, also consider citing the paper in which this code was originally presented:
```
@article{***,
  title={Chemprop: A Machine Learning Package for Chemical Property Prediction},
  author={Esther Heid ,Kevin P. Greenman ,Yunsie Chung ,Shih-Cheng Li ,David E. Graff ,Florence H. Vermeire ,Haoyang Wu ,William H. Green ,Charles J. McGill},
  journal={ChemRxiv},
  year={2023}
}
```
## Acknowledgement

- PyTorch implementation of chemprop: [https://github.com/xxxxx](https://github.com/xxxx)
