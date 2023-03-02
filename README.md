# scVIDR
Single Cell Variational Inference of the Dose Response (scVIDR) a  variational autoencoder tool used to predict expression of chemcial perturbations across cell types.

[![DOI](https://zenodo.org/badge/549268647.svg)](https://zenodo.org/badge/latestdoi/549268647)

## Preprint
https://www.biorxiv.org/content/10.1101/2022.10.05.510890v1

## Installation
```
git clone https://github.com/BhattacharyaLab/scVIDR.git
cd scVIDR
pip3 install -r requirements.txt
pip3 install geomloss==0.2.5
```

## Data
To get the data directory for figure notebooks: 
https://drive.google.com/file/d/11fzDbp0B19Dy47MtD742Jl4Hz2bdSiiq/view?usp=sharing

Once you download `data.zip` copy it to `scVIDR/data` and unzip it there.


## VAE model training (single- and multi- dose models)

```
To train a single dose VAE model run the `scvidr_train.py single_dose` command (Figure 2 of the manuscript).

To train a multi dose VAE model run the `scvidr_train.py multi_dose` command (Figure 3 of the manuscript).


Both types of models expect an AnnData file in the h5ad format as input. In addition to the scRNAseq data, the obs table is also expected to contain a dose and a cell type column.
The single dose model expects at least two distinct doses, whereas the multi dose model expects at least three.

Command arguments are the same for both types of models and are listed below.

 

usage: `scvidr_train.py {single_dose/multi_dose} [-h] [--dose_column DOSE_COLUMN] [--celltype_column CELLTYPE_COLUMN] [--test_celltype TEST_CELLTYPE] [--treated_dose CONTROL_DOSE] [--treated_dose TREATED_DOSE] [--celltypes_keep CELLTYPES_KEEP] h5ad_data_file model_path`
```

Train a VAE model applicable to scGen and scVIDR using a h5ad input dataset

positional arguments:
```
  h5ad_data_file        The data file containing the raw reads in hd5a format

  model_path            Path to the directory where the trained model will be saved

```

model arguments:
```
  -h, --help            show this help message and exit
  --dose_column DOSE_COLUMN
                        Name of the column within obs dataframe representing the dose (default "Dose")
  --celltype_column CELLTYPE_COLUMN
                        Name of the column within obs dataframe representing the cell type (default "celltype")
  --test_celltype TEST_CELLTYPE
                        Name of the cell type to be left out for testing - surround by quotation marks for cell types containing spaces (default "Hepatocytes - portal"
  --control_dose CONTROL_DOSE
                        Control dose (default "0")
  --treated_dose TREATED_DOSE
                        Treated dose (default "30")
  --celltypes_keep CELLTYPES_KEEP
                        Cell types to keep in the dataset during training/testing - either a file containing list of cell types (one cell type per line) or semicolon separated list of cell types (surround in quotation marks) - default all available cell types
                        (default "ALL")
```


To train all the single dose models for all individual cell types used in the manuscript execute

```
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - portal" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes - central.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - central" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes - portal.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Cholangiocytes" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Cholangiocytes.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Stellate Cells" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Stellate Cells.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Portal Fibroblasts" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Portal Fibroblasts.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Endothelial Cells" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Endothelial Cells.pt/"
```


To train all the multi dose models for all individual cell types used in the manuscript execute

```
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - central" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes - central.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - portal" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes - portal.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Cholangiocytes" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Cholangiocytes.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Stellate Cells" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Stellate Cells.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Portal Fibroblasts" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Portal Fibroblasts.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Endothelial Cells" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Endothelial Cells.pt/"
```


**Note**: all of these models are available pretrained (same names as listed above)


## Single dose model evaluation
```
usage: scvidr_eval.py [-h] [--dose_column DOSE_COLUMN] [--celltype_column CELLTYPE_COLUMN] [--test_celltype TEST_CELLTYPE] [--control_dose CONTROL_DOSE] [--treated_dose TREATED_DOSE] [--celltypes_keep CELLTYPES_KEEP] hd5a_data_file model_path
```

Evaluate a pretrained scGen and scVIDR models on a hd5a input dataset

positional arguments:
```
  hd5a_data_file        The data file containing the raw reads in hd5a format

  model_path            Path to the directory where the trained model will be saved
```

optional arguments:
```
  --dose_column DOSE_COLUMN
                        Name of the column within obs dataframe representing the dose
  --celltype_column CELLTYPE_COLUMN
                        Name of the column within obs dataframe representing the cell type
  --test_celltype TEST_CELLTYPE
                        Name of the cell type to be left out for testing - surround by quotation marks for cell types containing spaces
  --control_dose CONTROL_DOSE
                        Control dose
  --treated_dose TREATED_DOSE
                        Treated dose
  --celltypes_keep CELLTYPES_KEEP
                        Cell types to keep in the dataset during training/testing - either a file containing list of cell types (one cell type per line) or semicolon separated list of cell types (put in quotation marks) - default all available cell types
```



