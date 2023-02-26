# scVIDR
Single Cell Variational Inference of the Dose Response (scVIDR) a  variational autoencoder tool used to predict expression of chemcial perturbations across cell types.

[![DOI](https://zenodo.org/badge/549268647.svg)](https://zenodo.org/badge/latestdoi/549268647)

## Preprint
https://www.biorxiv.org/content/10.1101/2022.10.05.510890v1

## Installation
pip3 install -r requirements.txt

## Data
To get the data directory for figure notebooks: 
https://drive.google.com/file/d/11fzDbp0B19Dy47MtD742Jl4Hz2bdSiiq/view?usp=sharing

## Single dose model training

```
usage: scvidr_train.py [-h] [--dose_column DOSE_COLUMN] [--celltype_column CELLTYPE_COLUMN] [--test_celltype TEST_CELLTYPE] [--treated_dose TREATED_DOSE] [--celltypes_keep CELLTYPES_KEEP] hd5a_data_file model_path
```

Train a scVIDR model on a hd5a dataset

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
  --treated_dose TREATED_DOSE
                        Treated dose
  --celltypes_keep CELLTYPES_KEEP
                        Cell types to keep in the dataset during training/testing - either a file containing list of cell typ
```


To train all the single cell types model used in the manuscript execute

```
python scvidr_train.py --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - portal" ../data/nault2021_singleDose.h5ad ../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes - central.pt/
python scvidr_train.py --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - portal" ../data/nault2021_singleDose.h5ad ../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes - portal.pt/
python scvidr_train.py --celltypes_keep ../metadata/liver_celltypes --test_celltype "Cholangiocytes" ../data/nault2021_singleDose.h5ad ../data/VAE_Binary_Prediction_Dioxin_5000g_Cholangiocytes.pt/
python scvidr_train.py --celltypes_keep ../metadata/liver_celltypes --test_celltype "Stellate Cells" ../data/nault2021_singleDose.h5ad ../data/VAE_Binary_Prediction_Dioxin_5000g_Stellate Cells.pt/
python scvidr_train.py --celltypes_keep ../metadata/liver_celltypes --test_celltype "Portal Fibroblasts" ../data/nault2021_singleDose.h5ad ../data/VAE_Binary_Prediction_Dioxin_5000g_Portal Fibroblasts.pt/
python scvidr_train.py --celltypes_keep ../metadata/liver_celltypes --test_celltype "Endothelial Cells" ../data/nault2021_singleDose.h5ad ../data/VAE_Binary_Prediction_Dioxin_5000g_Endothelial Cells.pt/

**Note**: all of these models are available pretrained





