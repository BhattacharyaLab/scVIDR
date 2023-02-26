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

```
positional arguments:
  hd5a_data_file        The data file containing the raw reads in hd5a format

  model_path            Path to the directory where the trained model will be saved

optional arguments:

  --dose_column DOSE_COLUMN
                        Name of the column within obs dataframe representing the dose
  --celltype_column CELLTYPE_COLUMN
                        Name of the column within obs dataframe representing the cell type
  --test_celltype TEST_CELLTYPE
                        Name of the cell type to be left out for testing - put in quotation marks for cell types containing spaces
  --treated_dose TREATED_DOSE
                        Treated dose
  --celltypes_keep CELLTYPES_KEEP
                        Cell types to keep in the dataset during training/testing - put in quotation marks and separate with semicolon
```
