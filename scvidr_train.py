#Create Access to my code
import sys
sys.path.insert(1, '../vidr')

import argparse
import logging
logging.basicConfig(level=logging.INFO)

import scanpy as sc

from vidr import VIDR
from utils import prepare_data

# requirements
# input: h5ad ann datafile (scRNAseq counts)
#.    - with DOSE_COLUMN representing dose
#.    - with CELLTYPE_COLUMN representing cell type


# questions
# do we want model training params to be controllable?

parser = argparse.ArgumentParser(description="Train a scVIDR model on a hd5a dataset")

parser.add_argument('hd5a_data_file', help='The data file containing the raw reads in hd5a format')
parser.add_argument('model_path', help='Path to the directory where the trained model will be saved')
parser.add_argument('--dose_column', help='Name of the column within obs dataframe representing the dose', default='dose')
parser.add_argument('--celltype_column', help='Name of the column within obs dataframe representing the cell type', default='celltype')
parser.add_argument('--test_celltype', help='Name of the cell type to be left out for testing - put in quotation marks for cell types containing spaces', default='Hepatocytes - portal')
parser.add_argument('--treated_dose', help='Treated dose', default='30')
# e.g. --celltypes_keep "Hepatocytes - central;Hepatocytes - portal;Cholangiocytes;Stellate Cells;Portal Fibroblasts;Endothelial Cells"
parser.add_argument('--celltypes_keep', help='Cell types to keep in the dataset during training/testing - put in quotation marks and separate with semicolon', default='ALL')

script_args = parser.parse_args()


DATA_PATH = script_args.hd5a_data_file
CELLTYPE_COLUMN = script_args.celltype_column
DOSE_COLUMN = script_args.dose_column
TEST_CELLTYPE = script_args.test_celltype
TREATED_DOSE = script_args.treated_dose

MODEL_OUTPUT_DIR = script_args.model_path

CELLTYPES_OF_INTEREST = script_args.celltypes_keep

if CELLTYPES_OF_INTEREST != 'ALL':
    CELLTYPES_OF_INTEREST = CELLTYPES_OF_INTEREST.split(';')

logging.info(f'Loading data file: {DATA_PATH}\n\n')
adata = sc.read_h5ad(DATA_PATH)

#Prepare Data Set
logging.info(f'Cell types available in the dataset: ')
available_cell_types = adata.obs[CELLTYPE_COLUMN].unique()
for cell_type in available_cell_types:
    logging.info(f'--- {cell_type}')

if CELLTYPES_OF_INTEREST == 'ALL':
    CELLTYPES_OF_INTEREST = available_cell_types

for cell_type in CELLTYPES_OF_INTEREST:
    if not cell_type in available_cell_types:
        raise ValueError(f'Unknown cell type of interest: {cell_type}')
    

adata = adata[adata.obs[CELLTYPE_COLUMN].isin(CELLTYPES_OF_INTEREST)]


logging.info(f'\n\nNormalizing and preparing data...')
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000)

adata.obs[DOSE_COLUMN] = [str(d) for d in adata.obs.Dose]
adata = adata[:,adata.var.highly_variable]

logging.info(f'\n\nTraining scVIDR model...')
train_adata, test_adata = prepare_data(
    adata, 
    CELLTYPE_COLUMN, 
    DOSE_COLUMN, 
    TEST_CELLTYPE, 
    TREATED_DOSE, 
    normalized = True
)
train_adata.obs["cell_dose"] = [f"{j}_{str(i)}" for (i,j) in zip(train_adata.obs[DOSE_COLUMN], train_adata.obs[CELLTYPE_COLUMN])]
model = VIDR(train_adata, linear_decoder = False)


model.train(
    max_epochs=100,
    batch_size=128,
    early_stopping=True,
    early_stopping_patience=25
)

model.save(MODEL_OUTPUT_DIR)
