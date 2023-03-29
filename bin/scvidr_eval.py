#Create Access to my code
import os
import sys
sys.path.insert(1, '../vidr')

import argparse
import logging
logging.basicConfig(level=logging.INFO)

import scanpy as sc

from vidr import VIDR
from utils import normalize_data, prepare_data #, create_cell_dose_column

# requirements
# input: h5ad ann datafile (scRNAseq counts)
#.    - with DOSE_COLUMN representing dose
#.    - with CELLTYPE_COLUMN representing cell type


# questions
# do we want model training params as CLI arguments

parser = argparse.ArgumentParser(description="Evaluate a pretrained scGen and scVIDR models on a h5ad input dataset")

parser.add_argument('h5ad_data_file', help='The data file containing the raw reads in h5ad format')
parser.add_argument('model_path', help='Path to the directory where the trained model will be saved')
parser.add_argument('--dose_column', help='Name of the column within obs dataframe representing the dose', default='Dose')
parser.add_argument('--celltype_column', help='Name of the column within obs dataframe representing the cell type', default='celltype')
parser.add_argument('--test_celltype', help='Name of the cell type to be left out for testing - surround by quotation marks for cell types containing spaces', default='Hepatocytes - portal')
parser.add_argument('--control_dose', help='Control dose', default='0')
parser.add_argument('--treated_dose', help='Treated dose', default='30')
# e.g. --celltypes_keep "Hepatocytes - central;Hepatocytes - portal;Cholangiocytes;Stellate Cells;Portal Fibroblasts;Endothelial Cells"
parser.add_argument('--celltypes_keep', help='Cell types to keep in the dataset during training/testing - either a file containing list of cell types (one cell type per line) or semicolon separated list of cell types (put in quotation marks) - default all available cell types', default='ALL')

script_args = parser.parse_args()

# loading CLI arguments
DATA_PATH = script_args.h5ad_data_file
CELLTYPE_COLUMN = script_args.celltype_column
DOSE_COLUMN = script_args.dose_column
TEST_CELLTYPE = script_args.test_celltype
CONTROL_DOSE = script_args.control_dose
TREATED_DOSE = script_args.treated_dose

MODEL_OUTPUT_DIR = script_args.model_path

CELLTYPES_OF_INTEREST = script_args.celltypes_keep

if CELLTYPES_OF_INTEREST != 'ALL':
    if os.path.exists(CELLTYPES_OF_INTEREST):
        with open(CELLTYPES_OF_INTEREST) as f:
            CELLTYPES_OF_INTEREST = f.read().strip().split('\n')
    else:
        CELLTYPES_OF_INTEREST = CELLTYPES_OF_INTEREST.split(';')


logging.info(f'Loading data file: {DATA_PATH}\n\n')
adata = sc.read_h5ad(DATA_PATH)
adata.obs[DOSE_COLUMN] = adata.obs[DOSE_COLUMN].astype(str)
adata.obs['dose'] = adata.obs[DOSE_COLUMN]


#Prepare Data Set
logging.info(f'Cell types available in the dataset: ')
available_cell_types = adata.obs[CELLTYPE_COLUMN].unique()
for cell_type in available_cell_types:
    logging.info(f'--- {cell_type}')


logging.info(f'Verifying cell types of interest... ')
if CELLTYPES_OF_INTEREST == 'ALL':
    CELLTYPES_OF_INTEREST = available_cell_types

for cell_type in CELLTYPES_OF_INTEREST:
    if not cell_type in available_cell_types:
        raise ValueError(f'Unknown cell type of interest: {cell_type}')
    else:
        logging.info(f'--- {cell_type} EXISTS')
logging.info('---- DONE')
    

logging.info(f'\n\nNormalizing and preparing data...')
adata = adata[adata.obs[CELLTYPE_COLUMN].isin(CELLTYPES_OF_INTEREST)]
adata = normalize_data(adata)
logging.info('---- DONE')
 


train_adata, test_adata = prepare_data(
    adata, 
    CELLTYPE_COLUMN, 
    DOSE_COLUMN, 
    TEST_CELLTYPE, 
    TREATED_DOSE, 
    normalized = True
)

import pdb; pdb.set_trace()

#train_adata.obs["cell_dose"] = create_cell_dose_column(train_adata, CELLTYPE_COLUMN, DOSE_COLUMN)
logging.info('Loading model: {MODEL_OUTPUT_DIR}') 
model = VIDR(train_adata, linear_decoder = False)
model = model.load(MODEL_OUTPUT_DIR, train_adata)


ctrl_adata = adata[((adata.obs['celltype'] == TEST_CELLTYPE) & (adata.obs[DOSE_COLUMN] == CONTROL_DOSE))]
treat_adata = adata[((adata.obs['celltype'] == TEST_CELLTYPE) & (adata.obs[DOSE_COLUMN] == TREATED_DOSE))]
ctrl_adata.obs['Treatment'] = 'Control'
treat_adata.obs['Treatment'] = 'Treated'


def model_predict(model, control_dose, treated_dose, test_celltype, model_name):
    if model_name == 'scGen':
        regression = False
    elif model_name == 'scVIDR':
        regression = True
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    pred, delta, *other = model.predict(
        ctrl_key=control_dose,
        treat_key=treated_dose,
        cell_type_to_predict=test_celltype,
        regression = regression
    )
    pred.obs[DOSE_COLUMN] = 'pred'
    pred.obs['Treatment'] = 'scGen'

    return pred, delta




# evaluate scGen and scVIDR
pred_scgen, delta_scgen = model_predict(model, CONTROL_DOSE, TREATED_DOSE, TEST_CELLTYPE, 'scGen')
pred_scvidr, delta_scvidr = model_predict(model, CONTROL_DOSE, TREATED_DOSE, TEST_CELLTYPE, 'scVIDR')


eval_data = ctrl_adata.concatenate(treat_adata, pred_scgen, pred_scvidr)

# TODO: save eval_data; check if eval_data needs Dose or dose column
