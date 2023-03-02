#Create Access to my code
import os
import sys
sys.path.insert(1, '../vidr')

import argparse
import logging
logging.basicConfig(level=logging.INFO)

import scanpy as sc

from vidr import VIDR
from utils import normalize_data, prepare_data

# requirements
# input: h5ad ann datafile (scRNAseq counts)
#.    - with float DOSE_COLUMN representing dose
#.    - with str CELLTYPE_COLUMN representing cell type


# questions
# do we want model training params as CLI arguments

SINGLE_DOSE_COMMAND = 'single_dose'
MULTI_DOSE_COMMAND = 'multi_dose'


parser = argparse.ArgumentParser(description='Train a VAE model applicable to scGen and scVIDR using a h5ad input dataset')
subparsers = parser.add_subparsers(help='Train a single or multi dose model', dest='train_command')
parser_single = subparsers.add_parser(SINGLE_DOSE_COMMAND, help='Train a single dose model')
parser_multi = subparsers.add_parser(MULTI_DOSE_COMMAND, help='Train a multi dose model')

for subparser in [parser_single, parser_multi]: 
    subparser.add_argument('h5ad_data_file', help='The data file containing the raw reads in h5ad format')
    subparser.add_argument('model_path', help='Path to the directory where the trained model will be saved')
    subparser.add_argument('--dose_column', help='Name of the column within obs dataframe representing the dose (default "Dose")', default='Dose')
    subparser.add_argument('--celltype_column', help='Name of the column within obs dataframe representing the cell type (default "celltype")', default='celltype')
    subparser.add_argument('--test_celltype', help='Name of the cell type to be left out for testing - surround by quotation marks for cell types containing spaces (default "Hepatocytes - portal"', default='Hepatocytes - portal')
    subparser.add_argument('--control_dose', help='Control dose (default "0")', default='0')
    subparser.add_argument('--treated_dose', help='Treated dose (default "30")', default='30')
    # e.g. --celltypes_keep "Hepatocytes - central;Hepatocytes - portal;Cholangiocytes;Stellate Cells;Portal Fibroblasts;Endothelial Cells"
    subparser.add_argument('--celltypes_keep', help='Cell types to keep in the dataset during training/testing - either a file containing list of cell types (one cell type per line) or semicolon separated list of cell types (surround in quotation marks) - default all available cell types (default "ALL")', default='ALL')


                   
script_args = parser.parse_args()

# loading CLI arguments
DATA_PATH = script_args.h5ad_data_file
CELLTYPE_COLUMN = script_args.celltype_column
DOSE_COLUMN = script_args.dose_column
DOSE_CATEGORICAL_COLUMN = 'dose'
TEST_CELLTYPE = script_args.test_celltype
CONTROL_DOSE = script_args.control_dose
TREATED_DOSE = script_args.treated_dose

MODEL_OUTPUT_DIR = script_args.model_path

CELLTYPES_OF_INTEREST = script_args.celltypes_keep

TRAIN_COMMAND = script_args.train_command




logging.info(f'Loading data file: {DATA_PATH}\n\n')
adata = sc.read_h5ad(DATA_PATH)



# Check CLI arguments
                    
                    
# Dose checks
#########################
if DOSE_COLUMN == DOSE_CATEGORICAL_COLUMN:
    raise ValueError(f'Column name {DOSE_COLUMN} is reserved for internal processing, please use a different name.')

adata.obs[DOSE_CATEGORICAL_COLUMN] = adata.obs[DOSE_COLUMN].astype(str)
                    
available_doses = adata.obs[DOSE_CATEGORICAL_COLUMN].unique()

def check_dose(dose_name, dose, doses):
    if not dose in doses:
        raise ValueError(f'Dose "{dose_name}"={dose} not found in available doses: {doses}')

check_dose('control dose', CONTROL_DOSE, available_doses)
check_dose('treated dose', TREATED_DOSE, available_doses)

if TRAIN_COMMAND == MULTI_DOSE_COMMAND:
    if len(available_doses) < 3:
        raise ValueError(f'Multi dose training mode expects more than two doses (Only {len(available_doses)} doses available in the input dataset)')


logging.info(f'Doses available in the dataset: ')
for dose in available_doses:
    logging.info(f'--- {dose}')
                    

# Cell type checks
#########################

if CELLTYPES_OF_INTEREST != 'ALL':
    if os.path.exists(CELLTYPES_OF_INTEREST):
        with open(CELLTYPES_OF_INTEREST) as f:
            CELLTYPES_OF_INTEREST = f.read().strip().split('\n')
    else:
        CELLTYPES_OF_INTEREST = CELLTYPES_OF_INTEREST.split(';')
                    
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
    

# normalize and preprocess data
##########################
logging.info(f'\n\nNormalizing and preparing data...')
adata = adata[adata.obs[CELLTYPE_COLUMN].isin(CELLTYPES_OF_INTEREST)]
adata = normalize_data(adata)
logging.info('---- DONE')
 
logging.info(f'\n\nTraining scVIDR model...')
if TRAIN_COMMAND == SINGLE_DOSE_COMMAND:
    train_adata, test_adata = prepare_data(
        adata, 
        CELLTYPE_COLUMN, 
        DOSE_CATEGORICAL_COLUMN, 
        TEST_CELLTYPE, 
        TREATED_DOSE, 
        normalized = True
    )
                    
if TRAIN_COMMAND == MULTI_DOSE_COMMAND:
    train_adata, test_adata = prepare_cont_data(
        adata, 
        CELLTYPE_COLUMN, 
        DOSE_CATEGORICAL_COLUMN,
        DOSE_COLUMN, 
        TEST_CELLTYPE, 
        CONTROL_DOSE, 
        normalized=True
    )


                    
model = VIDR(train_adata, linear_decoder = False)


model.train(
    max_epochs=100,
    batch_size=128,
    early_stopping=True,
    early_stopping_patience=25
)
logging.info('---- DONE')

logging.info('Saving model to: {MODEL_OUTPUT_DIR}') 
model.save(MODEL_OUTPUT_DIR)
logging.info('---- DONE')
