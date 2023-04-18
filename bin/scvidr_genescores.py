#Create Access to my code
import os
import sys
sys.path.insert(1, '../vidr')

import argparse
import logging
logging.basicConfig(level=logging.INFO)

import scanpy as sc

from vidr import VIDR
from utils import normalize_data, prepare_data, prepare_cont_data

#Regression modules
from sklearn.linear_model import ElasticNet, Lars, Ridge, LinearRegression
import numpy as np
import pandas as pd
import torch
from scipy import spatial
from scipy import linalg
# requirements
# input: h5ad ann datafile (scRNAseq counts)
#.    - with DOSE_COLUMN representing dose
#.    - with CELLTYPE_COLUMN representing cell type


# questions
# do we want model training params as CLI arguments

parser = argparse.ArgumentParser(description='Interpret scVIDR predictions using ridge regression. Outputs CSV file of gene scores.')

parser.add_argument('h5ad_data_file', help='The data file containing the raw reads in h5ad format')
parser.add_argument('model_path', help='Path to the directory where the trained model was saved in the model training step')
parser.add_argument('output_path', help='Path to the driectory where the gene scores will be saved as a csv file')
parser.add_argument('--dose_column', help='Name of the column within obs dataframe representing the dose (default "Dose")', default='Dose')
parser.add_argument('--celltype_column', help='Name of the column within obs dataframe representing the cell type (default "celltype")', default='celltype')
parser.add_argument('--test_celltype', help='Name of the cell type to be left out for testing - surround by quotation marks for cell types containing spaces (default "Hepatocytes - portal"', default='Hepatocytes - portal')
parser.add_argument('--control_dose', help='Control dose (default "0")', default='0')
parser.add_argument('--treated_dose', help='Treated dose (default "30")', default='30')
# e.g. --celltypes_keep "Hepatocytes - central;Hepatocytes - portal;Cholangiocytes;Stellate Cells;Portal Fibroblasts;Endothelial Cells"
parser.add_argument('--celltypes_keep', help='Cell types to keep in the dataset during training/testing - either a file containing list of cell types (one cell type per line) or semicolon separated list of cell types (surround in quotation marks) - default all available cell types (default "ALL")', default='ALL')
parser.add_argument('--training_size', help='Number of samples generated from latent distribution', default = 100000)
                   
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
CSV_OUTPUT_DIR = script_args.output_path

CELLTYPES_OF_INTEREST = script_args.celltypes_keep

NUM_SAMPLES = script_args.training_size

logging.info(f'Loading data file: {DATA_PATH}\n\n')
adata = sc.read_h5ad(DATA_PATH)



# Check CLI arguments
                    
                    
# Dose checks
#########################
if DOSE_COLUMN == DOSE_CATEGORICAL_COLUMN:
    raise ValueError(f'Column name {DOSE_COLUMN} is reserved for internal processing, please use a different name.')

adata.obs[DOSE_CATEGORICAL_COLUMN] = adata.obs[DOSE_COLUMN].astype(str)
# changed
available_doses_str = adata.obs[DOSE_CATEGORICAL_COLUMN].unique()
available_doses = adata.obs[DOSE_COLUMN].unique()


#is_single_dose = TRAIN_COMMAND == SINGLE_DOSE_COMMAND
#is_multi_dose = TRAIN_COMMAND == MULTI_DOSE_COMMAND


def check_dose(dose_name, dose, doses):
    if not dose in doses:
        raise ValueError(f'Dose "{dose_name}"={dose} not found in available doses: {doses}')

check_dose('control dose', CONTROL_DOSE, available_doses_str)
check_dose('treated dose', TREATED_DOSE, available_doses_str)


# CHANGED#################
# stim_doses = available_doses.tolist()
# stim_doses.remove(float(CONTROL_DOSE))


#if is_single_dose:
if len(available_doses) < 2:
    raise ValueError(f'Prediction requires more than one dose (Only {len(available_doses)} doses available in the input dataset)')
#
#if is_multi_dose:
#    if len(available_doses) < 3:
#        raise ValueError(f'Multi dose training mode expects more than two doses (Only {len(available_doses)} doses available in the input dataset)')


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
 
logging.info(f'\n\nLoading scVIDR model...')

is_single_dose = True if len(available_doses) == 2 else False
is_multi_dose = True if len(available_doses) >= 3 else False
if is_single_dose:
    train_adata, test_adata = prepare_data(
        adata, 
        CELLTYPE_COLUMN, 
        DOSE_CATEGORICAL_COLUMN, 
        TEST_CELLTYPE, 
        TREATED_DOSE, 
        normalized = True
    )
else:
    train_adata, test_adata = prepare_cont_data(
        adata, 
        CELLTYPE_COLUMN, 
        DOSE_CATEGORICAL_COLUMN,
        DOSE_COLUMN, 
        TEST_CELLTYPE, 
        float(CONTROL_DOSE), 
        normalized=True
    )



logging.info(f'Loading model: {MODEL_OUTPUT_DIR}') 
model = VIDR(train_adata, linear_decoder = False)
model = model.load(MODEL_OUTPUT_DIR, train_adata)

def model_predict(model, control_dose, treated_dose, test_celltype, model_name, dose_column_type, doses=None, multi_dose=False):
    if model_name == 'scGen':
        regression = False
    elif model_name == 'scVIDR':
        regression = True
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    kwargs = {}
    if multi_dose:
        kwargs = {
            'continuous': True,
            'doses': doses
        }

    pred, delta, *other = model.predict(
        ctrl_key=control_dose,
        treat_key=treated_dose,
        cell_type_to_predict=test_celltype,
        regression = regression,
        **kwargs
    )
    if len(other):
        reg = other[0]
    else:
        reg = None
    
    # make single dose match the multi dose
    if multi_dose:
        for dose in pred.keys():
            pred[dose].obs[DOSE_COLUMN] = dose
            pred[dose].obs[DOSE_CATEGORICAL_COLUMN] = f'{dose}'
            pred[dose].obs['Model'] = model_name

    else:
        #pred.obs[DOSE_COLUMN] = 'pred'
        #pred.obs['Treatment'] = model_name
        pred.obs[DOSE_COLUMN] = treated_dose
        pred.obs[DOSE_COLUMN] = pred.obs[DOSE_COLUMN].astype(dose_column_type)
        pred.obs[DOSE_CATEGORICAL_COLUMN] = f'{treated_dose}'
        pred.obs['Model'] = model_name

        pred = {pred.obs[DOSE_COLUMN][0]: pred}

    return pred, delta, reg

dose_column_type = adata.obs[DOSE_COLUMN].dtype


# predict and scVIDR cell data
pred, delta, reg = model_predict(
    model=model,
    control_dose=CONTROL_DOSE, 
    treated_dose=TREATED_DOSE, 
    test_celltype=TEST_CELLTYPE, 
    model_name="scVIDR",
    dose_column_type=dose_column_type,
    doses=available_doses, 
    multi_dose=False
)

#Generating input dataset
latent = model.get_latent_representation()
mins = np.min(latent, axis = 0)
maxes = np.max(latent, axis = 0)
rand_samp = []
for (mn,mx) in zip(mins, maxes):
    rand_samp += [np.random.uniform(mn,mx,  size = (NUM_SAMPLES))]
rand_samp = np.array(rand_samp).T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen_samp = model.module.generative(torch.from_numpy(rand_samp).float())["px"].cpu().detach().numpy()

#Weighting sample based on distance from delta
weights = []
for i in range(rand_samp.shape[0]):
    weights += [1/spatial.distance.cosine(rand_samp[i, :], delta)]

#Fitting Ridge Model
reg = Ridge()
reg.fit(rand_samp, gen_samp, sample_weight=weights)

#Generating Test Set For Evaluating Model
test_samp = []
for (mn,mx) in zip(mins, maxes):
    test_samp += [np.random.uniform(mn,mx,  size = (20000))]
test_samp = np.array(test_samp).T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_gen_samp = model.module.generative(torch.from_numpy(test_samp).float())["px"].cpu().detach().numpy()

#Ridge Regression Score
eval_score = reg.score(test_samp,test_gen_samp)
logging.info(f"---- Ridge Regression Accuracy on Test Set: {eval_score}")

#Generating gene scores
gene_weights = reg.coef_
gene_norms = linalg.norm(reg.coef_, axis = 0)
gene_weights_norm = gene_weights / gene_norms

#Creating Pandas DataFrame
gene_scores = np.dot(gene_weights_norm, delta[:, np.newaxis]).squeeze()
gene_names = model.adata.var_names

gene_df = pd.DataFrame({"Gene":gene_names, "Score":gene_scores})



# TODO: save pred_dict
#Check if output directory exists and create one if it doesn't
if not os.path.isdir(CSV_OUTPUT_DIR):
    os.mkdir(CSV_OUTPUT_DIR)

#Save output
gene_df.to_csv(f"{CSV_OUTPUT_DIR}/GENE_SCORES.csv")
logging.info(f"---- Saved GENE_SCORES: {CSV_OUTPUT_DIR}/GENE_SCORES.csv")
