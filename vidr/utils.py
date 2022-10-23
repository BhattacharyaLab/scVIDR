#Import torch modules
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F

#Import single cell modules
import scanpy as sc
from scvi.data import setup_anndata
from scvi.dataloaders import AnnDataLoader

#Other important modules
import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from scipy import sparse

def prepare_data(
		adata,
		cell_type_key,
		treatment_key,
		cell_type_to_predict,
		treatment_to_predict,
		normalized = False
		):
	"""
	This preprocessing step assumes cell ranger data and non-logarithmized data
	"""
	if not normalized:
		sc.pp.noramlize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes = 5000)
	train_adata = adata[~((adata.obs[cell_type_key] == cell_type_to_predict) & 
		(adata.obs[treatment_key] == treatment_to_predict))]
	test_adata = adata[((adata.obs[cell_type_key] == cell_type_to_predict) & 
		(adata.obs[treatment_key] == treatment_to_predict))]
	train_adata = setup_anndata(train_adata, copy = True, batch_key = treatment_key, labels_key=cell_type_key)
	return train_adata, test_adata


def prepare_cont_data(
        adata,
        cell_type_key,
        treatment_key,
        dose_key,
        cell_type_to_predict,
        control_dose,
        normalized = False
        ):
    """
    This preprocessing step assumes cell ranger data and non-logarithmized data
    """
    if not normalized:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes = 5000)
        adata = adata[:, adata.var.highly_variable]
    train_adata = adata[~((adata.obs[cell_type_key] == cell_type_to_predict) & 
        (adata.obs[dose_key] > control_dose))]
    test_adata = adata[((adata.obs[cell_type_key] == cell_type_to_predict) & 
        (adata.obs[dose_key] > control_dose))]
    train_adata = setup_anndata(train_adata, copy = True, batch_key = treatment_key, labels_key=cell_type_key)
    return train_adata, test_adata

def calculate_r2_singledose(
    adata,
    cell: str,
    model: str,
    condition_key: str,
    axis_keys:dict,
    diff_genes = None,
    random_sample_coef = None,
    n_iter: int = 1,
):
    #Set condition key
    
    #Densify sparse matrix
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    
    #Treatment and Prediction
    treat = adata[adata.obs[condition_key] == axis_keys["y"]] 
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]
    
    r2_values_dict = {"R^2":[], "Gene Set":[]}
    for i in range(n_iter):
        
        if random_sample_coef is not None:
            treat_idx = np.random.choice(np.arange(0, treat.shape[0]), int(random_sample_coef*treat.shape[0]))
            pred_idx = np.random.choice(np.arange(0, pred.shape[0]), int(random_sample_coef*pred.shape[0]))
            treat_samp = treat[treat_idx, :]
            pred_samp = pred[pred_idx, :]
        
        else:
            treat_samp = treat
            pred_samp = samp
            
        if diff_genes is not None:
            treat_diff = treat_samp[:, diff_genes]
            pred_diff = pred_samp[:, diff_genes]

            x_diff = np.average(pred_diff.X, axis=0)
            y_diff = np.average(treat_diff.X, axis = 0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            r2_values_dict["R^2"].append(r_value_diff ** 2)
            r2_values_dict["Gene Set"].append("DEGs")
            
    
        x = np.average(treat_samp.X, axis = 0)
        y = np.average(pred_samp.X, axis = 0)
    
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        r2_values_dict["R^2"].append(r_value ** 2)
        r2_values_dict["Gene Set"].append("All HVGs")
        
    r2_values_df = pd.DataFrame(r2_values_dict)
    r2_values_df["Cell"] = cell
    r2_values_df["Model"] = model
        
    return r2_values_df

def calculate_r2_multidose(
    adata,
    cell: str,
    model: str,
    condition_key: str,
    axis_keys:dict,
    diff_genes = None,
    random_sample_coef = None,
    n_iter: int = 1,
):
    #Set condition key
    
    #Densify sparse matrix
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    
    #Treatment and Prediction
    treat = adata[adata.obs[condition_key] == axis_keys["y"]] 
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]
    
    r2_values_dict = {"R^2":[], "Gene Set":[]}
    for i in range(n_iter):
        
        if random_sample_coef is not None:
            treat_idx = np.random.choice(np.arange(0, treat.shape[0]), int(random_sample_coef*treat.shape[0]))
            pred_idx = np.random.choice(np.arange(0, pred.shape[0]), int(random_sample_coef*pred.shape[0]))
            treat_samp = treat[treat_idx, :]
            pred_samp = pred[pred_idx, :]
        
        else:
            treat_samp = treat
            pred_samp = samp
            
        if diff_genes is not None:
            treat_diff = treat_samp[:, diff_genes]
            pred_diff = pred_samp[:, diff_genes]

            x_diff = np.average(pred_diff.X, axis=0)
            y_diff = np.average(treat_diff.X, axis = 0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            r2_values_dict["R^2"].append(r_value_diff ** 2)
            r2_values_dict["Gene Set"].append("DEGs")
            
    
        x = np.average(treat_samp.X, axis = 0)
        y = np.average(pred_samp.X, axis = 0)
    
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        r2_values_dict["R^2"].append(r_value ** 2)
        r2_values_dict["Gene Set"].append("All HVGs")
        
    r2_values_df = pd.DataFrame(r2_values_dict)
    r2_values_df["Cell"] = cell
    r2_values_df["Model"] = model
    r2_values_df["Dose"] = axis_keys["y"]
    return r2_values_df


