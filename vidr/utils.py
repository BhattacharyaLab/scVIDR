# Import torch modules

# Import single cell modules
import scanpy as sc
from scvi.data import setup_anndata

# Other important modules
import numpy as np
import pandas as pd
from collections import Counter
from scipy import sparse
from scipy import stats


def create_cell_dose_column(adata, celltype_column, dose_column):
    ''' Adds a column to 'adata.obs' named by celltype and dose
    
    This function combines the values from `celltype_column` and `dose_column` 
    in each row of `adata.obs`, separated by an underscore (`_`), and returns 
    the resulting concatenated values as a new column. This can be useful for 
    creating unique identifiers for each combination of cell type and dose.

    Args:
        adata (AnnData): The AnnData object containing single-cell or similar 
            biological data, where metadata is stored in `adata.obs`.
        celltype_column (str): The name of the column in `adata.obs` that 
            contains cell type information (e.g., 'celltype').
        dose_column (str): The name of the column in `adata.obs` that 
            contains dose information (e.g., 'dose').

    Returns:
        pandas.Series: A series containing the concatenated values of 
        `celltype_column` and `dose_column`, which can be added as a new 
        column to `adata.obs`.

    Example:
        Suppose `adata.obs` contains the following columns:
        
        | celltype | dose  |
        |----------|-------|
        | T-cell   | low   |
        | B-cell   | high  |
        | T-cell   | medium|
        
        You can create a new column combining `celltype` and `dose` as follows:
        
        ```python
        adata.obs['cell_dose'] = create_cell_dose_column(adata, 'celltype', 'dose')
        ```
        
        The resulting `adata.obs` will look like:
        
        | celltype | dose  | cell_dose     |
        |----------|-------|---------------|
        | T-cell   | low   | T-cell_low    |
        | B-cell   | high  | B-cell_high   |
        | T-cell   | medium| T-cell_medium |
    '''
    return adata.obs.apply(lambda x: f"{x[celltype_column]}_{x[dose_column]}", axis=1)


def normalize_data(adata):
    ''' Normalizes, logarithmizes, and selects 5000 highly variable genes.
    
    Combines Scanpy preprocessing functions normalize_total, log1p, and
    highly_variable_genes to streamline the preprocessing workflow. 
   
    Args:
       adata (AnnData): The AnnData object containing single-cell or similar
        biological data.
        
    Returns:
        AnnData: A new AnnData object that includes only the top
        5000 highly variable genes.
        
    Example:
        If you have an AnnData object `adata` with raw single-cell RNA-seq data
        you can preprocess it by calling:
        
        ```python
        adata_filtered = normalize_data(adata)
        ```
        
        After running this, `adata_filtered` will contain only the top 
        5000 highly variable genes, normalized and logarithmically transformed.

    '''
    
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)

    return adata[:, adata.var.highly_variable]


def prepare_data(
    adata,
    cell_type_key,
    treatment_key,
    cell_type_to_predict,
    treatment_to_predict,
    normalized=False,
):
    ''' Prepares training and testing data for analysis based on cell type and treatment conditions.

    This function filters an `AnnData` object into train and test datasets. 
    The test set includes the specified cell type to predict and the treatment to predict.
    The training set includes all other data.
    
    cells of a specified type and treatment condition for prediction purposes. It also normalizes 
    the data if it's not already normalized, using the `normalize_data` function.

    Args:
        adata (AnnData): The AnnData object containing single-cell or similar biological data.
        cell_type_key (str): The column name in `adata.obs` that contains cell type information.
        treatment_key (str): The column name in `adata.obs` that contains treatment or condition information.
        cell_type_to_predict (str): The cell type to be separated out for testing/prediction.
        treatment_to_predict (str): The treatment or condition to be separated out for testing/prediction.
        normalized (bool, optional): Whether the data is already normalized. Defaults to False.

    Returns:
        train_adata (AnnData): The training dataset, containing all cells except those specified by 
            `cell_type_to_predict` and `treatment_to_predict`.
        test_adata (AnnData): The test dataset, containing only the cells with the specified 
            `cell_type_to_predict` and `treatment_to_predict`.

    Example:
        Create training and testing datasets from an AnnData object with cell type information in the 
        'celltype' column of adata.obs and dosing information in the 'dose' column of adata.obs.
        To select T-cells under a low-dose treatment as your testing set: 

        ```python
        train_adata, test_adata = prepare_data(
            adata, 
            cell_type_key='celltype', 
            treatment_key='dose', 
            cell_type_to_predict='T-cell', 
            treatment_to_predict='low', 
            normalized=False
        )
        ```

        This will return two `AnnData` objects:  `test_adata` containing only T-cells under low-dose treatment, 
        and train_adata` containing all cells except T-cells under low-dose treatment,

    Note:
        **This preprocessing step assumes the data is in the format output by Cell Ranger and is not 
        logarithmized by default unless specified.**
    '''
    if not normalized:
        adata = normalize_data(adata)
    train_adata = adata[
        ~(
            (adata.obs[cell_type_key] == cell_type_to_predict)
            & (adata.obs[treatment_key] == treatment_to_predict)
        )
    ]
    test_adata = adata[
        (
            (adata.obs[cell_type_key] == cell_type_to_predict)
            & (adata.obs[treatment_key] == treatment_to_predict)
        )
    ]
    train_adata = setup_anndata(
        train_adata, copy=True, batch_key=treatment_key, labels_key=cell_type_key
    )
    
    return train_adata, test_adata
    


def prepare_cont_data(
    adata,
    cell_type_key,
    treatment_key,
    dose_key,
    cell_type_to_predict,
    control_dose,
    normalized=False,
):
    ''' Prepares training and testing data for analysis based on cell type and dose conditions.

    This function filters an `AnnData` object into train and test datasets. 
    The test set includes the specified cell type to predict and doses greater than the control dose.
    The training set includes all other data.

    The function also normalizes the data if it's not already normalized, using Scanpy's normalization functions.

    Args:
        adata (AnnData): The AnnData object containing single-cell or similar biological data.
        cell_type_key (str): The column name in `adata.obs` that contains cell type information.
        treatment_key (str): The column name in `adata.obs` that contains treatment or condition information.
        dose_key (str): The column name in `adata.obs` that contains dose information.
        cell_type_to_predict (str): The cell type to be separated out for testing/prediction.
        control_dose (float): The dose level used as a threshold for separating training and test data.
        normalized (bool, optional): Whether the data is already normalized. Defaults to False.

    Returns:
        train_adata (AnnData): The training dataset, containing all cells except those with the specified
            `cell_type_to_predict` and doses greater than the `control_dose`.
        test_adata (AnnData): The test dataset, containing only the cells with the specified 
            `cell_type_to_predict` and doses greater than the `control_dose`.

    Example:
        Create training and testing datasets from an AnnData object with cell type information in the 
        'cell_type' column of adata.obs and dose information in the 'dose' column of adata.obs.
        To select T-cells treated with doses greater than 100 as your testing set:

        ```python
        train_adata, test_adata = prepare_cont_data(
            adata, 
            cell_type_key='cell_type', 
            treatment_key='treatment', 
            dose_key='dose', 
            cell_type_to_predict='T-cell', 
            control_dose=100, 
            normalized=False
        )
        ```

        This will return two `AnnData` objects: `test_adata` containing only T-cells with doses greater than 100,
        and `train_adata` containing all other cells.


    Note:
        **This preprocessing step assumes the data is in the format output by Cell Ranger and is not 
        logarithmized by default unless specified.**
'''
    if not normalized:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=5000)
        adata = adata[:, adata.var.highly_variable]
    train_adata = adata[
        ~(
            (adata.obs[cell_type_key] == cell_type_to_predict)
            & (adata.obs[dose_key] > control_dose)
        )
    ]
    test_adata = adata[
        (
            (adata.obs[cell_type_key] == cell_type_to_predict)
            & (adata.obs[dose_key] > control_dose)
        )
    ]
    train_adata = setup_anndata(
        train_adata, copy=True, batch_key=treatment_key, labels_key=cell_type_key
    )
    return train_adata, test_adata


def calculate_r2_singledose(
    adata,
    cell: str,
    model: str,
    condition_key: str,
    axis_keys: dict,
    diff_genes=None,
    random_sample_coef=None,
    n_iter: int = 1,
):
    '''_summary_

    Args:
        adata (AnnData): _description_
        cell (str): _description_
        model (str): _description_
        condition_key (str): _description_
        axis_keys (dict): _description_
        diff_genes (_type_, optional): _description_. Defaults to None.
        random_sample_coef (_type_, optional): _description_. Defaults to None.
        n_iter (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    '''

    # Densify sparse matrix
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    # Treatment and Prediction
    treat = adata[adata.obs[condition_key] == axis_keys["y"]]
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]

    r2_values_dict = {"R^2": [], "Gene Set": []}
    for i in range(n_iter):
        if random_sample_coef is not None:
            treat_idx = np.random.choice(
                np.arange(0, treat.shape[0]), int(random_sample_coef * treat.shape[0])
            )
            pred_idx = np.random.choice(
                np.arange(0, pred.shape[0]), int(random_sample_coef * pred.shape[0])
            )
            treat_samp = treat[treat_idx, :]
            pred_samp = pred[pred_idx, :]

        else:
            treat_samp = treat
            pred_samp = samp

        if diff_genes is not None:
            treat_diff = treat_samp[:, diff_genes]
            pred_diff = pred_samp[:, diff_genes]

            x_diff = np.average(pred_diff.X, axis=0)
            y_diff = np.average(treat_diff.X, axis=0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            r2_values_dict["R^2"].append(r_value_diff**2)
            r2_values_dict["Gene Set"].append("DEGs")

        x = np.average(treat_samp.X, axis=0)
        y = np.average(pred_samp.X, axis=0)

        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        r2_values_dict["R^2"].append(r_value**2)
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
    axis_keys: dict,
    diff_genes=None,
    random_sample_coef=None,
    n_iter: int = 1,
):
    '''_summary_

    Args:
        adata (_type_): _description_
        cell (str): _description_
        model (str): _description_
        condition_key (str): _description_
        axis_keys (dict): _description_
        diff_genes (_type_, optional): _description_. Defaults to None.
        random_sample_coef (_type_, optional): _description_. Defaults to None.
        n_iter (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    '''

    # Densify sparse matrix
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    # Treatment and Prediction
    treat = adata[adata.obs[condition_key] == axis_keys["y"]]
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]

    r2_values_dict = {"R^2": [], "Gene Set": []}
    for i in range(n_iter):
        if random_sample_coef is not None:
            treat_idx = np.random.choice(
                np.arange(0, treat.shape[0]), int(random_sample_coef * treat.shape[0])
            )
            pred_idx = np.random.choice(
                np.arange(0, pred.shape[0]), int(random_sample_coef * pred.shape[0])
            )
            treat_samp = treat[treat_idx, :]
            pred_samp = pred[pred_idx, :]

        else:
            treat_samp = treat
            pred_samp = samp

        if diff_genes is not None:
            treat_diff = treat_samp[:, diff_genes]
            pred_diff = pred_samp[:, diff_genes]

            x_diff = np.average(pred_diff.X, axis=0)
            y_diff = np.average(treat_diff.X, axis=0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            r2_values_dict["R^2"].append(r_value_diff**2)
            r2_values_dict["Gene Set"].append("DEGs")

        x = np.average(treat_samp.X, axis=0)
        y = np.average(pred_samp.X, axis=0)

        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        r2_values_dict["R^2"].append(r_value**2)
        r2_values_dict["Gene Set"].append("All HVGs")

    r2_values_df = pd.DataFrame(r2_values_dict)
    r2_values_df["Cell"] = cell
    r2_values_df["Model"] = model
    r2_values_df["Dose"] = axis_keys["y"]
    return r2_values_df


def random_sample(adata, key, max_or_min="max", replacement=True):
    '''Randomly samples and balances the populations of each cell group 
    based on the property of interest.

 

    This function randomly samples cells from groups defined by a specified key so that each group 
    has an equal number of cells. It can sample to match either the largest group size or the smallest 
    group size, with or without replacement.

    Args:
        adata (AnnData): The AnnData object containing the data to balance.
        key (str): The column in `adata.obs` representing the property used to group cells.
        max_or_min (str, optional): Whether to equalize the populations based on the maximum or minimum 
            group size. Defaults to "max".
        replacement (bool, optional): Whether to sample with replacement. If `max_or_min` is "max", 
            sampling will always be done with replacement. Defaults to True.

    Returns:
        AnnData: An AnnData object containing equal-sized populations for each group within the 
        specified property.

    Example:
        Create an AnnData object `adata` with cell type information in the 'cell_type' column 
        and sample the data so that all cell types are balanced to the size of the smallest group:

        ```python
        resampled_adata = random_sample(
            adata, 
            key='cell_type', 
            max_or_min='min', 
            replacement=False
        )
        ```

        In this example, `resampled_adata` will contain equal numbers of cells for each cell type, 
        with the smallest group size used for sampling. Sampling will be done without replacement.
    '''
    # list of groups within property of interest and find their cell population sizes
    pop_dict = Counter(adata.obs[key])
    # Find whether we want to take the maximum or the minimum size of the population
    eq = (
        np.max(list(pop_dict.values()))
        if max_or_min == "max"
        else np.min(list(pop_dict.values()))
    )
    # replacement?
    replacement = True if max_or_min == "max" else replacement
    # Find the indexes for sampling
    idxs = []
    for group in pop_dict.keys():
        group_bool = np.array(adata.obs[key] == group)
        group_idx = np.nonzero(group_bool)[0]
        group_idx_resampled = group_idx[
            np.random.choice(len(group_idx), eq, replace=replacement)
        ]
        idxs.append(group_idx_resampled)

    resampled_adata = adata[np.concatenate(idxs)].copy()
    return resampled_adata
