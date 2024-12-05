# Tutorial: Getting Started with scVIDR

**scVIDR** (Single Cell Variational Inference of Dose Response) is a computational framework designed to predict gene expression changes in single cells following chemical perturbations. By leveraging variational autoencoders, scVIDR captures dose-dependent responses across diverse cell types. This innovative approach empowers researchers to anticipate how individual cells react to different chemical exposures, making it a valuable tool in fields like pharmacology and toxicology.

These tutorials will guide you through scVIDR's features, from dataset preparation to advanced dose-response modeling, enabling you to harness its capabilities for your research.


## Installation

To ensure a smooth experience with SCVIDR, we have built a pre-configured Docker container that includes SCVIDR and all its dependencies. This eliminates compatibility issues and provides a stable environment for analysis.

If you haven't set up SCVIDR yet, please follow the steps outlined in the [Installation Guide](../installation/getting-started.md). The guide includes detailed instructions on downloading and running the Docker container, so you can get started quickly.

## Input Data

Several datasets were used in the analysis, but we highlight the key datasets below:

- **TCDD Liver Dose-Response Dataset**:
    - **Source**: Nault et al.
    - **Accession Number**: [GSE184506](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184506) (Gene Expression Omnibus).
    - **Description**: Single-nucleus RNA-seq (snRNA-seq) of flash-frozen mouse livers treated subchronically with TCDD every 4 days for 28 days, Immune cells were excluded due to known migration to the liver during TCDD administration.

- **IFN-γ PBMC Dataset**:
    - **Source**: Kang et al.
    - **Accession Number**: [GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583) (Gene Expression Omnibus).
    - **Description**: Single-cell PBMC dataset examining interferon-gamma (IFN-γ) perturbations.

- **Study B Data**:
    - **Source**: Zheng et al.
    - **Accession Number**: [SRP073767](https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP073767) (Sequence Read Archive).
    - **Description**: Dataset used for cross-experiment predictions of single-cell RNA-seq data.
        
- **LPS6 Species Dataset**:
    - **Source**: Hagai et al.
    - **Accession Number**: [E-MTAB-5919](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-5919/) (BioSciences).
    - **Description**: Dataset used for cross-species perturbation predictions (e.g., using data from pigs, rabbits, and mice).

## Preprocessing

The cell expression vectors were normalized to the median total expression counts for each cell to ensure consistency across cells. This normalization step was carried out using the `scanpy.pp.normalize_total` function.

Following normalization, cell counts were log-transformed with a pseudo-count of 1. This transformation reduces skewness and normalizes the data distribution. The transformation was implemented using the `scanpy.pp.log1p` function.

To focus the analysis on the most informative features, the top 5,000 highly variable genes (HVGs) were selected. This step was performed using the `scanpy.pp.highly_variable` function, ensuring that the analysis prioritizes biologically relevant variations.

Batch effects across different experiments and datasets were accounted for using the `scvi.data.setup_anndata` function. This adjustment ensures that technical variability does not obscure meaningful biological signals.

Finally, differential abundances of cells across groups were handled through a balancing process. Random sampling with replacement was applied for dose groups to equalize representation, while random sampling without replacement was used to balance cell types.




## Index
### Examples of scVIDR Applied to Datasets


### [Prediction of In Vivo Single-Cell Gene Expression of Portal Hepatocytes](Figure2.md)
1. [UMAP Projection of Latent Space](Figure2.md#figure-2a-umap-projection-of-latent-space)
2. [PCA Plots Comparing Prediction Models](Figure2.md#figure-2b-pca-plots-comparing-prediction-models)
3. [Regression Analysis with Predicted Data](Figure2.md#figure-2c-regression-analysis-with-predicted-data)
4. [Comparison of Gene Expression Across Treatments](Figure2.md#figure-2d-comparison-of-gene-expression-across-treatments)
5. [Comprehensive Analysis of Gene Expression Predictions Across Cell Types](Figure2.md#figure-2e-comprehensive-analysis-of-gene-expression-predictions-across-cell-types)


### [Prediction of In Vivo TCDD Dose-Response Across Cell Types](Figure3.md)
1. [Regression and Evaluation Across TCDD Doses](Figure3.md#regression-and-evaluation-across-tcdd-doses)
2. [UMAP Visualization of Dose-Response](Figure3.md#figure-3a-umap-visualization-of-dose-response)
3. [Dose-Response Prediction for Portal Hepatocytes](Figure3.md#figure-3c-dose-response-prediction-for-portal-hepatocytes)
4. [Comparison of Prediction Accuracy Across All Cell Types](Figure3.md#figure-3d-comparison-of-prediction-accuracy-across-all-cell-types)
5. [Regression Analysis of Predicted Gene Expression](Figure3.md#regression-analysis-of-predicted-gene-expression)
6. [Non-Regression Analysis of Predicted Gene Expression Across Doses](Figure3.md#non-regression-analysis-of-predicted-gene-expression-across-doses)
7. [Dose-Response Prediction for Ahrr Gene](Figure3.md#figure-3b-dose-response-prediction-for-ahrr-gene)

### [Interrogation of VAE Using Ridge Regression in Portal Hepatocyte Response Prediction](Figure4.md)
1. [Ridge Regression Analysis on Latent Representations](Figure4.md#ridge-regression-analysis-on-latent-representations)
2. [Visualization of Top Gene Scores](Figure4.md#figure-4b-visualization-of-top-gene-scores)
3. [Gene Enrichment Analysis and Visualization](Figure4.md#figure-4c-gene-enrichment-analysis-and-visualization)
4. [Gene Enrichment Analysis and Visualization](Figure4.md#figure-4d-gene-enrichment-analysis-and-visualization)
5. [Logistic Fit of Median Pathway Scores](Figure4.md#figure-4e-logistic-fit-of-median-pathway-scores)

### [Pseudo-Dose Ordering of Hepatocytes Across TCDD Dose-Response](Figure5.md)
1. [Training VIDR Model for Hepatocyte Dose Response Prediction](Figure5.md#training-vidr-model-for-hepatocyte-dose-response-prediction)
2. [Calculating Pseudodose and Preparing PCA Data](Figure5.md#calculating-pseudodose-and-preparing-pca-data)
3. [Orthogonal Projection of Hepatocytes onto the Pseudo-Dose Axis](Figure5.md#figure-5a-orthogonal-projection-of-hepatocytes-onto-the-pseudo-dose-axis)
4. [Pseudo-Dose Visualization in PCA Space](Figure5.md#figure-5b-pseudo-dose-visualization-in-pca-space)
5. [Pseudo-Dose vs. Fmo3 Expression](Figure5.md#figure-5d-pseudo-dose-vs-fmo3-expression)
6. [Pseudo-Dose vs. Log Dose](Figure5.md#figure-5c-pseudo-dose-vs-log-dose)
7. [PCA of Hepatocyte Zones](Figure5.md#figure-5e-pca-of-hepatocyte-zones)
8. [Pseudo-Dose Distribution Across Hepatocyte Zones](Figure5.md#figure-5f-pseudo-dose-distribution-across-hepatocyte-zones)

## Workflow for Using scVIDR Command-Line Tools

### VAE Model Training (Single- and Multi- Dose Models)

### Overview

To train a single dose VAE model, use the `scvidr_train.py single_dose` command (Figure 2 of the manuscript).

To train a multi dose VAE model, use the `scvidr_train.py multi_dose` command (Figure 3 of the manuscript).

Both types of models require an AnnData file in the `h5ad` format as input. This file should include scRNAseq data along with an observation table (`obs`) containing a dose and a cell type column. The single dose model expects at least two distinct doses, while the multi dose model expects at least three.

### Command Arguments

The arguments for both models are the same and are listed below:

```
usage: scvidr_train.py {single_dose/multi_dose} [-h] [--dose_column DOSE_COLUMN]
                                               [--celltype_column CELLTYPE_COLUMN]
                                               [--test_celltype TEST_CELLTYPE]
                                               [--control_dose CONTROL_DOSE]
                                               [--treated_dose TREATED_DOSE]
                                               [--celltypes_keep CELLTYPES_KEEP] 
                                               h5ad_data_file model_path
```

#### Positional Arguments:

- **`h5ad_data_file`**: Path to the data file containing the raw reads in `h5ad` format.
- **`model_path`**: Path to the directory where the trained model will be saved.

#### Optional Arguments:

- **`-h, --help`**: Show this help message and exit.
- **`--dose_column DOSE_COLUMN`**: Name of the column in the `obs` dataframe representing the dose (default `"Dose"`).
- **`--celltype_column CELLTYPE_COLUMN`**: Name of the column in the `obs` dataframe representing the cell type (default `"celltype"`).
- **`--test_celltype TEST_CELLTYPE`**: Name of the cell type to leave out for testing. Use quotes for names with spaces (default `"Hepatocytes - portal"`).
- **`--control_dose CONTROL_DOSE`**: Value of the control dose (default `"0"`).
- **`--treated_dose TREATED_DOSE`**: Value of the treated dose (default `"30"`).
- **`--celltypes_keep CELLTYPES_KEEP`**: Specify cell types to keep in the dataset for training/testing. Provide either a file with one cell type per line or a semicolon-separated list (default `"ALL"`).

### Example Usage

#### Single Dose Model Training:

To train all single dose models for individual cell types as used in the manuscript:

```bash
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - portal" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes - central.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - central" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes - portal.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Cholangiocytes" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Cholangiocytes.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Stellate Cells" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Stellate Cells.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Portal Fibroblasts" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Portal Fibroblasts.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Endothelial Cells" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Endothelial Cells.pt/"
```

#### Multi Dose Model Training:

To train all multi dose models for individual cell types as used in the manuscript:

```bash
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - central" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes - central.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - portal" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes - portal.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Cholangiocytes" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Cholangiocytes.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Stellate Cells" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Stellate Cells.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Portal Fibroblasts" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Portal Fibroblasts.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Endothelial Cells" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Endothelial Cells.pt/"
```

**Note:** Pretrained models corresponding to these commands are available under the specified paths.


### scVIDR Prediction Workflow

The `scvidr_predict.py` script allows you to use a trained scVIDR model to predict treatment conditions for single-dose or multi-dose datasets. Below are the details for using this script.

#### Single-Dose Model Prediction

#### Usage
```bash
scvidr_predict.py single_dose [-h] [--model MODEL]
                              [--dose_column DOSE_COLUMN]
                              [--celltype_column CELLTYPE_COLUMN]
                              [--test_celltype TEST_CELLTYPE]
                              [--control_dose CONTROL_DOSE]
                              [--treated_dose TREATED_DOSE]
                              [--celltypes_keep CELLTYPES_KEEP]
                              h5ad_data_file model_path output_path
```

#### Description
Predict treatment conditions using a pretrained scVIDR or scGEN model.

#### Positional Arguments
- `h5ad_data_file`: The data file containing the raw reads in h5ad format.
- `model_path`: Path to the directory where the trained model was saved during the model training step.
- `output_path`: Path to the directory where the predicted AnnData will be saved in an h5ad format.

#### Optional Arguments
- `--model MODEL`: Specify whether to use scVIDR or scGEN for prediction (default: `scVIDR`).
- `--dose_column DOSE_COLUMN`: Name of the column in the `obs` dataframe representing the dose (default: `"Dose"`).
- `--celltype_column CELLTYPE_COLUMN`: Name of the column in the `obs` dataframe representing the cell type (default: `"celltype"`).
- `--test_celltype TEST_CELLTYPE`: Name of the cell type to be left out for testing. Use quotes for names with spaces (default: `"Hepatocytes - portal"`).
- `--control_dose CONTROL_DOSE`: Control dose (default: `"0"`).
- `--treated_dose TREATED_DOSE`: Treated dose (default: `"30"`).
- `--celltypes_keep CELLTYPES_KEEP`: Specify the cell types to keep during training/testing. Provide either a file containing the list of cell types (one per line) or a semicolon-separated list (default: `"ALL"`).

#### Example
```bash
python scvidr_predict.py single_dose \
  ../data/nault2021_singleDose.h5ad \
  ../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes-portal.pt/ \
  ../data/SingleDose_TCDD \
  --model scVIDR \
  --dose_column Dose \
  --celltype_column celltype \
  --test_celltype "Hepatocytes - portal" \
  --control_dose 0 \
  --treated_dose 30 \
  --celltypes_keep ../metadata/liver_celltypes
```

---

### Multi-Dose Model Prediction

#### Usage
```bash
scvidr_predict.py multi_dose [-h] [--model MODEL]
                             [--dose_column DOSE_COLUMN]
                             [--celltype_column CELLTYPE_COLUMN]
                             [--test_celltype TEST_CELLTYPE]
                             [--control_dose CONTROL_DOSE]
                             [--treated_dose TREATED_DOSE]
                             [--celltypes_keep CELLTYPES_KEEP]
                             h5ad_data_file model_path output_path
```

#### Description
Predict treatment conditions across multiple doses using a pretrained scVIDR or scGEN model.

#### Positional Arguments
- `h5ad_data_file`: The data file containing the raw reads in h5ad format.
- `model_path`: Path to the directory where the trained model was saved during the model training step.
- `output_path`: Path to the directory where the predicted AnnData will be saved in an h5ad format.

#### Optional Arguments
- `--model MODEL`: Specify whether to use scVIDR or scGEN for prediction (default: `scVIDR`).
- `--dose_column DOSE_COLUMN`: Name of the column in the `obs` dataframe representing the dose (default: `"Dose"`).
- `--celltype_column CELLTYPE_COLUMN`: Name of the column in the `obs` dataframe representing the cell type (default: `"celltype"`).
- `--test_celltype TEST_CELLTYPE`: Name of the cell type to be left out for testing. Use quotes for names with spaces (default: `"Hepatocytes - portal"`).
- `--control_dose CONTROL_DOSE`: Control dose (default: `"0"`).
- `--treated_dose TREATED_DOSE`: Treated dose (default: `"30"`).
- `--celltypes_keep CELLTYPES_KEEP`: Specify the cell types to keep during training/testing. Provide either a file containing the list of cell types (one per line) or a semicolon-separated list (default: `"ALL"`).

#### Example
```bash
python scvidr_predict.py multi_dose \
  ../data/nault2021_multiDose.h5ad \
  ../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes-portal.pt/ \
  ../data/MultiDose_TCDD \
  --model scVIDR \
  --dose_column Dose \
  --celltype_column celltype \
  --test_celltype "Hepatocytes - portal" \
  --control_dose 0.0 \
  --treated_dose 30.0 \
  --celltypes_keep ../metadata/liver_celltypes

```

---

### Notes
1. The prediction scripts assume that the trained models are stored in the `model_path` directory.
2. Ensure the `h5ad_data_file` contains the appropriate `dose` and `celltype` columns in the `obs` table.
3. The output file will be an AnnData object saved in the specified `output_path`.

### Gene Score Calculation Using scVIDR

### Usage

The `scvidr_genescores.py` script interprets scVIDR predictions using ridge regression and outputs gene scores in a CSV format.

#### Command-Line Arguments

#### Positional Arguments
- `h5ad_data_file`: The data file containing the raw reads in h5ad format.
- `model_path`: Path to the directory where the trained model was saved in the model training step.
- `output_path`: Path to the directory where the gene scores will be saved as a CSV file.

#### Optional Arguments
- `--dose_column DOSE_COLUMN`: Name of the column in the `obs` dataframe representing the dose (default: `"Dose"`).
- `--celltype_column CELLTYPE_COLUMN`: Name of the column in the `obs` dataframe representing the cell type (default: `"celltype"`).
- `--test_celltype TEST_CELLTYPE`: Name of the cell type to be left out for testing. Use quotes for names with spaces (default: `"Hepatocytes - portal"`).
- `--control_dose CONTROL_DOSE`: Control dose (default: `"0"`).
- `--treated_dose TREATED_DOSE`: Treated dose (default: `"30"`).
- `--celltypes_keep CELLTYPES_KEEP`: Specify the cell types to keep during training/testing. Provide either a file containing the list of cell types (one per line) or a semicolon-separated list (default: `"ALL"`).
- `--training_size TRAINING_SIZE`: Number of samples generated from the latent distribution (default: `100000`).

#### Example Command

```bash
python scvidr_genescores.py \
  ../data/nault2021_multiDose.h5ad \
  ../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes-portal.pt \
  ../data/MultiDose_TCDD \
  --dose_column Dose \
  --celltype_column celltype \
  --test_celltype "Hepatocytes - portal" \
  --control_dose 0.0 \
  --treated_dose 30.0 \
  --celltypes_keep ../metadata/liver_celltypes

```


### Notes
Ensure that the required data is preprocessed, and a pre-trained model is available before running the script.
