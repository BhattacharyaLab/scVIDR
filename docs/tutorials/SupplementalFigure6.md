```python
#Create Access to my code
import sys
sys.path.insert(1, '../vidr/')

#Import hte vaedr functions we have created
from vidr import VIDR
from PCAEval import PCAEval
from utils import *

#Import important modules
import scanpy as sc
import scgen as scg
import pandas as pd
import numpy as np
import torch
import seaborn as sns
from scipy import stats
from scipy import linalg
from scipy import spatial
from anndata import AnnData
from scipy import sparse
from statannotations.Annotator import Annotator
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

sc.set_figure_params(dpi = 150, frameon = True)
sns.set_style("dark")
sc.settings.figdir = "../figures"
```

    /mnt/home/kanaomar/miniconda3/lib/python3.9/site-packages/numba/np/ufunc/parallel.py:365: NumbaWarning: [1mThe TBB threading layer requires TBB version 2019.5 or later i.e., TBB_INTERFACE_VERSION >= 11005. Found TBB_INTERFACE_VERSION = 6103. The TBB threading layer is disabled.[0m
      warnings.warn(problem)
    OMP: Info #273: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.



```python
adata = sc.read_h5ad("../data/srivatsan2020.h5ad")
```

To see how the dataframes are generated, see the supplemental figure 4 notebook.


```python
test_drugs = np.loadtxt("../data/srivatsan2020_testDrugList.txt", dtype = str, delimiter="\t")
```


```python
df_list = []
doses = [0.0, 10.0, 100.0, 1000.0, 10000.0]
for idx, drug in enumerate(test_drugs):
    for d in doses[1:]:
        df = pd.read_csv(f"../data/Sciplex3_VAEArith_{d}_{drug}.csv")
        df["drug"] = drug
        df_list.append(df)
for idx, drug in enumerate(test_drugs):
    for d in doses[1:]:
        df = pd.read_csv(f"../data/Sciplex3_scVIDR_{d}_{drug}.csv")
        df["drug"] = drug
        df_list.append(df)
```


```python
df_full = pd.concat(df_list)
```


```python
df_full["Dose"] = df_full["Dose"].astype(str)
```


```python
pathway_dict = {drug:pathway for drug, pathway in zip(adata.obs["product_name"], adata.obs["pathway"])}
```


```python

```
