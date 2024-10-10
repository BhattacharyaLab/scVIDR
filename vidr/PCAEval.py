# Import PCA Model
from sklearn.decomposition import PCA


import numpy
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text
from anndata import AnnData
from matplotlib import pyplot
from scipy import sparse, stats

# Create Access to my code
from utils import *


class PCAEval:
    '''
    Implementation of scGen model for batch removal and perturbation prediction.

    Args:
        adata (AnnData): 
            An AnnData object that has been registered via `scgen.setup_anndata`.
        hidden_dim (int): 
            Number of nodes per hidden layer.
        latent_dim (int): 
            Dimensionality of the latent space.
        n_layers (int): 
            Number of hidden layers used for encoder and decoder NNs.
        dropout_rate (float): 
            Dropout rate for neural networks.
        **model_kwargs: 
            Additional keyword arguments for `scgen.SCGENVAE`.

    Examples:
        >>> vae = scgen.SCGEN(adata)
        >>> vae.train()
        >>> adata.obsm["X_scgen"] = vae.get_latent_representation()

    '''

    def __init__(
        self,
        adata: AnnData,
        latent_dim: int = 100,
    ):
        super(PCAEval, self).__init__()
        self.adata = adata

        self.pca = PCA(n_components=latent_dim, svd_solver="arpack")
        if sparse.issparse(self.adata.X):
            self.adata.X = self.adata.X.A

        self.pca.fit(adata.X)

        self._model_summary_string = (
            "PCA Model with the following params: \n latent_dim: {}"
        ).format(
            latent_dim,
        )

    def predict(
        self,
        cell_type_key=None,
        treatment_key=None,
        ctrl_key=None,
        treat_key=None,
        cell_type_to_predict=None,
        regression=False,
        continuous=False,
        doses=None,
    ) -> AnnData:
        '''
        Predicts the cell type provided by the user in the treated condition.

        Args:
            ctrl_key (str): 
                Key for the `control` part of the `data` found in `condition_key`.
            treat_key (str): 
                Key for the `treated` part of the `data` found in `condition_key`.
            adata_to_predict (AnnData): 
                AnnData object for unperturbed cells to be predicted.
            celltype_to_predict (str): 
                The cell type to be predicted.
            restrict_arithmetic_to (str or dict): 
                Dictionary of cell types to observe for prediction.

        Returns:
            numpy.ndarray: 
                Predicted cells in the primary space.
            float: 
                Difference (delta) between treated and control cells in latent space.
        

        Args:
            cell_type_key (_type_, optional): _description_. Defaults to None.
            treatment_key (_type_, optional): _description_. Defaults to None.
            ctrl_key (_type_, optional): _description_. Defaults to None.
            treat_key (_type_, optional): _description_. Defaults to None.
            cell_type_to_predict (_type_, optional): _description_. Defaults to None.
            regression (bool, optional): _description_. Defaults to False.
            continuous (bool, optional): _description_. Defaults to False.
            doses (_type_, optional): _description_. Defaults to None.

        Returns:
            AnnData: _description_
        '''

        # use keys registered from `setup_anndata()

        ctrl_x = self.adata[self.adata.obs[treatment_key] == ctrl_key]
        treat_x = self.adata[self.adata.obs[treatment_key] == treat_key]
        ctrl_x = random_sample(ctrl_x, cell_type_key)
        treat_x = random_sample(treat_x, cell_type_key)

        # Balancing across treatments
        new_adata = ctrl_x.concatenate(treat_x)
        new_adata = random_sample(
            new_adata, treatment_key, max_or_min="min", replacement=False
        )
        if sparse.issparse(new_adata.X):
            new_adata.X = new_adata.X.A

        # Getting testing data
        ctrl_data = new_adata[
            (new_adata.obs[cell_type_key] == cell_type_to_predict)
            & (new_adata.obs[treatment_key] == ctrl_key)
        ]
        latent_cd = self.pca.transform(ctrl_data.X)

        # Are we regressing the delta on the latent space or not
        if not regression:
            # No regression on latent space
            ctrl_x = new_adata[new_adata.obs[treatment_key] == ctrl_key].copy()
            treat_x = new_adata[new_adata.obs[treatment_key] == treat_key].copy()
            # Compress data to latent space and then calculate the delta
            latent_ctrl = np.average(self.pca.transform(ctrl_x.X), axis=0)
            latent_treat = np.average(self.pca.transform(treat_x.X), axis=0)
            delta = latent_treat - latent_ctrl
        else:
            # Regression on latent space
            latent_X = self.pca.transform(new_adata.X)
            latent_adata = sc.AnnData(X=latent_X, obs=new_adata.obs.copy())
            # Get deltas and control centroids for each cell tpye in the training dataset
            deltas = []
            latent_centroids = []
            cell_types = np.unique(latent_adata.obs[cell_type_key])
            for cell in cell_types:
                if cell != cell_type_to_predict:
                    latent_ctrl = latent_adata[
                        (latent_adata.obs[cell_type_key] == cell)
                        & (latent_adata.obs[treatment_key] == ctrl_key)
                    ]
                    latent_treat = latent_adata[
                        (latent_adata.obs[cell_type_key] == cell)
                        & (latent_adata.obs[treatment_key] == treat_key)
                    ]
                    ctrl_centroid = np.average(latent_ctrl.X, axis=0)
                    treat_centroid = np.average(latent_treat.X, axis=0)
                    delta = np.average(latent_treat.X, axis=0) - np.average(
                        latent_ctrl.X, axis=0
                    )
                    deltas.append(delta)
                    latent_centroids.append(ctrl_centroid)
            lr = LinearRegression()
            reg = lr.fit(latent_centroids, deltas)
            delta = reg.predict([np.average(latent_cd, axis=0)])[0]

        # Continuous or Binary Perturbation
        if not continuous:
            treat_pred = delta + latent_cd
            predicted_cells = self.pca.inverse_transform(treat_pred)
            predicted_adata = sc.AnnData(
                X=predicted_cells,
                obs=ctrl_data.obs.copy(),
                var=ctrl_data.var.copy(),
                obsm=ctrl_data.obsm.copy(),
            )
            if not regression:
                return predicted_adata, delta
            else:
                return predicted_adata, delta, reg
        else:
            treat_pred_dict = {
                d: delta * (np.log1p(d) / np.log1p(max(doses))) + latent_cd
                for d in doses
                if d > min(doses)
            }
            predicted_cells_dict = {
                d: self.pca.inverse_transform(treat_pred_dict[d])
                for d in doses
                if d > min(doses)
            }
            predicted_adata_dict = {
                d: sc.AnnData(
                    X=predicted_cells_dict[d],
                    obs=ctrl_data.obs.copy(),
                    var=ctrl_data.var.copy(),
                    obsm=ctrl_data.obsm.copy(),
                )
                for d in doses
                if d > min(doses)
            }
            if not regression:
                return predicted_adata_dict, delta
            else:
                return predicted_adata_dict, delta, reg

    def reg_mean_plot(
        self,
        adata,
        axis_keys,
        labels,
        condition_key,
        path_to_save="./reg_mean.pdf",
        save=True,
        gene_list=None,
        show=False,
        top_100_genes=None,
        verbose=False,
        legend=True,
        title=None,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        **kwargs,
    ):
        '''
        Plots mean matching figure for a set of specific genes.

    Args:
        adata (AnnData): 
            AnnData object with equivalent structure to the initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model. Must have been set up with `batch_key` and `labels_key`
            corresponding to batch and cell type metadata, respectively.
        axis_keys (dict): 
            Dictionary of `adata.obs` keys that are used by the axes of the plot. Must be in the form 
            `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
        labels (dict): 
            Dictionary of axis labels in the form `{"x": "x-axis-name", "y": "y-axis name"}`.
        path_to_save (str): 
            Path where the plot will be saved.
        save (bool): 
            If `True`, the plot will be saved to the specified path.
        gene_list (list): 
            List of gene names to be plotted.
        show (bool): 
            If `True`, the plot will be displayed after saving.

    Examples:
        >>> import anndata
        >>> import scgen
        >>> import scanpy as sc
        >>> train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
        >>> scgen.setup_anndata(train)
        >>> network = scgen.SCGEN(train)
        >>> network.train()
        >>> unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
        >>> pred, delta = network.predict(
        >>>     adata=train,
        >>>     adata_to_predict=unperturbed_data,
        >>>     ctrl_key="control",
        >>>     treat_key="treatulated"
        >>> )
        >>> pred_adata = anndata.AnnData(
        >>>     pred,
        >>>     obs={"condition": ["pred"] * len(pred)},
        >>>     var={"var_names": train.var_names},
        >>> )
        >>> CD4T = train[train.obs["cell_type"] == "CD4T"]
        >>> all_adata = CD4T.concatenate(pred_adata)
        >>> network.reg_mean_plot(
        >>>     all_adata,
        >>>     axis_keys={"x": "control", "y": "pred", "y1": "treatulated"},
        >>>     gene_list=["ISG15", "CD3D"],
        >>>     path_to_save="tests/reg_mean.pdf",
        >>>     show=False
        >>> )
        '''
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        if sparse.issparse(adata.X):
            adata.X = adata.X.A

        diff_genes = top_100_genes
        treat = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            treat_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = numpy.average(ctrl_diff.X, axis=0)
            y_diff = numpy.average(treat_diff.X, axis=0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("top_100 DEGs mean: ", r_value_diff**2)
        x = numpy.average(ctrl.X, axis=0)
        y = numpy.average(treat.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print("All genes mean: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(numpy.arange(start, stop, step))
            ax.set_yticks(numpy.arange(start, stop, step))
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="black"))
                pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
                force_points=(0.0, 0.0),
            )
        if legend:
            pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=fontsize)
        else:
            pyplot.title(title, fontsize=fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )
        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2

    def get_gene_perturb_coef(self, delta):
        '''Get gene perturbation coefficients for a given delta.
    
        This method calculates gene-specific perturbation coefficients using the decoder's weights and the provided delta.
        If you have performed linear regression, you can input the regression weights for delta. This will provide general
        regression weights instead of gene-specific ones. The method uses the dot product of the linear decoder's weights
        with the delta to compute the coefficients and returns a DataFrame of genes and their perturbation coefficients.

        Args:
            delta (numpy.ndarray): The delta vector, representing the difference between control and treated
                                conditions in latent space.

        Returns:
            pd.DataFrame: A DataFrame containing genes as the index and their perturbation coefficients as a column.

        Raises:
            Exception: If the model is not trained or the decoder is non-linear.

        Example:
            >>> delta = np.random.randn(latent_dim)
            >>> gene_coef_df = model.get_gene_perturb_coef(delta)
            >>> print(gene_coef_df)

    '''
        if self.is_trained_ & self._is_linear_decoder:
            W = model.module.decoder[0].weight.cpu().detach().numpy()
            gene_weights = np.dot(W, delta)
            gene_weight_df = pd.DataFrame(
                    {i: j for (i, j) in zip(train.var_names, gene_weights)},
                    index=["gene_weights"],
                ).T
            return gene_weight_df
        else:
            raise Exception(
                "Either model isn't trained or has a non-linear decoder."
                )

    def get_pseudo_dose(self, delta, cell_types=None):
        ''' Calculate the pseudodose ordering of cells by projecting them orthogonally onto the span of the delta.

        This method computes the pseudodose values of the cells by projecting the cells' expression data onto the 
        latent vector (delta). It can be applied to specific cell types if provided. The pseudodose values represent 
        the orthogonal projection of each cell onto the delta, which can be interpreted as a measure of cell state.

        Args:
            delta (numpy.ndarray): The latent vector representing the perturbation or difference between two conditions 
                                in latent space.
            cell_types (list of str, optional): A list of cell types to include in the pseudodose calculation. 
                                                If `None`, all cell types in the AnnData object will be used.

        Returns:
            numpy.ndarray: An array of pseudodose values for each cell, representing its position along the span of delta.

        Example:
            >>> delta = np.random.randn(latent_dim)
            >>> pseudo_dose = model.get_pseudo_dose(delta, cell_types=["CD4T", "B-cells"])
            >>> print(pseudo_dose)

        '''
        cell_type_key = self.scvi_setup_dict_["categorical_mappings"][
                "_scvi_labels"
            ]["original_key"]
        if cell_types == None:
            adata = self.adata
        else:
            adata = self.adata[self.adata.obs[cell_type_key].isin(cell_types), :]

        pt_values = []
        for i in adata.X:
            pt_value = np.dot(i, delta) / np.dot(delta, delta)
            pt_values.append(pt_value)

        return np.array(pt_values)
