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
		sc.pp.recipe_zheng17(adata,  n_top_genes=5000)
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

def vae_train(train_adata, model, lr, n_epochs = 100, batch_size = 128, device = "cpu", beta = 1):
	"""
	training VAE model
	"""
	model.train()
	#Create Dataloader
	dataloader = AnnDataLoader(train_adata, batch_size = batch_size, shuffle = True)
	#Determine Device and Parallelize
	if device == "cuda":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Training on {device}")

		if torch.cuda.device_count() > 1:
			print(f"Let's use {torch.cuda.device_count()} GPUs!")
			model = nn.DataParallel(model)
	
	#Put model on determined device
	model.to(device)

	#Create optimizer
	optimizer = optim.Adam(model.parameters(), lr = lr)

	#Create kld_weight
	data_size = len(dataloader.dataset)

	#Training loop
	for t in range(n_epochs):
		print(f"Epoch {t+1}\n--------------------------")
		for batch, X_dict in enumerate(dataloader):
			#Get data and put it on device
			X = X_dict["X"]
			X.to(device)
			
			#Calculate loss
			recon, inputs, mean, var = model(X)
			loss_dict = model.module.loss_function(recon,inputs, mean, var, beta = beta)
			recon_loss = loss_dict["recon_loss"].mean().item()
			loss = loss_dict["loss"]

			#Optimization and back prop
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch % 10 == 0:
				loss, current = loss.item(), batch * len(X)
				print(f"loss: {loss:>7f}  recon_loss: {recon_loss:>7f} [{current:>5d}/{data_size:>5d}]")
	print("Done!")
	model.to("cpu")

def random_sample(
		adata,
		key,
		max_or_min="max",
		replacement = True
		):
	"""
	Makes the populations of each cell group equal based on the property of interest. 
	Will take the populations of each group within the property of interest and take an equal random sample of each group.

	Parameters
	----------

	adata: anndata object to balance populations for
	key: property of interest to balance
	max_or_min: to make the maximum population size or minimum population size as the size we want all populations to be at
	replacement: sampling with or without replacement. If max_or_min == "max", we automatically sample with replacement.

	Return
	----------
	resampled_adata: AnnData
	an anndata object containing equal size populations for each group within the property of interest.
	"""
	#list of groups within property of interest and find their cell population sizes
	pop_dict = Counter(adata.obs[key])
	#Find whether we want to take the maximum or the minimum size of the population
	eq = np.max(list(pop_dict.values())) if max_or_min == "max" else np.min(list(pop_dict.values()))
	#replacement?
	replacement = True if max_or_min == "max" else replacement
	#Find the indexes for sampling
	idxs = []
	for group in pop_dict.keys():
		group_bool = np.array(adata.obs[key] == group)
		group_idx = np.nonzero(group_bool)[0]
		group_idx_resampled = group_idx[np.random.choice(len(group_idx), eq, replace = replacement)]
		idxs.append(group_idx_resampled)
	
	resampled_adata = adata[np.concatenate(idxs)].copy()
	return resampled_adata

#@torch.no_grad()
#def gene2latent(
#    adata,
#    model,
#    batch_size = 256,
#    indicies = None,
#):
#    adata = model._validate_anndata(adata)
#    scdl = model._make_data_loader(
#        adata=adata, indices=indicies, batch_size=batch_size
#    )
#    latent = []
#    batch_index = []
#    library = []
#    for tensors in scdl:
#        inference_inputs = model.module._get_inference_input(tensors)
#        outputs = model.module.inference(**inference_inputs)
#        latent += [outputs["qz_m"].cpu()]
#        batch_index += [inference_inputs['batch_index'].cpu()]
#        library += [outputs['library'].cpu()]
#    latent = torch.cat(latent).numpy()
#    batch_index = torch.cat(batch_index).numpy()
#    library = torch.cat(library).numpy()
#    return [latent, batch_index, library]
	
def predict(
		adata,
		model,
		cell_type_key = "cell_type",
		treatment_key = "condition",
		ctrl_key = None,
		treat_key = None,
		continuous = False,
		regression = False,
		cell_type_to_predict = None,
		low_dose = False,
		):
	"""
	Predicts the perturbation to the cell type of choice. If continuous is true, we calculate a continuous perturbation along 
	concentrations of drug or toxin of interest.

	Parameters 
	----------	

	adata: anndata object for predicting and validating data of interest
	model: trained model for predictin perturbation to new cell types and doses
	cell_type_key: key in anndata object for the cell type to predict
	treatment_key: key in anndata object for toxin or drug treatment to cell type of interest
	ctrl_key: if the perturbation is binary, this will indicate the set of cells not treated
	treat_key: if the perturbation is binary, this will indicate the set of cells treated
	continuous: if the perturbation is continuous, this will indicate the degree of pertubation to each of the cells. The zero or lowest dose will be assumed to be the ctrl dose,
	and the maximum dose will be assumed to be the treatulation dose.
	cell_type_to_predict: Which cell type should we use latent space arithmetic to predict their perturbation experssion

	Returns
	----------
	
	pred_adata_dict: dict
	dict of numpy nd-arrays for each dose of the prediction
	delta: numpy nd array
	a d dimensional vector representing the perturbation in latent space
	"""
	#Check if model is trained
	if not model.training:
		raise Exception(
				"It looks like your model hasn't been trained, please train it with the either the vae_train or ae_train command"
				)
	
	#Determine if we want to predict continuous or binary perturbation
	if continuous:
		doses = np.unique(adata.obs[treatment_key]).astype(float)
		if low_dose:
			#Make the lowest dose the highest dose
			doses = -1 * doses
		ctrl_key =  np.abs(np.min(doses))
		treat_key = np.abs(np.max(doses))
		doses = np.abs(doses)

	#Preparing control and treatment groups
	ctrl_x = adata[adata.obs[treatment_key] == ctrl_key]
	treat_x = adata[adata.obs[treatment_key] == treat_key]
	ctrl_x = random_sample(ctrl_x, cell_type_key)
	treat_x = random_sample(treat_x, cell_type_key)
	
	
	#Balancing across treatments 
	new_adata = ctrl_x.concatenate(treat_x)
	new_adata = random_sample(new_adata, treatment_key, max_or_min = "min", replacement = False)

	if sparse.issparse(new_adata.X):
		new_adata.X = new_adata.X.A
	
	#Getting testing data
	ctrl_data = new_adata[(new_adata.obs[cell_type_key] == cell_type_to_predict) & (new_adata.obs[treatment_key] == ctrl_key)] 
	
	latent_cd = model.gene2latent(ctrl_data.X).numpy()
	
	#Are we regressing the delta on the latent space or not
	if not regression:
		#No regression on latent space
		ctrl_x = new_adata[new_adata.obs[treatment_key] == ctrl_key].copy()
		treat_x = new_adata[new_adata.obs[treatment_key] == treat_key].copy()
		#Compress data to latent space (maybe I should call method compress?) and then calculate the delta
		latent_ctrl = np.average(model.gene2latent(ctrl_x.X).numpy(), axis = 0)
		latent_treat = np.average(model.gene2latent(treat_x.X).numpy(), axis = 0)
		delta = latent_treat - latent_ctrl
	else:
		#Regression on latent space
		latent_X =  model.gene2latent(new_adata.X).numpy()
		latent_adata = sc.AnnData(X=latent_X, obs = new_adata.obs.copy())
		#Get deltas and control centroids for each cell tpye in the training dataset
		deltas = []
		latent_centroids = []
		cell_types = np.unique(latent_adata.obs[cell_type_key])
		for cell in cell_types:
			if cell != cell_type_to_predict:
				latent_ctrl = latent_adata[(latent_adata.obs[cell_type_key] == cell) & (latent_adata.obs[treatment_key] == ctrl_key)]
				latent_treat = latent_adata[(latent_adata.obs[cell_type_key] == cell) & (latent_adata.obs[treatment_key] == treat_key)]
				ctrl_centroid = np.average(latent_ctrl.X, axis = 0)
				treat_centroid = np.average(latent_treat.X, axis = 0)
				delta = np.average(latent_treat.X, axis = 0) - np.average(latent_ctrl.X, axis = 0)
				deltas.append(delta)
				latent_centroids.append(ctrl_centroid)
		lr = LinearRegression()
		reg = lr.fit(latent_centroids, deltas)
		delta = reg.predict([np.average(latent_cd, axis = 0)])[0]


	if not continuous:
		treat_pred = delta + latent_cd
		predicted_cells = model.latent2gene(treat_pred).numpy() 
		predicted_adata = sc.AnnData(X=predicted_cells , obs=ctrl_data.obs.copy(), var=ctrl_data.var.copy(),obsm=ctrl_data.obsm.copy(),)
		return predicted_adata, delta
	else:
		treat_pred_dict = {d:delta*(np.log1p(d)/np.log1p(max(doses))) + latent_cd  for d in doses if d > min(doses)}
		predicted_cells_dict = {d:model.module.generative(torch.Tensor(treat_pred_dict[d]))["px"].cpu().detach().numpy() for d in doses if d > min(doses)}
		predicted_adata_dict = {d:sc.AnnData(X=predicted_cells_dict[d], obs=ctrl_data.obs.copy(), var=ctrl_data.var.copy(),obsm=ctrl_data.obsm.copy(),) for d in doses if d > min(doses)}
		return predicted_adata_dict, delta

#Test Code
"""
from models import LDVAE

adata = sc.read("./tests/data/train_kang.h5ad", backup_url='https://drive.google.com/uc?id=1r87vhoLLq6PXAYdmyyd89zG90eJOFYLk')
train_adata, test_adata = prepare_data(adata, "cell_type", "condition", "CD4T", "stimulated", normalized = True)
model = LDVAE(input_dim = train_adata.shape[1], hidden_dim = 512, latent_dim = 100, n_hidden = 2, dropout_rate = 0.2)
vae_train(train_adata, model, 1e-3, device = "cuda", n_epochs = 10)
predict(
		adata = train_adata,
		model = model,
		ctrl_key = "control",
		treat_key = "stimulated",
		cell_type_to_predict = "CD4T",
		)

"""		
