# %%
import os
import sys
import pickle
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import einops
from tqdm import tqdm
from collections import defaultdict
import einops
import gc

from scipy.stats import ttest_ind
from jaxtyping import Bool, Float
import scanpy as sc
# import anndata as ad
import numpy as np
import torch
from nnsight import NNsight

BASE_PATH = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(BASE_PATH.as_posix())

from scgpt_clean.load_model_from_pretrain import create_clean_model_from_pretrain

# %%
EVAL_DATA_PATH = Path("/maiziezhou_lab3/zihang/interTFM_data/eval_data")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")

# %%
data_set_name = "norman"

model_dir = BASE_PATH / "data/whole-human-pretrain"
scgptmodel, tokenizer = create_clean_model_from_pretrain(model_dir, device=device)
scgpt_nn = NNsight(scgptmodel)

# %%
adata = sc.read_h5ad(BASE_PATH / f"data/perturbation_gears/{data_set_name}/perturb_processed.h5ad")
adata.X = adata.X.toarray()

gene_names = adata.var['gene_name'].to_numpy()
gene_ids = [tokenizer.vocab.get(gene, -1) for gene in gene_names]
# missing_genes = gene_names[np.array(gene_ids) == -1]
# remaining_genes = gene_names[np.array(gene_ids) != -1]
adata = adata[:, np.array(gene_ids) >= 0]
all_zero_genes = adata.X.sum(axis=0) > 0
adata = adata[:, all_zero_genes]
adata_remaining_genes = adata.var['gene_name'].to_numpy()
condition_list_all = adata.obs["condition"].to_numpy()

adata_cond_ctrl = adata[adata.obs["condition"] == "ctrl"]

# print("KLF1+MAP2K6" in pert_data.adata.obs["condition"].tolist())
# print("KLF1+ctrl" in pert_data.adata.obs["condition"].tolist())
# print("MAP2K6+ctrl" in pert_data.adata.obs["condition"].tolist())

# %%
binary_matrix_renamed_nonzero = pd.read_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gene_concepts_nonzero.csv", index_col=0)
concepts = pd.read_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gprofiler_gene_concepts_columns.txt", index_col=0)

concept_id_to_index = {cid: idx for idx, cid in enumerate(binary_matrix_renamed_nonzero.index)}
concept_index_to_id = {idx: cid for cid, idx in concept_id_to_index.items()}
concept_index_to_term = {idx: concepts.loc[cid, 'term_name'] for idx, cid in concept_index_to_id.items()}

# %%
gene_select = "DUSP9"
# scale_list = [1, 2, 3, 4, 5]
# scale_list = list(set([0.1, 0.5, 1.0, 2.0] + [0.3, 0.6, 0.7, 0.8, 0.9])) # + [0.1, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]))
scale_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
condition_list = ["ctrl"]
if f"{gene_select}+ctrl" in condition_list_all:
    condition_list += [f"{gene_select}+ctrl"]
if f"ctrl+{gene_select}" in condition_list_all:
    condition_list += [f"ctrl+{gene_select}"]
assert all([cond in condition_list_all for cond in condition_list]), "Some conditions not found in the data"

concept_idx_union = np.where(binary_matrix_renamed_nonzero[gene_select] != 0)[0].tolist()
gene_select_pos = np.where(adata_remaining_genes == gene_select)[0].item() + 1 # +1 for CLS token

# %% read single gene lp values
org_act_save_dir = EVAL_DATA_PATH / f"{data_set_name}_intervene" / "condition_org"

ctrl_act = torch.load(org_act_save_dir / f"ctrl_activations.pt")

pert_cond_list = condition_list.copy()
pert_cond_list.remove("ctrl")

gene_ctrl_act = [
    torch.load(org_act_save_dir / f"{cond}_activations.pt")
    for cond in pert_cond_list
]
gene_ctrl_act = torch.cat(gene_ctrl_act, dim=0)

# %%
intv_act_save_dir = EVAL_DATA_PATH / f"{data_set_name}_intervene" / f"{gene_select}" / "intervene"
intv_act = {}
for scale in scale_list:
    load_path = intv_act_save_dir / f"scale_{scale}" / f"intv_act_cls.pt"
    temp = torch.load(load_path)
    if type(temp) == list:
        temp = torch.cat(temp, dim=0)
    intv_act[scale] = temp

# %%
ctrl_label = ["ctrl"] * ctrl_act.shape[0]
pert_label = [f"{gene_select}"] * gene_ctrl_act.shape[0]
intv_label_dict = {scale: [f"lp_scale{scale}"] * intv_act[scale].shape[0] for scale in scale_list}

data_all = torch.cat(
    [ctrl_act, gene_ctrl_act] + [intv_act[scale] for scale in scale_list],
    dim=0
)

obs_all = np.concatenate(
    [
        ctrl_label,
        pert_label,
        *[intv_label_dict[scale] for scale in scale_list],
    ],
    axis=0
)

# %%
obs_order = ["ctrl"] + [f"lp_scale{scale}" for scale in scale_list] + [f"{gene_select}"]
cls_data = sc.AnnData(X=data_all.numpy())
cls_data.obs["label"] = pd.Categorical(obs_all, categories=obs_order, ordered=True)

sc.pp.neighbors(cls_data, use_rep="X")
sc.tl.umap(cls_data)

# %%
sc.settings.figdir = ("figures_intv")
custom_palette = {
    "lp_scale0.7": "#1f77b4",
    "lp_scale0.8": "#ff7f0e",
    "lp_scale0.9": "#2ca02c",
    "lp_scale1.0": "#d62728",
    "lp_scale1.1": "#9467bd",
    # "lp_scale1.2": "#8c564b",
    "ctrl": "#b0b0b0",       # light gray
    "DUSP9": "black",        # strong black
}
cls_data_plot = cls_data[cls_data.obs["label"].isin(list(custom_palette.keys()))]
point_standard_size = 120000 / cls_data_plot.shape[0]
point_size = [point_standard_size if lbl not in ["DUSP9", "ctrl"] else point_standard_size * 10 for lbl in cls_data_plot.obs["label"]]
sc.pl.umap(
    cls_data_plot, color="label", title="UMAP of cell embeddings", frameon=True, palette=custom_palette, size=point_size,
    save=f"_lp_intervene_umap_{data_set_name}_{gene_select}_small_scale.pdf"
)

# %%
# df_umap = pd.DataFrame({
#     "UMAP1": cls_data.obsm["X_umap"][:, 0],
#     "UMAP2": cls_data.obsm["X_umap"][:, 1],
#     "label": cls_data.obs["label"]
# })

# plt.figure(figsize=(6,5))
# df_umap_plot = df_umap[df_umap["label"].isin(list(custom_palette.keys()))]
# sns.scatterplot(data=df_umap_plot, x="UMAP1", y="UMAP2", hue="label", palette=custom_palette, alpha=0.7)
# df_gene_select = df_umap[df_umap["label"] == gene_select]
# plt.scatter(df_gene_select["UMAP1"], df_gene_select["UMAP2"], s=50, color="black", label=gene_select)
# plt.title(f"UMAP of lp_all by obs_all")
# plt.savefig(sc.settings.figdir / f"lp_intervene_umap_{data_set_name}_{gene_select}_small_scale_custom.png", dpi=300)
# plt.close()


# %%
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca_fit_data = cls_data[cls_data.obs["label"].isin(["DUSP9", "ctrl"])].X
# pca.fit(pca_fit_data)

# cls_data_plot = cls_data[cls_data.obs["label"].isin(["DUSP9", "ctrl", "lp_scale1.0"])]
# X_pca = pca.fit_transform(cls_data_plot.X)
# df_pca = pd.DataFrame({
#     "PC1": X_pca[:, 0],
#     "PC2": X_pca[:, 1],
#     "label": cls_data_plot.obs["label"]
# })
# plt.figure(figsize=(6,5))
# sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label", alpha=0.7)
# plt.title(f"PCA of lp_all by obs_all")
# plt.savefig(sc.settings.figdir / f"lp_intervene_pca_{data_set_name}_{gene_select}_all_scale.png", dpi=300)
# plt.close()



# %%
# import torch
# from torch import nn, Tensor
# from evaluation.steering.train_cls_decoder import ClsDecoder, ClsLabel

# cls_label = ClsLabel(condition_list_all)
# n_cls = cls_label.unique_len

# cls_decoder = ClsDecoder(d_model=512, n_cls=n_cls, nlayers=3).to(device)
# cls_decoder.load_state_dict(torch.load(BASE_PATH / "evaluation/steering/cls_decoder" / "cls_decoder_trial-1-9.pt"))
# cls_decoder.eval()

# with torch.no_grad():
#     pred_logits = cls_decoder(data_all.to(device)).cpu().numpy()

# pred_logits = pred_logits[:, :n_cls]
# pred_indices = np.argmax(pred_logits, axis=1)
# pred_labels = cls_label.get_conditions(pred_indices)

# y_true = np.concatenate([
#     ctrl_label,
#     pert_label,
#     *[[f"{gene_select}+ctrl"] * intv_act[scale].shape[0] for scale in scale_list]
# ], axis=0)

# df = pd.DataFrame({
#     "obs_condition": obs_all,
#     "y_true": y_true,
#     "y_pred": pred_labels
# })

# %%
# from scipy.spatial.distance import cdist

# def avg_distance(X, Y):
#     return cdist(X, Y).mean()

# dist_list = defaultdict(list)
# for scale in scale_list:
#     disct_ctrl = avg_distance(intv_act[scale], ctrl_act)
#     disct_pert = avg_distance(intv_act[scale], gene_ctrl_act)
#     dist_list[scale] += [disct_ctrl, disct_pert]
#     print(f"Scale {scale}: Distance to ctrl: {disct_ctrl:.4f}, Distance to {gene_select}: {disct_pert:.4f}")


# %%
from sklearn import linear_model

X_train = np.vstack([ctrl_act, gene_ctrl_act])
y_train = np.array([0]*len(ctrl_act) + [1]*len(gene_ctrl_act))

clf = linear_model.SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    random_state=70,
).fit(X_train, y_train)
for scale in scale_list:
    X = intv_act[scale].numpy()
    pred = clf.predict(X).mean()  # fraction of samples labeled as cluster B
    cluster = gene_select if pred > 0.5 else "ctrl"
    print(f"scale {scale}: closer to cluster {cluster} with prediction {pred:.4f}")

# %%
# from sklearn.neural_network import MLPClassifier
# X_train = np.vstack([ctrl_act, gene_ctrl_act])
# y_train = np.array([0]*len(ctrl_act) + [1]*len(gene_ctrl_act))

# clf = MLPClassifier(solver='lbfgs', alpha=1e-2,
#                     hidden_layer_sizes=(5, 2), random_state=1)

# # seeds 22, 30, 35, 70
# clf.fit(X_train, y_train)
# for scale in scale_list:
#     X = intv_act[scale].numpy()
#     pred = clf.predict(X).mean()  # fraction of samples labeled as cluster B
#     cluster = gene_select if pred > 0.5 else "ctrl"
#     print(f"scale {scale}: closer to cluster {cluster} with probability {pred:.4f}")
