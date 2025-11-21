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
condition_list_all = adata.obs["condition"].unique().tolist()

# print("KLF1+MAP2K6" in pert_data.adata.obs["condition"].tolist())
# print("KLF1+ctrl" in pert_data.adata.obs["condition"].tolist())
# print("MAP2K6+ctrl" in pert_data.adata.obs["condition"].tolist())

# %% load probes
lp_all = {}
# layer = 11
for layer in range(12):
    print(f"Loading linear probe for layer {layer}")
    lp_path = BASE_PATH / "linear_probe" / f"probes_{data_set_name}" / f"linear_probe_trial-L{layer}.pt"
    lp = torch.load(lp_path, map_location=device)
    lp.requires_grad_(False)
    lp_all[layer] = lp

# %%
binary_matrix_renamed_nonzero = pd.read_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gene_concepts_nonzero.csv", index_col=0)
concepts = pd.read_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gprofiler_gene_concepts_columns.txt", index_col=0)

concept_id_to_index = {cid: idx for idx, cid in enumerate(binary_matrix_renamed_nonzero.index)}
concept_index_to_id = {idx: cid for cid, idx in concept_id_to_index.items()}
concept_index_to_term = {idx: concepts.loc[cid, 'term_name'] for idx, cid in concept_index_to_id.items()}

# %%
gene_select = "DUSP9"
condition_list = ["ctrl"]
if f"{gene_select}+ctrl" in condition_list_all:
    condition_list += [f"{gene_select}+ctrl"]
if f"ctrl+{gene_select}" in condition_list_all:
    condition_list += [f"ctrl+{gene_select}"]
assert all([cond in condition_list_all for cond in condition_list]), "Some conditions not found in the data"

concept_idx_union = np.where(binary_matrix_renamed_nonzero[gene_select] != 0)[0].tolist()

# %%
adata_cond = adata[adata.obs["condition"].isin(condition_list)]
# all_zero_genes = adata_cond.X.sum(axis=0) > 0

# adata_cond = adata_cond[:, all_zero_genes]
remaining_genes = adata_remaining_genes

adata_ctrl = adata[adata.obs["condition"] == "ctrl"].copy()
gene_select_pos = np.where(remaining_genes == gene_select)[0].item() + 1 # +1 for CLS token


# %% read single gene cls (multiple layers)
org_act_save_dir = EVAL_DATA_PATH / f"{data_set_name}_intervene" / f"{gene_select}" / "org"
# org_act = []
cls_act_dict = {}

for layer in range(scgpt_nn.config.nlayers):
    load_path = org_act_save_dir / "cls_act" / f"layer_{layer}" / f"cls_act.pt"
    cls_act_dict[layer] = torch.load(load_path).to(device)

# %%
lp_act_dict = {}
for layer in range(scgpt_nn.config.nlayers):
    lp = lp_all[layer][..., concept_idx_union]
    cls_acts = cls_act_dict[layer]  # (num_cells, seq_len, d_model)
    # lp_acts = torch.einsum("bnd,dc->bnc", cls_acts, lp)  # (num_cells, seq_len, num_concepts)
    lp_act_dict[layer] = einops.einsum(
        cls_acts, lp,
        "batch d_model, d_model num_concepts -> batch num_concepts"
    )

torch.manual_seed(123)
random_lp = torch.randn(512, len(concept_idx_union), device=device) / np.sqrt(
    512
)
random_lp_act_dict = {}
for layer in range(scgpt_nn.config.nlayers):
    cls_acts = cls_act_dict[layer]  # (num_cells, seq_len, d_model)
    random_lp_act = einops.einsum(
        cls_acts, random_lp,
        "batch d_model, d_model num_concepts -> batch num_concepts"
    )
    random_lp_act_dict[layer] = random_lp_act

# %%
ctrl_idx = np.where(
    (adata_cond.obs["condition"].isin([f"ctrl"]))
)[0].tolist()

pert_cond_list = condition_list.copy()
pert_cond_list.remove("ctrl")
pert_1_idx = np.where(
    (adata_cond.obs["condition"].isin(pert_cond_list))
)[0].tolist()

# %% lp act analysis plotting
lp_act_names = [
    concept_index_to_term[c_idx]
    for c_idx in concept_idx_union
]

lp_ctrl_gene_dict = {
    layer: lp_act_dict[layer][ctrl_idx]
    for layer in range(scgpt_nn.config.nlayers)
}

lp_pert_1_gene_dict = {
    layer: lp_act_dict[layer][pert_1_idx]
    for layer in range(scgpt_nn.config.nlayers)
}

all_data = []
for layer in range(scgpt_nn.config.nlayers):
    for cond, tensor in [("ctrl", lp_ctrl_gene_dict), (f"{gene_select}", lp_pert_1_gene_dict)]:
        df_temp = pd.DataFrame(tensor[layer].cpu().numpy())
        df_temp.columns = lp_act_names
        df_temp = df_temp.melt(var_name="feature", value_name="activation")
        df_temp["layer"] = layer
        df_temp["condition"] = cond
        all_data.append(df_temp)
df_all = pd.concat(all_data)

# %%
# df_avg = df_all.groupby(['feature', 'layer', 'condition'], as_index=False)['activation'].mean()

# # 2. pivot so we have columns for each condition
# pivot = df_avg.pivot_table(index=['feature', 'layer'], columns='condition', values='activation')

# # 3. compute difference between CEBPA and ctrl
# pivot['diff'] = pivot['CEBPA'] - pivot['ctrl']

# # 4. average across layers for each feature
# diff_by_feature = pivot.groupby('feature')['diff'].mean().reset_index()

# # 5. optional: sort by difference
# diff_by_feature = diff_by_feature.sort_values('diff', ascending=False)

# print(diff_by_feature.head())

# # 6. Get the top 5 features with largest mean difference
# topn = 5
# top_features = diff_by_feature.nlargest(5, 'diff')['feature']

# # 7. Filter the original dataframe to keep only those features
# df_top = df_all[df_all['feature'].isin(top_features)]


# %%
fig, axs = plt.subplots(2, 6, figsize=(24, 8), sharex=True)

plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

ax_flat = axs.flatten()
for layer in range(scgpt_nn.config.nlayers):
    ax = ax_flat[layer]
    df_layer = df_all[df_all["layer"] == layer]
    sns.violinplot(data=df_layer, x="feature", y="activation", hue="condition", ax=ax)
    ax.set_ylabel("Linear Probe Activation")
    ax.set_xlabel("Features")
    ax.set_title(f"Layer {layer}")
    ax.tick_params(axis='x', rotation=30)

    if layer != 0:
        ax.get_legend().remove()

fig.suptitle(f"Activation Distributions per Layer for cells")
fig.tight_layout()
fig.savefig(f"figures_intv/lp_act_analysis_{data_set_name}_{gene_select}_cell_act.pdf")

# %% lp act analysis plotting
random_act_names = [
    f"random_{i}"
    for i in range(len(concept_idx_union))
]

random_lp_ctrl_gene_dict = {
    layer: random_lp_act_dict[layer][ctrl_idx]
    for layer in range(scgpt_nn.config.nlayers)
}

random_lp_pert_1_gene_dict = {
    layer: random_lp_act_dict[layer][pert_1_idx]
    for layer in range(scgpt_nn.config.nlayers)
}

fig, axs = plt.subplots(2, 6, figsize=(24, 8), sharex=True)
ax_flat = axs.flatten()
for layer in range(scgpt_nn.config.nlayers):
    ax = ax_flat[layer]

    all_data = []
    for cond, tensor in [("ctrl", random_lp_ctrl_gene_dict), (f"{gene_select}", random_lp_pert_1_gene_dict)]:
        df_temp = pd.DataFrame(tensor[layer].cpu().numpy())
        df_temp.columns = random_act_names
        df_temp = df_temp.melt(var_name="feature", value_name="activation")
        df_temp["layer"] = layer
        df_temp["condition"] = cond
        all_data.append(df_temp)

    df_all = pd.concat(all_data)

    sns.violinplot(data=df_all, x="feature", y="activation", hue="condition", ax=ax)
    ax.set_ylabel("Linear Probe Activation")
    ax.set_title(f"Layer {layer}")
    ax.tick_params(axis='x', rotation=45)

    if layer != 0:
        ax.get_legend().remove()

fig.suptitle(f"Activation Distributions per Layer for cells")
fig.tight_layout()
fig.savefig(f"figures_intv/random_lp_act_analysis_{data_set_name}_{gene_select}_cell_act.pdf")

# %% single layer
# fig, ax = plt.subplots(figsize=(24, 8), sharex=True)
# layer = 11
# all_data = []
# for cond, tensor in [("ctrl", lp_ctrl_gene_dict), (f"{gene_select}", lp_pert_1_gene_dict)]:
#     df_temp = pd.DataFrame(tensor[layer].cpu().numpy())
#     df_temp.columns = lp_act_names
#     df_temp = df_temp.melt(var_name="feature", value_name="activation")
#     df_temp["layer"] = layer
#     df_temp["condition"] = cond
#     all_data.append(df_temp)

# df_all = pd.concat(all_data)

# sns.violinplot(data=df_all, x="feature", y="activation", hue="condition", ax=ax)
# ax.set_ylabel("Linear Probe Activation")
# ax.set_title(f"Layer {layer}")
# ax.tick_params(axis='x', rotation=45)


# %% p-value calculation
# lp_p_values = {}
# for layer in range(scgpt_nn.config.nlayers):
#     ctrl_acts = lp_ctrl_gene_dict[layer].cpu().numpy()
#     pert_acts = lp_pert_1_gene_dict[layer].cpu().numpy()
#     p_values = []
#     for concept_idx in range(len(concept_idx_union)):
#         t_stat, p_val = ttest_ind(
#             ctrl_acts[:, concept_idx],
#             pert_acts[:, concept_idx],
#             equal_var=False
#         )
#         p_values.append(p_val)
#     lp_p_values[layer] = p_values

# random_lp_p_values = {}
# for layer in range(scgpt_nn.config.nlayers):
#     ctrl_acts = random_lp_ctrl_gene_dict[layer].cpu().numpy()
#     pert_acts = random_lp_pert_1_gene_dict[layer].cpu().numpy()
#     p_values = []
#     for concept_idx in range(len(concept_idx_union)):
#         t_stat, p_val = ttest_ind(
#             ctrl_acts[:, concept_idx],
#             pert_acts[:, concept_idx],
#             equal_var=False
#         )
#         p_values.append(p_val)
#     random_lp_p_values[layer] = p_values

# fig, ax = plt.subplots(figsize=(10, 6))
# for layer in range(scgpt_nn.config.nlayers):
#     ax.plot(
#         range(len(concept_idx_union)),
#         -np.log10(lp_p_values[layer]),
#         label=f"Layer {layer} LP"
#     )
#     ax.plot(
#         range(len(concept_idx_union)),
#         -np.log10(random_lp_p_values[layer]),
#         linestyle="--",
#         label=f"Layer {layer} Random LP"
#     )

# %%
