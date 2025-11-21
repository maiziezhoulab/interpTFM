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
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:2" if torch.cuda.is_available() else "cpu")

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
# scale_list = [0.3, 0.6, 0.7, 0.8, 0.9]
scale_list = [0.2, 0.4]

condition_list = ["ctrl"]
if f"{gene_select}+ctrl" in condition_list_all:
    condition_list += [f"{gene_select}+ctrl"]
if f"ctrl+{gene_select}" in condition_list_all:
    condition_list += [f"ctrl+{gene_select}"]

assert all([cond in adata.obs["condition"].tolist() for cond in condition_list]), "Some conditions not found in the data"

concept_idx_union = np.where(binary_matrix_renamed_nonzero[gene_select] != 0)[0].tolist()

# %%
adata_cond = adata[adata.obs["condition"].isin(condition_list)]
# all_zero_genes = adata_cond.X.sum(axis=0) > 0

# adata_cond = adata_cond[:, all_zero_genes]
remaining_genes = adata_remaining_genes

adata_ctrl = adata[adata.obs["condition"] == "ctrl"].copy()
gene_select_pos = np.where(remaining_genes == gene_select)[0].item() + 1 # +1 for CLS token

# %% save single gene cls & gene values
# org_act_save_dir = EVAL_DATA_PATH / f"{data_set_name}_intervene" / f"{gene_select}" / "org"

# meta_save_dir = org_act_save_dir / f"meta"
# meta_save_dir.mkdir(parents=True, exist_ok=True)
# with open(meta_save_dir / f"remaining_genes.pkl", "wb") as f:
#     pickle.dump(remaining_genes, f)
# with open(meta_save_dir / f"concept_idx_union.pkl", "wb") as f:
#     pickle.dump(concept_idx_union, f)
# with open(meta_save_dir / f"concept_index_to_term.pkl", "wb") as f:
#     pickle.dump(concept_index_to_term, f)

# batch_size = 256
# cls_act_dict = defaultdict(list)
# gene_act_dict = defaultdict(list)
# for start in tqdm(range(0, adata_cond.n_obs, batch_size)):
#     torch.cuda.empty_cache()
#     gc.collect()
#     end = min(start + batch_size, adata_cond.n_obs)
#     batch_adata = adata_cond[start:end].copy()

#     genes, expressions, attention_mask = tokenizer(
#         batch_adata.X, remaining_genes,
#         include_zero_genes=True, normalize=False, add_cls=True
#     )
    
#     # save obs
#     obs_save_path = meta_save_dir / f"obs_{start}_{end}.pkl"
#     obs_save_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(obs_save_path, "wb") as f:
#         pickle.dump(batch_adata.obs, f)
    
#     # save lp activations
#     # cls_act_dict = {}
#     # gene_act_dict = {}
#     with torch.no_grad(), scgpt_nn.trace(genes, expressions, attention_mask):
#         for layer in range(scgpt_nn.config.nlayers):
#             act = scgpt_nn.transformer_encoder.layers[layer].norm2.output
#             cls_act = act[:, 0, :]
#             cls_act_dict[layer].append(cls_act.cpu().save())
#             gene_act = act[:, gene_select_pos, :]
#             gene_act_dict[layer].append(gene_act.cpu().save())

# for layer, cls_act in cls_act_dict.items():
#     cls_act = torch.cat(cls_act, dim=0)
#     save_path = org_act_save_dir / "cls_act" / f"layer_{layer}" / f"cls_act.pt"
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(cls_act, save_path)

# for layer, gene_act in gene_act_dict.items():
#     gene_act = torch.cat(gene_act, dim=0)
#     save_path = org_act_save_dir / "gene_act" / f"layer_{layer}" / f"gene_act.pt"
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(gene_act, save_path)

# del cls_act_dict
# del gene_act_dict
# torch.cuda.empty_cache()
# gc.collect()

# %% helper_fn & intervene
def apply_scale(
    resid: Float[torch.Tensor, "batch seq_len d_model"],
    flip_dir: Float[torch.Tensor, "d_model"],
    scale: torch.Tensor,
    pos: int,
) -> torch.Tensor:
    flip_dir_normed = flip_dir / flip_dir.norm()
    alpha = resid[:, pos] @ flip_dir_normed
    resid[:, pos] += (scale) * alpha * flip_dir_normed
    return resid

def apply_scale_2D(
    resid: Float[torch.Tensor, "batch seq_len d_model"],
    flip_dir: Float[torch.Tensor, "d_model n_concepts"],
    scale: torch.Tensor,
    pos: int,
) -> torch.Tensor:
    flip_dir_normed = flip_dir / flip_dir.norm(dim=0, keepdim=True)
    alpha = einops.einsum(
        resid[:, pos],
        flip_dir_normed,
        "... d_model, d_model n_concepts -> ... n_concepts"
    )
    resid[:, pos] += (scale) * einops.einsum(
        alpha,
        flip_dir_normed,
        "... n_concepts, d_model n_concepts -> ... d_model"
    )
    return resid

intv_act_save_dir = EVAL_DATA_PATH / f"{data_set_name}_intervene" / f"{gene_select}" / "intervene"
for scale in scale_list:
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Intervening with scale {scale}")
    activation_list = []
    batch_size = 256
    for start in tqdm(range(0, adata_ctrl.n_obs, batch_size)):
        torch.cuda.empty_cache()
        gc.collect()
        end = min(start + batch_size, adata_ctrl.n_obs)
        batch_adata = adata_ctrl[start:end].copy()

        genes, expressions, attention_mask = tokenizer(
            batch_adata.X, remaining_genes,
            include_zero_genes=True, normalize=False, add_cls=True
        )

        with torch.no_grad(), scgpt_nn.trace(genes, expressions, attention_mask):
            for layer in range(scgpt_nn.config.nlayers):
                acts = scgpt_nn.transformer_encoder.layers[layer].norm2.output
                # acts_old = torch.stack(acts.unbind(), dim=0).save()
                apply_scale_2D(
                    acts,
                    lp_all[layer][...,concept_idx_union],
                    scale=scale,
                    pos=gene_select_pos,
                )
            intv_act = scgpt_nn.output.cpu().save()
        
        intv_act_cls = intv_act[:, 0, :]
        activation_list.append(intv_act_cls)

    activation_list = torch.cat(activation_list, dim=0)
    save_path = intv_act_save_dir / f"scale_{scale}" / f"intv_act_cls.pt"
    # print(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(activation_list, save_path)

    # for layer, act in activations_dict.items():
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     act = torch.stack(act.unbind(), dim=0)  # (batch, seq_len, d_model)
    #     lp_act = einops.einsum(
    #         act,
    #         lp_all[layer][...,concept_idx_union],
    #         "batch seq_len d_model, d_model num_concepts -> batch seq_len num_concepts"
    #     ).cpu()
        
    #     save_path = intervene_save_dir / f"layer_{layer}" / f"lp_act_{start}_{end}.pkl"
    #     save_path.parent.mkdir(parents=True, exist_ok=True)
    #     with open(save_path, "wb") as f:
    #         pickle.dump(lp_act, f)

del activation_list
torch.cuda.empty_cache()
gc.collect()

# %% save all cell cls
# org_act_save_dir = EVAL_DATA_PATH / f"{data_set_name}_intervene" / f"condition_org"

# meta_save_dir = org_act_save_dir / f"meta"
# meta_save_dir.mkdir(parents=True, exist_ok=True)
# with open(meta_save_dir / f"remaining_genes.pkl", "wb") as f:
#     pickle.dump(adata_remaining_genes, f)
# with open(meta_save_dir / f"concept_idx_union.pkl", "wb") as f:
#     pickle.dump(concept_idx_union, f)
# with open(meta_save_dir / f"concept_index_to_term.pkl", "wb") as f:
#     pickle.dump(concept_index_to_term, f)

# # batch_size = 128
# # for start in tqdm(range(0, adata.n_obs, batch_size)):
# #     torch.cuda.empty_cache()
# #     gc.collect()
# #     end = min(start + batch_size, adata.n_obs)
# #     batch_adata = adata[start:end].copy()

# #     genes, expressions, attention_mask = tokenizer(
# #         batch_adata.X, adata_remaining_genes,
# #         include_zero_genes=True, normalize=False, add_cls=True
# #     )
    
# #     # save obs
# #     obs_save_path = meta_save_dir / f"obs_{start}_{end}.pkl"
# #     obs_save_path.parent.mkdir(parents=True, exist_ok=True)
# #     with open(obs_save_path, "wb") as f:
# #         pickle.dump(batch_adata.obs, f)
    
# #     # save lp activations
# #     with torch.no_grad(), scgpt_nn.trace(genes, expressions, attention_mask):
# #         org_act = scgpt_nn.output.cpu().save()
    
# #     org_act_cls = org_act[:, 0, :]
# #     save_path = org_act_save_dir / f"org_act_cls_{start}_{end}.pkl"
# #     save_path.parent.mkdir(parents=True, exist_ok=True)
# #     with open(save_path, "wb") as f:
# #         pickle.dump(org_act_cls, f)

# for condition in tqdm(condition_list_all):
#     torch.cuda.empty_cache()
#     gc.collect()
#     activation_list = []

#     adata_select = adata[adata.obs["condition"] == condition].copy()

#     # save obs
#     obs_save_path = meta_save_dir / f"{condition}_obs.pkl"
#     obs_save_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(obs_save_path, "wb") as f:
#         pickle.dump(adata_select.obs, f)

#     batch_size = 128
#     for start in range(0, adata_select.n_obs, batch_size):
#         torch.cuda.empty_cache()
#         gc.collect()
#         end = min(start + batch_size, adata_select.n_obs)
#         batch_adata = adata_select[start:end].copy()

#         genes, expressions, attention_mask = tokenizer(
#             batch_adata.X, adata_remaining_genes,
#             include_zero_genes=True, normalize=False, add_cls=True
#         )
        
#         with torch.no_grad(), scgpt_nn.trace(genes, expressions, attention_mask):
#             org_act = scgpt_nn.output.cpu().save()
            
#         activation_list += [org_act[:, 0, :]]
    
#     activation_list = torch.cat(activation_list, dim=0)
#     # save activations for this condition
#     save_path = org_act_save_dir / f"{condition}_activations.pt"
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(activation_list, save_path)
#     print(f"Saved activations for condition {condition} to {save_path}")


# %% save single gene lp values (depracated)
# org_act_save_dir = EVAL_DATA_PATH / f"{data_set_name}_intervene" / f"{gene_select}" / "org"

# meta_save_dir = org_act_save_dir / f"meta"
# meta_save_dir.mkdir(parents=True, exist_ok=True)
# with open(meta_save_dir / f"remaining_genes.pkl", "wb") as f:
#     pickle.dump(remaining_genes, f)
# with open(meta_save_dir / f"concept_idx_union.pkl", "wb") as f:
#     pickle.dump(concept_idx_union, f)
# with open(meta_save_dir / f"concept_index_to_term.pkl", "wb") as f:
#     pickle.dump(concept_index_to_term, f)

# batch_size = 128
# for start in tqdm(range(0, adata_cond.n_obs, batch_size)):
#     torch.cuda.empty_cache()
#     gc.collect()
#     end = min(start + batch_size, adata_cond.n_obs)
#     batch_adata = adata_cond[start:end].copy()

#     genes, expressions, attention_mask = tokenizer(
#         batch_adata.X, remaining_genes,
#         include_zero_genes=True, normalize=False, add_cls=True
#     )
    
#     # save obs
#     obs_save_path = meta_save_dir / f"obs_{start}_{end}.pkl"
#     obs_save_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(obs_save_path, "wb") as f:
#         pickle.dump(batch_adata.obs, f)
    
#     # save lp activations
#     lp_act_dict = {}
#     with torch.no_grad(), scgpt_nn.trace(genes, expressions, attention_mask):
#         for layer in range(scgpt_nn.config.nlayers):
#             act = scgpt_nn.transformer_encoder.layers[layer].norm2.output
#             lp_act = einops.einsum(
#                 act,
#                 lp_all[layer][...,concept_idx_union],
#                 "batch seq_len d_model, d_model num_concepts -> batch seq_len num_concepts"
#             )
#             lp_act_dict[layer] = lp_act.cpu().save()

#     for layer, lp_act in lp_act_dict.items():        
#         save_path = org_act_save_dir / "lp_act" / f"layer_{layer}" / f"lp_act_{start}_{end}.pkl"
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(save_path, "wb") as f:
#             pickle.dump(lp_act, f)
