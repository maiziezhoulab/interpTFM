#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from compare_activations_to_concepts import analyze_all_shards_in_set
from pathlib import Path


# In[2]:


import json
from pathlib import Path
from typing import List, Tuple, Union, Optional
import os
import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm
import scanpy as sc
import pandas as pd

import sys
sys.path.append('/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/sae')
from dictionary import AutoEncoder
from inference import get_sae_feats_in_batches, load_sae


# In[6]:


def count_unique_nonzero_dense(matrix: torch.Tensor) -> List[int]:
    """
    Count unique non-zero values in each column of a dense matrix.

    Args:
        matrix: Dense PyTorch tensor to analyze

    Returns:
        List of counts of unique non-zero values for each column
    """
    # Initialize list to store counts
    unique_counts = []

    # Iterate through each column
    for col in range(matrix.shape[1]):
        # Get unique values in the column
        unique_values = torch.unique(matrix[:, col])
        # Count how many unique values are non-zero
        count = torch.sum(unique_values != 0).item()
        unique_counts.append(count)

    return unique_counts


def calc_metrics_dense(
    sae_feats: torch.Tensor,
    per_token_labels_sparse: Union[np.ndarray, sparse.spmatrix],
    threshold_percents: List[float],
    is_aa_level_concept: List[bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized GPU-compatible metric computation for dense matrices.
    """
    device = sae_feats.device
    labels = torch.tensor(per_token_labels_sparse.astype(np.float32), device=device)  # [N, C]
    N, F = sae_feats.shape
    C = labels.shape[1]
    T = len(threshold_percents)

    # Thresholds as tensor [T, 1, 1] for broadcasting
    thresholds = torch.tensor(threshold_percents, dtype=torch.float32, device=device).view(T, 1, 1)

    # Expand and binarize features: [T, N, F]
    feats_exp = sae_feats.unsqueeze(0)  # [1, N, F]
    bin_feats = (feats_exp > thresholds).float()  # [T, N, F]

    # Labels: [1, N, C]
    labels_exp = labels.T.unsqueeze(0)  # [1, C, N]

    # Calculate TP: [T, C, F]
    tp = torch.matmul(labels_exp, bin_feats)  # [T, C, F]
    tp = tp.permute(1, 2, 0).contiguous()  # [C, F, T]

    # Calculate FP: [T, C, F]
    not_labels_exp = (1.0 - labels.T).unsqueeze(0)  # [1, C, N]
    fp = torch.matmul(not_labels_exp, bin_feats)  # [T, C, F]
    fp = fp.permute(1, 2, 0).contiguous()  # [C, F, T]

    # Calculate TP per domain for non-AA-level only
    tp_per_domain = torch.zeros_like(tp)

    # non_aa_indices = [i for i, flag in enumerate(is_aa_level_concept) if not flag]
    # if non_aa_indices:
    #     non_aa_mask = torch.zeros(C, dtype=torch.bool, device=device)
    #     non_aa_mask[non_aa_indices] = True

    #     # For non-AA concepts: compute domain-level TP (number of examples with ≥1 positive feature)
    #     for t_idx in range(T):
    #         # For each threshold: binary_feats [N, F], labels [N, C]
    #         bf = bin_feats[t_idx]  # [N, F]
    #         l = labels  # [N, C]

    #         # Multiply elementwise [N, F] * [N, C] -> [N, C, F]
    #         combined = (bf.unsqueeze(1) * l.unsqueeze(2))  # [N, C, F]
    #         per_domain_tp = (combined.sum(dim=0) > 0).float()  # [C, F]
    #         tp_per_domain[:, :, t_idx] = per_domain_tp

    positive_labels = labels.sum(dim=0)  # [C]
    positive_labels = positive_labels.view(-1, 1, 1)  # [C, 1, 1]

    fn = positive_labels - tp  # [C, F, T]
    fn = torch.clamp(fn, min=0)  # Optional, to avoid negative values due to float precision

    return tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy(), tp_per_domain.cpu().numpy()


# In[9]:


def load_concept_names(concept_name_path: Path) -> List[str]:
    """Load concept names from a file."""
    with open(concept_name_path, "r") as f:
        return f.read().strip().split("\n")


def process_shard(
    sae: AutoEncoder,
    device: torch.device,
    esm_embeddings_pt_path: str,
    per_token_labels: Union[np.ndarray, sparse.spmatrix],
    threshold_percents: List[float],
    is_aa_concept_list: List[bool],
    feat_chunk_max: int = 512,
    is_sparse: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a shard of data by splitting it into manageable chunks for feature calculation.

    Args:
        sae: Normalized SAE model
        device: PyTorch device to use for computation
        esm_embeddings_pt_path: Path to ESM embeddings file
        per_token_labels: Label matrix
        threshold_percents: List of threshold values to evaluate
        is_aa_concept_list: Boolean flags indicating if each concept is AA-level
        feat_chunk_max: Maximum chunk size for feature processing
        is_sparse: Whether to use sparse matrix operations

    Returns:
        Tuple of arrays (tp, fp, tp_per_domain) containing calculated metrics
    """
    # Load embeddings to specified device
    esm_acts = torch.load(
        esm_embeddings_pt_path, map_location=device, weights_only=True
    )

    # Calculate chunking parameters
    feature_chunk_size = min(feat_chunk_max, sae.dict_size)
    total_features = sae.dict_size
    num_chunks = int(np.ceil(total_features / feature_chunk_size))
    print(f"Calculating over {total_features} features in {num_chunks} chunks")

    # Initialize result arrays
    n_concepts = per_token_labels.shape[1]
    n_thresholds = len(threshold_percents)
    n_features = sae.dict_size
    tp = np.zeros((n_concepts, n_features, n_thresholds))
    fp = np.zeros((n_concepts, n_features, n_thresholds))
    fn = np.zeros((n_concepts, n_features, n_thresholds))
    tp_per_domain = np.zeros((n_concepts, n_features, n_thresholds))

    # Convert labels to appropriate format
    # per_token_labels = (
    #     sparse.csr_matrix(per_token_labels) if is_sparse else per_token_labels.toarray()
    # )

    # Process each chunk of features
    for feature_list in tqdm(np.array_split(range(total_features), num_chunks)):
        # Get SAE features for current chunk
        sae_feats = get_sae_feats_in_batches(
            sae=sae,
            device=device,
            esm_embds=esm_acts,
            chunk_size=1024,
            feat_list=feature_list,
        )

        # Calculate metrics using either sparse or dense implementation
        # if is_sparse:
        #     sae_feats_sparse = sparse.csr_matrix(sae_feats.cpu().numpy())
        #     metrics = calc_metrics_sparse(
        #         sae_feats_sparse,
        #         per_token_labels,
        #         threshold_percents,
        #         is_aa_concept_list,
        #     )
        # else:
        metrics = calc_metrics_dense(
            sae_feats, per_token_labels, threshold_percents, is_aa_concept_list
        )

        # Update results arrays with computed metrics
        tp_subset, fp_subset, fn_subset, tp_per_domain_subset = metrics
        tp[:, feature_list] = tp_subset
        fp[:, feature_list] = fp_subset
        fn[:, feature_list] = fn_subset
        tp_per_domain[:, feature_list] = tp_per_domain_subset

    return (tp, fp, fn, tp_per_domain)


def analyze_concepts(
    adata_path: Path,
    gene_ids_path: Path,
    concepts_path: Path,
    gene_ignore: List,
    sae_dir: Path,
    esm_embds_dir: Path = Path("../../data/processed/embeddings"),
    eval_set_dir: Path = Path("../../data/processed/valid"),
    output_dir: Path = "concept_results",
    threshold_percents: List[float] = [0, 0.15, 0.5, 0.6, 0.8],
    shard: Optional[str] = None, # 'shard_55'
    is_sparse: bool = True,
):
    """
    Analyzes concepts in protein sequences using a Sparse Autoencoder (SAE) model.

    Args:
        sae_dir (Path): Directory containing the normalized SAE model file 'ae_normalized.pt'
        esm_embds_dir (Path, optional): Directory containing ESM embeddings.
        eval_set_dir (Path, optional): Directory containing validation dataset and metadata.
        output_dir (Path, optional): Directory where results will be saved.
        threshold_percents (List[float], optional): List of threshold values for concept detection.
        shard (int | None): Specific shard number to process. Must exist in evaluation set.
        is_sparse (bool, optional): Whether to use sparse matrix operations.

    Returns:
        None: Results are saved to disk as NPZ file with following arrays:
            - tp: True positives counts
            - fp: False positives counts
            - tp_per_domain: True positives counts per domain

    Raises:
        ValueError: If normalized SAE model is not found in sae_dir
        ValueError: If specified shard is not in the evaluation set
    """

    # # Load evaluation set metadata from JSON file
    # with open(eval_set_dir / "metadata.json", "r") as f:
    #     eval_set_metadata = json.load(f)

    # Verify that the normalized SAE model exists
    if not (sae_dir / "ae_normalized.pt").exists():
        raise ValueError(f"Normalized SAE model not found in {sae_dir}")

    # Validate that the specified shard exists in the evaluation set

    # # do we need a validation set for this?
    # if shard not in eval_set_metadata["shard_source"]:
    #     raise ValueError(f"Shard {shard} is not in this evaluation set")

    
    # is_aa_concept_list = [
    #     is_aa_level_concept(concept_name) for concept_name in concept_names
    # ]

    # TODO: 
    # we have to build this ourselved through masking genes not expressed [mask1 * mask2 * expression(a corresponding section of it)]
    # this is a binary mat of shape len(emb lenth) * len(concepts) -> len(aa) * len(func)
    
    ad_ = sc.read_h5ad(adata_path)

    # keep for only current shard
    ad_subset = ad_[ad_.obs["shards"] == shard].copy()

    # remove genes not in vocab
    if "index" in ad_subset.var.columns:
        genes_to_keep = ~ad_subset.var["index"].isin(gene_ignore)
    ad_subset = ad_subset[:, genes_to_keep]

    # print(shard)
    # file_path = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/scgpt/gene_ids/shard_0/all_input_gene_ids.txt"
    with open(gene_ids_path / shard / "all_input_gene_ids.txt", "r") as f:
        gene_ids = f.read().split()
    
    # Load the binary concept-gene matrix
    df = pd.read_csv(concepts_path / 'gene_concepts.csv', index_col=0)  # Set index_col=0 if the first column is a concept name

    # Create a dictionary: gene (column) ➜ list of 0/1 for all concepts
    gene_to_concepts = {gene: df[gene].tolist() for gene in df.columns}
    concept_names = load_concept_names(concepts_path / "gprofiler_gene_concepts_columns.txt")
    # print(concept_names)
    per_token_labels = np.zeros((len(gene_ids), len(concept_names)))

    # print(per_token_labels.shape)

    

    for i, gene in enumerate(gene_ids):
        if gene in gene_to_concepts:
            # print(len(gene_to_concepts[gene]))
            # print(gene_to_concepts[gene])
            per_token_labels[i] = gene_to_concepts[gene]
        else:
            # Keep all-zero vector (already initialized)
            pass

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the normalized SAE model
    sae = load_sae(model_path=sae_dir / "ae_normalized.pt", device=device)

    # Process the shard and get results (true positives, false positives, and true positives per domain)
    (tp, fp, fn, tp_per_domain) = process_shard(
        sae,
        device,
        esm_embds_dir / f"{shard}" / "activations.pt",
        per_token_labels,
        threshold_percents,
        concept_names,
        feat_chunk_max=250,
        is_sparse=is_sparse,
    )

    # Create output directory if it doesn't exist and save results
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{shard}_counts.npz",
        tp=tp,
        fp=fp,
        fn=fn,
        tp_per_domain=tp_per_domain,
    )

    # Save per_token_labels to npz in f"shard_{i}/aa_concepts.npz"
    shard_dir = output_dir / f"{shard}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Convert to sparse matrix (recommended if data is mostly zeros)
    per_token_labels_matrix = sparse.csr_matrix(per_token_labels)
    sparse.save_npz(shard_dir / "gene_concepts.npz", per_token_labels_matrix)

def analyze_all_shards_in_set(
    adata_path: Path,
    sae_dir: Path,
    embds_dir: Path,
    concepts_path: Path,
    eval_set_dir: Path,
    gene_ids_path: Path,
    gene_ignore: List,
    output_dir: Path = "concept_results",
    threshold_percents: List[float] = [0, 0.15, 0.5, 0.6, 0.8],
    is_sparse: bool = True,
):
    """Wrapper to scan calculate metrics across all shards in an evaluation set.

    Args:
        sae_dir (Path): Directory containing the normalized SAE model file 'ae_normalized.pt'
        embds_dir (Path): Directory containing ESM embeddings
        eval_set_dir (Path): Directory containing validation dataset and metadata
        output_dir (Path, optional): Directory where results will be saved.
        threshold_percents (List[float], optional): List of threshold values for concept detection.
        is_sparse (bool, optional): Whether to use sparse matrix operations.

    Returns:
        None: Results for each shard are saved to disk in the output_dir

    Raises:
        FileNotFoundError: If metadata.json is not found in eval_set_dir
        ValueError: If any individual shard analysis fails (inherited from analyze_concepts)
    """
    # Load list of shards to evaluate from metadata
    print(eval_set_dir)
    # with open(eval_set_dir / "metadata.json", "r") as f:
    shards_to_eval = os.listdir(eval_set_dir)
    print(f"Analyzing set {eval_set_dir.stem} with {shards_to_eval} shards")

    # Process each shard sequentially
    for shard in shards_to_eval:
        analyze_concepts(
            adata_path,
            gene_ids_path,
            concepts_path,
            gene_ignore,
            sae_dir,
            embds_dir,
            eval_set_dir,
            output_dir,
            threshold_percents,
            shard,
            is_sparse,
        )


# In[10]:


sae_dir=Path('/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/sae/sae_output_layer4')
embds_dir=Path('/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/scgpt/activations/layer_4')
eval_set_dir=Path('/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/scgpt/test_layer4')
output_dir=Path('/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/output')
gene_ids_path = Path('/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/scgpt/gene_ids')
concepts_path = Path('/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/gprofiler_annotation/')

adata_path = Path('/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad')

filtered_genes = ['RGS5', 'CCL3L3']

analyze_all_shards_in_set(
        adata_path=adata_path,
        sae_dir=sae_dir,
        embds_dir=embds_dir,
        concepts_path=concepts_path,
        eval_set_dir=eval_set_dir,
        output_dir=output_dir,
        gene_ids_path=gene_ids_path,
        gene_ignore=filtered_genes
    )

# todo 1 optimize script efficiency on tp fp calc -> done
# finish f1 and report (last 2 steps) -> done
# vision on ooo / mis / other cell level analysis (ccc)

