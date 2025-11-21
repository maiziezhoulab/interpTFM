"""
Convert a .h5ad to a directories of scGPT layer activations organized
by layer and shard with specific metadata used for SAE training.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import scanpy as sc
from load_model_from_pretrain import get_batch_cell_embeddings, create_model_from_pretrain_clean


def embed_ge_for_all_layers(
    model_dir: str,
    adata,
    output_dir: Path,
    layers: List[int],
    gfm_model_name,
    batch_size=1,
    device=torch.device("mps" if torch.backends.mps.is_available() else "cuda:2" if torch.cuda.is_available() else "cpu"),
    shard=0
):

    model, vocab, configs = create_model_from_pretrain_clean(model_dir=model_dir, device=device)

    adata.var["id_in_vocab"] = vocab.lookup_indices(adata.var["index"].to_list())

    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    total_cells = len(adata)
    # cell_ids = adata.obs_names.tolist()

    # Prepare hook handlers dict
    h_list = {}
    shard_of_genes = []

    # Dictionary to store activations per layer
    activation_store = {}

    for start in tqdm(range(0, total_cells, batch_size), desc="Embedding cells"):
        end = min(start + batch_size, total_cells)
        batch_adata = adata[start:end].copy()
        current_cell_ids = batch_adata.obs_names.tolist()
        current_cell_types = batch_adata.obs["author_cell_type"].tolist()
        # print(current_cell_ids, current_cell_types)
        # Define hooks dynamically with access to current batch IDs and cell types


        # [bs, a, 512] [bs, b, 512] [bs, c, 512] [bs, d, 512]
        # 0 1 0 0 0 1
        # 1 0 1 0 0 1
        # padding?
        # -> [a+b+c+d, 512]
        # def getActivation(layer_name):
        #     def hook(model, input, output):
        #         batch_activations = output.detach().cpu()  # [B, ..., H]
        #         print(batch_activations.shape)
        #         for i, (cell_id, cell_type) in enumerate(zip(current_cell_ids, current_cell_types)):
        #             safe_cell_type = str(cell_type).replace("/", "_").replace(" ", "_")  # Avoid path issues
        #             filename = f"{cell_id}_{safe_cell_type}.pt"
        #             cell_file = output_dir / f"{layer_name}" / filename
        #             cell_file.parent.mkdir(parents=True, exist_ok=True)
        #             torch.save(batch_activations[i], cell_file)
        #             print(i, batch_activations[i].shape)
        #     return hook

        

        def getActivation(layer_name):
            def hook(model, input, output):
                batch_activations = output.detach().cpu()  # shape: [B, ..., H]
                # print(f"{layer_name} activations: {batch_activations.shape}")
                
                # Flatten first two dimensions: [Batch, T, Hidden] → [B*T, H]
                B, T, H = batch_activations.shape
                flat_activations = batch_activations.view(B * T, H)

                # Append to existing activations for the layer, or initialize
                if layer_name not in activation_store:
                    activation_store[layer_name] = [flat_activations]
                else:
                    activation_store[layer_name].append(flat_activations)
            
            return hook

        # Register hooks for this batch
        for layer_num in layers:
            layer_name = f"layer_{layer_num}"
            h = model.transformer_encoder.layers[layer_num].norm2.register_forward_hook(getActivation(layer_name))
            h_list[layer_name] = h

        # Run forward pass for this batch
        with torch.no_grad():
            _, gene_ids, cell_ids = get_batch_cell_embeddings(
                batch_adata,
                cell_embedding_mode="cls",
                model=model,
                vocab=vocab,
                max_length=1200,
                batch_size=batch_size,
                model_configs=configs,
                gene_ids=None,
                use_batch_labels=False,
            )

            shard_of_genes.append((gene_ids, cell_ids))

        # Remove hooks after batch
        for h in h_list.values():
            h.remove()
        h_list.clear()
    
    # Save to .txt file (2-column format: cell_id, gene_id)
    save_path = output_dir / "gene_ids" / f"shard_{shard}" / "cell_gene_pairs.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        for gene_ids, cell_ids in shard_of_genes:
            for cid, gid in zip(cell_ids, gene_ids):
                f.write(f"{cid}\t{gid}\n")

    print(f"Saved cell-gene pairs to {save_path}")
    
    for layer_name, activations_list in activation_store.items():
        all_activations = torch.cat(activations_list, dim=0)  # [total_tokens, H]
        layer_output_dir = output_dir / "activations" / layer_name / f"shard_{shard}"
        save_path = layer_output_dir / "activations.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_activations, save_path)
        print(f"Saved {all_activations.shape} to {save_path}")

        metadata_file = layer_output_dir / "metadata.json"

        # Save metadata
        metadata = {
            "model": 'scgpt',
            "total_tokens": all_activations.shape[0],
            "d_model": 512,
            "dtype": str(all_activations.dtype),
            "layer": layer_name,
            "shard": shard,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

    print(f"Finished saving per-cell activations for {total_cells} cells across layers: {layers}")


def process_h5ad(
    adata,
    output_dir,
    model_name,
    model_dir,
    layers,
    device,
    shards=10,
    batch_size=128
):
    """
    Load an AnnData object from file and extract scGPT layer activations.

    This function reads the `.h5ad` file, computes activations from specified
    layers using a pre-trained scGPT model, and saves them with metadata.

    Args:
        adata_path: Path to input .h5ad file
        output_dir: Directory to save activation outputs
        model_name: Name of the scGPT model (used in metadata)
        model_dir: Path to the pre-trained scGPT model
        layers: List of layer indices to extract
    """

    print(f"Filtered AnnData: now has {adata.n_obs} cells (rows)")

    for i in range(shards):
        adata_subset = adata[adata.obs['shards'] == f'shard_{i}'].copy()
        embed_ge_for_all_layers(
            model_dir=model_dir,
            adata=adata_subset,
            output_dir=output_dir,
            layers=layers,
            gfm_model_name=model_name,
            device=device,
            shard=i,
            batch_size=batch_size
        )


if __name__ == "__main__":
    # from tap import tapify
    # tapify(process_shard_range)
    pass