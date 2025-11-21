# %%
import math
import os
import sys
import json
from pathlib import Path

from dataclasses import dataclass
import einops
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
# import wandb

# sys.path.insert(0, '/maiziezhou_lab/zihang/SpatialFoundationModel/SFM/model/scgpt_model')
# sys.path.insert(0, '/maiziezhou_lab/zihang/SpatialFoundationModel/SFM/model')
sys.path.insert(0, '/maiziezhou_lab2/yunfei/Projects/FM_temp')
from scgpt.gene_tokenizer import GeneVocab
from scgpt.util import load_pretrained
from scgpt.model.model import TransformerModel
from scgpt.data_collator import DataCollator
from scgpt.tokenizer import GeneVocab

# from cell_embed import get_batch_cell_embeddings

# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
use_fast_transformer = False

MAIN = __name__ == "__main__"

# %%
def create_model_from_pretrain_clean(model_dir: str, device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")) -> TransformerModel:
    ## ----- ----- name files ----- ----- ##
    vocab_file = os.path.join(model_dir, "vocab.json")
    model_config_file = os.path.join(model_dir, "args.json")
    model_file = os.path.join(model_dir, "best_model.pt")

    ## ----- ----- build vocab ----- ----- ##
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.
    vocab.set_default_index(vocab["<pad>"])
    
    ## ----- ----- load model ----- ----- ##
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    pretrained_params = torch.load(model_file, map_location=device)
    load_pretrained(model, pretrained_params, verbose=False)
    model.to(device)
    model.eval()
    return model, vocab, model_configs


# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, count_matrix, gene_ids, batch_ids=None, vocab=None, model_configs=None):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.vocab = vocab
        self.model_configs = model_configs

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, self.model_configs["pad_value"])
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output


# %%
def get_batch_cell_embeddings(
    adata,
    cell_embedding_mode: str = "cls",
    model=None,
    vocab=None,
    max_length=1200,
    batch_size=64,
    model_configs=None,
    gene_ids=None,
    use_batch_labels=False,
):
    """
    Get the cell embeddings for a batch of cells.

    Returns:
        cell_embeddings: np.ndarray of shape [n_cells, emb_dim]
        all_input_gene_ids: list of gene names for all tokens
        all_cell_ids: list of cell ids aligned with gene_ids
    """

    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
    )

    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    if use_batch_labels:
        batch_ids = np.array(adata.obs["batch_id"].tolist())

    if cell_embedding_mode == "cls":
        dataset = Dataset(
            count_matrix, gene_ids, batch_ids if use_batch_labels else None,
            vocab=vocab, model_configs=model_configs
        )
        collator = DataCollator(
            do_padding=True,
            pad_token_id=vocab[model_configs["pad_token"]],
            pad_value=model_configs["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), batch_size),
            pin_memory=True,
        )

        device = next(model.parameters()).device
        cell_embeddings = np.zeros(
            (len(dataset), model_configs["embsize"]), dtype=np.float32
        )

        input_gene_ids_store = []
        input_cell_ids_store = []

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for data_dict in tqdm(data_loader, desc="Embedding cells"):
                input_gene_ids = data_dict["gene"].to(device)     # [B, T]

                B, T = input_gene_ids.shape
                flat_input_gene_ids = input_gene_ids.view(B * T)

                # gene names
                indices = flat_input_gene_ids.cpu().tolist()
                gene_names = vocab.lookup_tokens(indices)

                # cell ids (repeat per token)
                batch_cell_ids = adata.obs_names[count : count + B].tolist()
                cell_ids = [cid for cid in batch_cell_ids for _ in range(T)]

                input_gene_ids_store.append(gene_names)
                input_cell_ids_store.append(cell_ids)

                # embeddings
                src_key_padding_mask = input_gene_ids.eq(
                    vocab[model_configs["pad_token"]]
                )
                embeddings = model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(device)
                    if use_batch_labels else None,
                )

                embeddings = embeddings[:, 0, :]  # CLS embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count : count + len(embeddings)] = embeddings
                count += len(embeddings)

        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        # flatten lists of lists
        all_input_gene_ids = [gene for sublist in input_gene_ids_store for gene in sublist]
        all_cell_ids = [cid for sublist in input_cell_ids_store for cid in sublist]

    else:
        raise ValueError(f"Unknown cell embedding mode: {cell_embedding_mode}")

    return cell_embeddings, all_input_gene_ids, all_cell_ids


def get_batch_activations(
    adata,
    cell_embedding_mode: str = "cls",
    model=None,
    vocab=None,
    max_length=1200,
    batch_size=64,
    model_configs=None,
    gene_ids=None,
    use_batch_labels=False,
) -> np.ndarray:
    """
    Get the cell embeddings for a batch of cells.

    Args:
        adata (AnnData): The AnnData object.
        cell_embedding_mode (str): The mode to get the cell embeddings. Defaults to "cls".
        model (TransformerModel, optional): The model. Defaults to None.
        vocab (GeneVocab, optional): The vocabulary. Defaults to None.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        model_configs (dict, optional): The model configurations. Defaults to None.
        gene_ids (np.ndarray, optional): The gene vocabulary ids. Defaults to None.
        use_batch_labels (bool): Whether to use batch labels. Defaults to False.

    Returns:
        np.ndarray: The cell embeddings.
    """

    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
    )

    # gene vocabulary ids
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    if use_batch_labels:
        batch_ids = np.array(adata.obs["batch_id"].tolist())

    if cell_embedding_mode == "cls":
        dataset = Dataset(
            count_matrix, gene_ids, batch_ids if use_batch_labels else None, vocab=vocab, model_configs=model_configs
        )
        collator = DataCollator(
            do_padding=True,
            pad_token_id=vocab[model_configs["pad_token"]],
            pad_value=model_configs["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), batch_size),
            pin_memory=True,
        )

        device = next(model.parameters()).device
        cell_embeddings = np.zeros(
            (len(dataset), model_configs["embsize"]), dtype=np.float32
        )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for data_dict in tqdm(data_loader, desc="Embedding cells"):
                input_gene_ids = data_dict["gene"].to(device)
                src_key_padding_mask = input_gene_ids.eq(
                    vocab[model_configs["pad_token"]]
                )
                embeddings, activations_list = model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(device)
                    if use_batch_labels
                    else None,
                )

                embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count : count + len(embeddings)] = embeddings
                count += len(embeddings)
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
    else:
        raise ValueError(f"Unknown cell embedding mode: {cell_embedding_mode}")
    return activations_list


# %%
if MAIN:
    model_dir = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"
    model, vocab, model_configs = create_model_from_pretrain_clean(model_dir)

    cell_embeddings = get_batch_cell_embeddings(
        adata,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )


# %%
