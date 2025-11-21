# %%
import pickle
from pathlib import Path
from collections import defaultdict
# import torch as t
import numpy as np
import einops
import pandas as pd
# import eindex
from typing import Callable, Optional, Union
import pickle
from time import sleep
from tqdm import tqdm, trange

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from dataclasses import dataclass
from datasets import load_dataset
# from huggingface_hub import hf_hub_download
import torch
import wandb

import sys
sys.path.append(str(Path("./").resolve().parent))
from train.load_sharded_acts import LazyMultiDirectoryTokenDataset

# from nnsight import NNsight
# from nnsight import LanguageModel, CONFIG

# %%
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
torch.set_grad_enabled(True)

BASE_PATH = Path("/maiziezhou_lab2/yunfei/Projects/interpTFM/")
SAVE_PATH = Path("/maiziezhou_lab2/zihang/interpTFM/linear_probe/filtered_data/")

# %%

# gfm_embd_dir='/maiziezhou_lab2/yunfei/Projects/interpTFM/activations_cosmx_lung_cancer/activations/layer_4'

# acts_dataset = LazyMultiDirectoryTokenDataset(gfm_embd_dir)

# layer = acts_dataset.datasets[0]["layer"]
# plm_name = acts_dataset.datasets[0]["plm_name"]

# print(f"Using activations from layer {layer} of {plm_name}")

# # dataloader = DataLoader(
# #     acts_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
# # )
# print(f"Loaded dataset with {len(acts_dataset):,} tokens")

# %%
concept_matrix_dir = BASE_PATH / "gprofiler_annotation" / "gene_concepts.csv"
concept_gene_map = pd.read_csv(concept_matrix_dir, index_col=0)
concept_gene_map = concept_gene_map.sort_index()

# %%
shard_number = 60
layer_number = 12

for shard in trange(shard_number):
    meta_data_dir = BASE_PATH / "activations_cosmx_lung_cancer" / "gene_ids" / f"shard_{shard}" / "cell_gene_pairs.txt"
    meta_df = pd.read_csv(meta_data_dir, sep='\t', index_col=0, header=None)
    mask = meta_df[1].isin(concept_gene_map.columns)
    filtered_genes = meta_df[mask].reset_index(drop=True)
    filtered_concepts = torch.tensor(concept_gene_map[filtered_genes[1]].values.T, dtype=torch.float32)

    filtered_genes_save_dir = SAVE_PATH / "gene_ids" / f"shard_{shard}" / "filtered_genes.csv"
    filtered_concepts_save_dir = SAVE_PATH / "gene_ids" / f"shard_{shard}" / "filtered_concepts.pt"

    filtered_genes_save_dir.parent.mkdir(parents=True, exist_ok=True)
    filtered_concepts_save_dir.parent.mkdir(parents=True, exist_ok=True)

    with open(filtered_genes_save_dir, 'wb') as f:
        pickle.dump(filtered_genes, f)
    
    torch.save(filtered_concepts, filtered_concepts_save_dir)

    for layer in range(layer_number):
        activations_dir = BASE_PATH / "activations_cosmx_lung_cancer" / "activations" / f"layer_{layer}" / f"shard_{shard}" / "activations.pt"

        activations = torch.load(activations_dir)
        filtered_activations = activations[mask]   # torch indexing

        filtered_save_dir = SAVE_PATH / "activations" / f"layer_{layer}" / f"shard_{shard}" / "filtered_activations.pt"
        filtered_save_dir.parent.mkdir(parents=True, exist_ok=True)
        torch.save(filtered_activations, filtered_save_dir)

# %%
shard_number = 60
layer_number = 12

for shard in trange(shard_number):
    meta_data_dir = BASE_PATH / "activations_cosmx_lung_cancer" / "gene_ids" / f"shard_{shard}" / "cell_gene_pairs.txt"
    meta_df = pd.read_csv(meta_data_dir, sep='\t', index_col=0, header=None)
    mask = meta_df[1].isin(concept_gene_map.columns)

    filtered_genes_number_dir = SAVE_PATH / "gene_ids" / f"shard_{shard}" / "filtered_genes_number.txt"
    filtered_genes_number_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(filtered_genes_number_dir, 'wb') as f:
        pickle.dump(mask.sum().item(), f)

# %%
# data_dict = defaultdict(dict)

# if (save_path is not None) and any(save_path.iterdir()):
#     print(f"Loading existing data entries from {save_path}.")
#     for i_file in save_path.iterdir():
#         with open(i_file, "rb") as f:
#             data_entry = pickle.load(f)
#         if data_entry["llm_answer"] != data_entry["correct_answer"]:
#             continue
#         data_dict[data_entry["question_id"]] = data_entry
#     return data_dict

# data_dir = BASE_PATH / "cot_data/"
# activation_dir = BASE_PATH / "activations_cosmx_lung_cancer" / "activations" / f"layer{layer}"
# answer_dir = BASE_PATH / "cot_anthropic_answer_Sonnet/"

# for i in trange(num_samples_max):
#     with open(data_dir / f"mmlu_results_batch_{i}.pkl", "rb") as f:
#         result = pickle.load(f)

#     if result['finish_reason'] != 'finish':
#         # print(f"Warning: Batch {i} did not finish properly with reason {result['finish_reason']}")
#         continue

#     if "</think>" not in result['text']:
#         # print(f"Warning: Batch {i} missing </think> token.")
#         continue

#     with open(activation_dir / f"mmlu_activations_batch_{i}.pkl", "rb") as f:
#         activations = pickle.load(f)

#     with open(answer_dir / f"mmlu_llm_answer_batch_{i}.pkl", "rb") as f:
#         answer = pickle.load(f)
    
#     if answer == "UNKNOWN":
#         # print(f"Warning: Batch {i} has UNKNOWN answer.")
#         continue
    
#     data = result['question']
#     query = prompt_template.format(
#         question=data["question"],
#         choice_text="\n".join(
#             f"{choice_list[idx]}. {text}" for idx, text in enumerate(data["choices"])
#         )
#     )
#     all_text = query + result['text']
#     inputs = model.tokenizer(all_text, padding=True, return_tensors="pt")['input_ids']

#     # ----- ----- after thinking tokens ----- ----- #
#     end_thinking_pos = (inputs == end_thinking_token_id).nonzero(as_tuple=True)[1][0]
#     data_entry = {
#         "question_id": i if tag is None else f"{tag}_{i}",
#         "all_tokens": inputs[0][(end_thinking_pos + 1):],
#         "activation": activations[0][(end_thinking_pos + 1):],
#         "llm_answer": choice_to_idx[answer],
#         "correct_answer": result['question']['answer'],
#     }

#     # ----- ----- all tokens ----- ----- #
#     # data_entry = {
#     #     "question_id": i,
#     #     "all_tokens": inputs[0],
#     #     "activation": activations[0],
#     #     "llm_answer": choice_to_idx[answer],
#     #     "correct_answer": result['question']['answer'],
#     # }

#     # ----- ----- index tokens ----- ----- #
#     # with open(BASE_PATH / "train_probe_index" / f"mmlu_train_data_batch_{i}.pkl", "rb") as f:
#     #     index_data = pickle.load(f)
#     # first_stable_idx = index_data['token_index']
#     # data_entry = {
#     #     "question_id": i,
#     #     "all_tokens": inputs[0][(first_stable_idx + 1):],
#     #     "activation": activations[0][(first_stable_idx + 1):],
#     #     "llm_answer": choice_to_idx[answer],
#     #     "correct_answer": result['question']['answer'],
#     # }

#     # ----- ----- index tokens to thinking tokens ----- ----- #
#     # with open(BASE_PATH / "train_probe_index" / f"mmlu_train_data_batch_{i}.pkl", "rb") as f:
#     #     index_data = pickle.load(f)
#     # first_stable_idx = index_data['token_index']
#     # end_thinking_pos = (inputs == end_thinking_token_id).nonzero(as_tuple=True)[1][0]
#     # data_entry = {
#     #     "question_id": i,
#     #     "all_tokens": inputs[0][(first_stable_idx + 1):(end_thinking_pos+1)],
#     #     "activation": activations[0][(first_stable_idx + 1):(end_thinking_pos+1)],
#     #     "llm_answer": choice_to_idx[answer],
#     #     "correct_answer": result['question']['answer'],
#     # }

#     if data_entry["llm_answer"] != data_entry["correct_answer"]:
#         # print(f"Warning: Batch {i} answer mismatch: LLM answer {data_entry['llm_answer']} vs correct answer {data_entry['correct_answer']}")
#         continue
    
#     data_dict[data_entry["question_id"]] = data_entry

#     if save_path is not None:
#         out_file_name = save_path / f"train_data_{i}.pkl"
#         out_file_name.parent.mkdir(parents=True, exist_ok=True)
#         with open(out_file_name, "wb") as f:
#             pickle.dump(data_entry, f)

# return data_dict
