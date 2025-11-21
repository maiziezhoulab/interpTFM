# %%
from pathlib import Path
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import einops

# %%
# device = "cuda:1" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")
torch.set_grad_enabled(False)

BASE_PATH = Path("/maiziezhou_lab2/yunfei/Projects/interpTFM/")

# %% Load concept-gene mapping
concept_matrix_dir = BASE_PATH / "gprofiler_annotation" / "gene_concepts.csv"
concept_gene_map = pd.read_csv(concept_matrix_dir, index_col=0)
concept_gene_map = concept_gene_map.sort_index()

# %% Load Linear probes
lp_all = {}
for layer in range(12):
    lp_path = BASE_PATH / "linear_probe" / "probes" / f"linear_probe_trial-L{layer}.pt"
    lp = torch.load(lp_path, map_location="cpu")
    lp_all[layer] = lp
    print(f"Loaded linear probe for layer {layer}")

lp_tensor = torch.stack([lp_all[layer] for layer in range(12)], dim=0).to(device)  # (12, d_model, num_concepts)
lp_tensor_norm = lp_tensor / lp_tensor.norm(dim=1, keepdim=True)  # (12, d_model, num_concepts)

# %% linear probe similarity across layers
# concept_id = 0
# cos_sim_concept = lp_tensor_norm[:, :, concept_id] @ lp_tensor_norm[:, :, concept_id].T  # (12, 12)
cos_sim_concept = einops.einsum(
    lp_tensor_norm,
    lp_tensor_norm,
    "layer1 d_model concept_id, layer2 d_model concept_id -> layer1 layer2 concept_id",
)
cos_sim_concept = einops.reduce(cos_sim_concept, "layer1 layer2 concept_id -> layer1 layer2", "mean")

plt.figure(figsize=(8, 6))
sns.heatmap(cos_sim_concept.cpu(), annot=True, fmt=".2f", cmap="viridis")
plt.title(f"Cosine Similarity of linear probes between Layers across concepts")
plt.xlabel("Layer")
plt.ylabel("Layer")
plt.xticks(ticks=[0.5+i for i in range(12)], labels=[f"L{i}" for i in range(12)])
plt.yticks(ticks=[0.5+i for i in range(12)], labels=[f"L{i}" for i in range(12)], rotation=0)
plt.tight_layout()
plt.show()

# %%
# choose a layer
layer_id = 4
lp_layer = lp_tensor_norm[layer_id, :, :]  # (d_model, num_concepts)

cos_sim_layer = einops.einsum(
    lp_layer,
    lp_layer,
    "d_model concept_id1, d_model concept_id2 -> concept_id1 concept_id2",
)

# plt.figure(figsize=(8, 6))
# sns.heatmap(cos_sim_layer.cpu()[:10,:10], annot=True, fmt=".2f", cmap="viridis")
# plt.xlabel("Concept ID")
# plt.ylabel("Concept ID")
# plt.title(f"Cosine Similarity of linear probes within Layer {layer_id}")
# plt.tight_layout()
# plt.show()

# %% use the d_model dimension to perform clustering for concepts

