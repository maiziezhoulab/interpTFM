# %%
import pickle
import sys
from pathlib import Path
import numpy as np
import einops
import gc
import os
import pickle
from tqdm import tqdm
from dataclasses import dataclass

# from huggingface_hub import hf_hub_download
import wandb
from dotenv import load_dotenv

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split

# from nnsight import NNsight
# from nnsight import LanguageModel, CONFIG

# %%
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

BASE_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
DATASET_NAME = "norman"

ACTIVATION_PATH = BASE_PATH / "data" / "lp_training" / f"filtered_data_{DATASET_NAME}" / "activations"
GENE_ID_PATH = BASE_PATH / "data" / "lp_training" / f"filtered_data_{DATASET_NAME}" / "gene_ids"

PROBE_PATH = BASE_PATH / "linear_probe" / f"probes_{DATASET_NAME}"

# %%
class ShardedPiece(Dataset):
    def __init__(self, data_path, label_path, shard_size):
        self.data_path = data_path
        self.label_path = label_path
        self.shard_size = shard_size

        self.data = None
        self.labels = None
        self.accessed_indices = set()

    def __len__(self):
        return self.shard_size

    def __getitem__(self, idx):
        if self.data is None or self.labels is None:
            self.data = torch.load(self.data_path, map_location=device, weights_only=True)
            self.labels = torch.load(self.label_path, map_location=device, weights_only=True)

        self.accessed_indices.add(idx)
        if len(self.accessed_indices) == self.shard_size:
            # All indices have been accessed, unload the tensors
            data_item = self.data[idx].clone()
            label_item = self.labels[idx].clone()
            self.data = None
            self.labels = None
            self.accessed_indices.clear()
            gc.collect()
            torch.cuda.empty_cache()
            
            return data_item, label_item
        
        return self.data[idx], self.labels[idx]
    
# %%
class ShardedDataset(Dataset):
    def __init__(self, data_paths, label_paths, shard_sizes):
        assert len(data_paths) == len(label_paths), "Mismatched shard counts"
        assert len(data_paths) == len(shard_sizes), "Mismatched shard sizes"
        self.shard_piece = [
            ShardedPiece(data_paths[i], label_paths[i], shard_sizes[i])
            for i in range(len(data_paths))
        ]
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.shard_sizes = shard_sizes
        self.cumulative_sizes = torch.cumsum(torch.tensor(self.shard_sizes), 0).tolist()

        gc.collect()
        torch.cuda.empty_cache()

    def __len__(self):
        return sum(self.shard_sizes)

    def __getitem__(self, idx):
        # Find which shard this idx belongs to
        shard_idx = next(i for i, cs in enumerate(self.cumulative_sizes) if idx < cs)
        local_idx = idx - (self.cumulative_sizes[shard_idx - 1] if shard_idx > 0 else 0)

        return self.shard_piece[shard_idx][local_idx]

# %%
def construct_dataloader(
    layer: int,
    shard_number: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    torch.manual_seed(42)
    data_paths = [
        ACTIVATION_PATH / f"layer_{layer}" / f"shard_{i}" / "filtered_activations.pt"
        for i in range(shard_number)
    ]
    label_paths = [
        GENE_ID_PATH / "gene_ids" / f"shard_{i}" / "filtered_concepts.pt"
        for i in range(shard_number)
    ]

    data_list = [
        torch.load(data_paths[i], map_location="cpu") for i in range(shard_number)
    ]
    label_list = [
        torch.load(label_paths[i], map_location="cpu") for i in range(shard_number)
    ]
    data_list = torch.cat(data_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data_list, label_list, test_size=0.2, random_state=42, shuffle=True
    )

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, test_loader

def construct_dataloader_memory_efficient(
    layer: int,
    shard_number: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    torch.manual_seed(42)
    data_paths = [
        ACTIVATION_PATH / f"layer_{layer}" / f"shard_{i}" / "filtered_activations.pt"
        for i in range(shard_number)
    ]
    label_paths = [
        GENE_ID_PATH / f"shard_{i}" / "filtered_concepts.pt"
        for i in range(shard_number)
    ]

    shard_sizes = []
    for i in range(shard_number):
        filtered_genes_number_dir = GENE_ID_PATH / f"shard_{i}" / "filtered_genes.csv"
        with open(filtered_genes_number_dir, 'rb') as f:
            filtered_genes = pickle.load(f)
        shard_sizes.append(len(filtered_genes))
    
    train_shard_number = int(0.8 * shard_number)

    train_dataset = ShardedDataset(data_paths[:train_shard_number], label_paths[:train_shard_number], shard_sizes[:train_shard_number])
    test_dataset = ShardedDataset(data_paths[train_shard_number:], label_paths[train_shard_number:], shard_sizes[train_shard_number:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader

    # dataset = ShardedDataset(data_paths, label_paths, shard_sizes)
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=False,
    #                     num_workers=num_workers, pin_memory=pin_memory)
    # return dataloader

    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size

    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    # return train_loader, test_loader


# %%
@dataclass
class ProbeTrainingArgs:
    hidden_size: int = 4096
    options: int = 4 # Number of answer choices (A, B, C, D)

    # Hyperparams for optimizer
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.01

    # Saving & logging
    use_wandb: bool = False
    wandb_entity: str | None = "yunfei-hu-vanderbilt-university"
    wandb_project: str | None = "linear-probe-norman"
    wandb_name: str | None = "trial-1"

    def setup_linear_probe(self):
        torch.manual_seed(42)
        linear_probe = torch.randn(self.hidden_size, self.options, device=device) / np.sqrt(
            self.hidden_size
        )
        linear_probe.requires_grad = True
        linear_probe.to(device)
        return linear_probe

# %%
class LinearProbeTrainer:
    def __init__(self, args: ProbeTrainingArgs):
        self.args = args
        self.linear_probe = args.setup_linear_probe()
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch_X, batch_y):
        probe_logits = einops.einsum(
            batch_X, self.linear_probe, "batch hidden, hidden options -> batch options"
        )
        # probe_probs = probe_logits.softmax(-1)
        # loss = self.criterion(probe_probs, batch_y)       

        probe_logprobs = probe_logits.log_softmax(-1)
        # correct_probe_logprobs = probe_logprobs[torch.arange(batch_y.size(0)), batch_y] # single option
        correct_probe_logprobs = probe_logprobs[batch_y.bool()] # multi-option
        loss = -correct_probe_logprobs.mean()

        if self.args.use_wandb:
            wandb.log(dict(loss=loss.item()), step=self.step)
        self.step += 1

        return loss

    def train(self, train_loader, test_loader):
        self.step = 0
        if self.args.use_wandb:
            load_dotenv()
            assert (wandb_api_key := os.getenv("WANDB_API_KEY")), "WANDB_API_KEY environment variable not set"

            wandb.login(key = wandb_api_key)
            wandb.init(
                entity=self.args.wandb_entity,
                project=self.args.wandb_project,
                name=self.args.wandb_name,
                config=self.args
            )

        optimizer = torch.optim.AdamW(
            [self.linear_probe],
            lr=self.args.lr,
            betas=self.args.betas,
            weight_decay=self.args.weight_decay,
        )

        for epoch in range(self.args.epochs):
            # self.linear_probe.train()
            # print(f"Epoch {epoch + 1}/{self.args.epochs}")
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}"):
                torch.cuda.empty_cache()
                gc.collect()
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                loss = self.training_step(batch_X, batch_y)
                loss.backward()
                optimizer.step()
            
            # self.linear_probe.eval()
            self.evaluate(test_loader)
    
    # def evaluate(self, test_loader):
    #     total = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for batch_X, batch_y in test_loader:
    #             torch.cuda.empty_cache()
    #             batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    #             probe_logits = einops.einsum(
    #                 batch_X, self.linear_probe, "batch hidden, hidden options -> batch options"
    #             )
    #             predictions = probe_logits.argmax(dim=-1)
    #             total += batch_y.size(0)
    #             correct += (predictions == batch_y).sum().item()
        
    #     accuracy = correct / total
    #     print(f"Test Accuracy: {accuracy:.4f}")
    #     if self.args.use_wandb:
    #         wandb.log(dict(test_accuracy=accuracy), step=self.step)
    #     return accuracy

    def evaluate(self, test_loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                torch.cuda.empty_cache()
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                probe_logits = einops.einsum(
                    batch_X, self.linear_probe, "batch hidden, hidden options -> batch options"
                )

                topk_per_row = batch_y.sum(dim=1)
                topk_hits = torch.zeros_like(batch_y, dtype=torch.bool)
                for i in range(batch_y.shape[0]):
                    _, topk_idx = torch.topk(probe_logits[i], int(topk_per_row[i].item()))
                    topk_hits[i, topk_idx] = True
                predictions = topk_hits & batch_y.bool()

                total += batch_y.sum().item()
                correct += predictions.sum().item()
        
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        if self.args.use_wandb:
            wandb.log(dict(test_accuracy=accuracy), step=self.step)
        return accuracy

# %%
def main():
    hidden_size = 512
    options = 1179
    # layer = 4
    layer = int(sys.argv[1])
    epochs=20

    print(f"Training linear probe for layer {layer}")
    torch.set_grad_enabled(True)

    args = ProbeTrainingArgs(
        hidden_size=hidden_size,
        options=options, ##### concept number #####
        epochs=epochs,
        batch_size=8192,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        use_wandb=True,
        wandb_name=f"trial-L{layer}", ##### wandb run name #####
    )

    ###### dataset loading ######
    # train_loader, test_loader = construct_dataloader(
    #     layer=layer,
    #     shard_number=60,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=False,
    # )

    train_loader, test_loader = construct_dataloader_memory_efficient(
        layer=layer,
        shard_number=60,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    ##### training ######
    trainer = LinearProbeTrainer(args)
    trainer.train(train_loader, test_loader)

    probe_save_dir = PROBE_PATH / f"linear_probe_{args.wandb_name}.pt"
    probe_save_dir.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.linear_probe, probe_save_dir)
    print("Training complete and model saved.")
    # return trainer

# %%
if __name__ == "__main__":
    main()
    