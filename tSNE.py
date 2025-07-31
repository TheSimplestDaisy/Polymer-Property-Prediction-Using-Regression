import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

import torch
from torch.nn import MaxPool1d
from torch.utils.data import DataLoader

from transformers import RobertaConfig
from openTSNE import TSNE
import matplotlib.pyplot as plt

from PolymerSmilesTokenization_fixed import PolymerSmilesTokenizer
from dataset import Dataset_Emb, TransPolymerEmbeddings

def emb_convert(file_path, tokenizer, config, tsne_config, device):
    data = pd.read_csv(file_path)
    dataset = Dataset_Emb(data, tokenizer, tsne_config['blocksize'], config)
    dataloader = DataLoader(dataset, batch_size=tsne_config['batch_size'], shuffle=False, num_workers=0)

    all_embeddings = []

    for step, batch in enumerate(dataloader):
        batch = batch.to(device)
        embeddings = batch.squeeze()
        embeddings = torch.transpose(embeddings, dim0=1, dim1=2)
        max_pool = MaxPool1d(kernel_size=tsne_config['blocksize'], padding=0)
        pooled = max_pool(embeddings)
        pooled = torch.transpose(pooled, dim0=1, dim1=2).reshape(pooled.size(0), -1).cpu().detach().numpy()
        all_embeddings.append(pooled)

    return np.vstack(all_embeddings)

def main(tsne_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ FIX: Initialize tokenizer correctly
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base")
    tokenizer.max_len = tsne_config['blocksize']

    # Setup Roberta configuration
    config = RobertaConfig(
        vocab_size=50265,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    label_keys = ["pretrain", "PE_I", "PE_II", "Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "OPV"]
    data_dict = {}

    # Load and embed each dataset
    for key in label_keys:
        print(f"Processing {key}...")
        path_key = f"{key}_path"
        data_dict[key] = emb_convert(tsne_config[path_key], tokenizer, config, tsne_config, device)

    # Fit t-SNE using only pretrain
    print("Fitting t-SNE on pretrain data...")
    tSNE = TSNE(
        perplexity=tsne_config['perplexity'],
        metric=tsne_config['metric'],
        n_jobs=tsne_config['n_jobs'],
        verbose=tsne_config['verbose'],
    )
    pretrain_tSNE = tSNE.fit(data_dict["pretrain"])

    # Transform all other datasets
    embeddings_tsne = {key: pretrain_tSNE.transform(data_dict[key]) if key != "pretrain" else pretrain_tSNE
                       for key in label_keys}

    # Plot
    print("Plotting t-SNE results...")
    color_map = {
        "pretrain": "lightgrey", "PE_I": "maroon", "PE_II": "coral", "Egc": "darkorange",
        "Egb": "gold", "Eea": "lawngreen", "Ei": "green", "Xc": "cyan",
        "EPS": "blue", "Nc": "violet", "OPV": "deeppink"
    }

    fig, ax = plt.subplots(figsize=(15, 15))
    for key in label_keys:
        ax.scatter(
            embeddings_tsne[key][:, 0],
            embeddings_tsne[key][:, 1],
            s=50, edgecolors='None', linewidths=0.4,
            c=color_map[key], label=tsne_config[f"{key}_label"]
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.legend(fontsize=20, loc='upper left')
    plt.savefig(tsne_config['save_path'], bbox_inches='tight')
    print(f"Saved t-SNE plot to {tsne_config['save_path']}")

if __name__ == "__main__":
    tsne_config = yaml.load(open("config_tSNE.yaml", "r"), Loader=yaml.FullLoader)
    main(tsne_config)
