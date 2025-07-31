# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 06:21:21 2025

@author: zzulk
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 06:53:35 2025
@author: zzulk
"""

import pandas as pd
import numpy as np
import yaml
import os

import torch
from torch.nn import MaxPool1d
from torch.utils.data import DataLoader

from transformers import RobertaConfig
from openTSNE import TSNE
import matplotlib.pyplot as plt

from PolymerSmilesTokenization_fixed import PolymerSmilesTokenizer
from dataset import Dataset_Emb


def emb_convert(file_path, tokenizer, config, tsne_config, device):
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}. Skipping...")
        return None

    data = pd.read_csv(file_path)
    if data.empty:
        print(f"⚠️ File is empty: {file_path}. Skipping...")
        return None

    dataset = Dataset_Emb(data, tokenizer, tsne_config['blocksize'], config)
    dataloader = DataLoader(dataset, batch_size=tsne_config['batch_size'], shuffle=False, num_workers=0)

    all_embeddings = []
    for step, batch in enumerate(dataloader):
        if batch is None:
            continue
        batch = batch.to(device)
        embeddings = batch.squeeze()
        embeddings = torch.transpose(embeddings, dim0=1, dim1=2)
        max_pool = MaxPool1d(kernel_size=tsne_config['blocksize'], padding=0)
        pooled = max_pool(embeddings)
        pooled = torch.transpose(pooled, dim0=1, dim1=2).reshape(pooled.size(0), -1).cpu().detach().numpy()
        all_embeddings.append(pooled)

    if not all_embeddings:
        print(f"⚠️ No embeddings generated from {file_path}. Skipping...")
        return None

    return np.vstack(all_embeddings)


def main(tsne_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = PolymerSmilesTokenizer(
        vocab_file=tsne_config['vocab_file'],
        merges_file=tsne_config['merges_file']
    )
    tokenizer.max_len = tsne_config['blocksize']

    config = RobertaConfig(
        vocab_size=50265,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    # auto-create subset files if not exist
    df_pretrain = pd.read_csv(tsne_config['pretrain_path'])
    for size in [500000, 50000, 5000]:
        subset_path = tsne_config[f'pretrain_{size//1000}k_path']
        if not os.path.exists(subset_path):
            df_pretrain.iloc[:size].to_csv(subset_path, index=False)
            print(f"✅ Generated subset: {subset_path}")

    label_keys = [
        "pretrain", "pretrain_500k", "pretrain_50k", "pretrain_5k",
        "PE_I", "PE_II", "Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "OPV"
    ]

    data_dict = {}
    for key in label_keys:
        print(f"Processing {key}...")
        path_key = f"{key}_path"
        if path_key in tsne_config:
            data = emb_convert(tsne_config[path_key], tokenizer, config, tsne_config, device)
            if data is not None:
                data_dict[key] = data
        else:
            print(f"⚠️ Skipping {key}, path '{path_key}' not found in config.")

    print("Fitting t-SNE on pretrain data...")
    tSNE = TSNE(
        perplexity=tsne_config['perplexity'],
        metric=tsne_config['metric'],
        n_jobs=tsne_config['n_jobs'],
        verbose=tsne_config['verbose'],
    )
    pretrain_tSNE = tSNE.fit(data_dict["pretrain"])

    embeddings_tsne = {
        key: pretrain_tSNE.transform(data_dict[key]) if key != "pretrain" else pretrain_tSNE
        for key in data_dict
    }

    print("Plotting t-SNE results...")
    color_map = {
        "pretrain": "lightgrey", "pretrain_500k": "black", "pretrain_50k": "brown", "pretrain_5k": "olive",
        "PE_I": "maroon", "PE_II": "coral", "Egc": "darkorange", "Egb": "gold", "Eea": "lawngreen",
        "Ei": "green", "Xc": "cyan", "EPS": "blue", "Nc": "violet", "OPV": "deeppink"
    }

    fig, ax = plt.subplots(figsize=(15, 15))
    for key in embeddings_tsne:
        ax.scatter(
            embeddings_tsne[key][:, 0],
            embeddings_tsne[key][:, 1],
            s=50, edgecolors='None', linewidths=0.4,
            c=color_map.get(key, "gray"), label=tsne_config.get(f"{key}_label", key)
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.legend(fontsize=20, loc='upper left')
    os.makedirs(os.path.dirname(tsne_config['save_path']), exist_ok=True)
    plt.savefig(tsne_config['save_path'], bbox_inches='tight')
    print(f"✅ Saved t-SNE plot to {tsne_config['save_path']}")


if __name__ == "__main__":
    tsne_config = yaml.load(open("config_tSNE.yaml", "r"), Loader=yaml.FullLoader)
    main(tsne_config)
