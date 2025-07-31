
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from PolymerSmilesTokenization import PolymerSmilesTokenizer
from model import TransPolymer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ['Eea', 'Egb', 'Egc', 'Ei', 'EPS', 'Nc']
MAX_LENGTH = 175
BATCH_SIZE = 32
EPOCHS = 10
LR = 5e-5

# Dataset
class PolymerDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.dropna(subset=TARGETS)
        self.tokenizer = tokenizer
        self.smiles = self.df['SMILES'].tolist()
        self.targets = self.df[TARGETS].values.astype(np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        encoding = self.tokenizer(
            smi,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.targets[idx])
        return item

# Load data
df = pd.read_csv('data/polymer_props.csv')
tokenizer = PolymerSmilesTokenizer.from_pretrained('ckpt/poc_fast_model.pt')
dataset = PolymerDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load base model
model = TransPolymer(vocab_size=tokenizer.vocab_size)
model.regression_head = nn.Linear(model.hidden_size, len(TARGETS))  # 6 outputs
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = model.regression_head(outputs[:, 0])  # CLS token

        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

# Save model
os.makedirs('runs', exist_ok=True)
torch.save(model.state_dict(), 'runs/final_model.pt')
print("✅ Model saved to: runs/final_model.pt")
