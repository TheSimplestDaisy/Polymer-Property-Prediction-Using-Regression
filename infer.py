
import torch
import torch.nn as nn
import pandas as pd
from model import TransPolymer
from PolymerSmilesTokenization import PolymerSmilesTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGETS = ['Eea', 'Egb', 'Egc', 'Ei', 'EPS', 'Nc']
MAX_LENGTH = 175

# Load tokenizer and model
def load_model_and_tokenizer():
    tokenizer = PolymerSmilesTokenizer.from_pretrained("ckpt/poc_fast_model.pt")
    model = TransPolymer()
    model.regression_head = nn.Linear(model.hidden_size, len(TARGETS))
    model.load_state_dict(torch.load("runs/final_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

# Predict properties from SMILES
def predict_properties(smiles, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(smiles, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = model.regression_head(outputs[:, 0])  # CLS token
        preds = preds.cpu().numpy().flatten()
        return dict(zip(TARGETS, preds.tolist()))
