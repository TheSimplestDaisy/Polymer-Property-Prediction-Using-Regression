import torch
from src.model import TransPolymer
from src.tokenizer import PolymerSmilesTokenizer

def load_model_and_tokenizer():
    tokenizer = PolymerSmilesTokenizer()
    model = TransPolymer()
    model.load_state_dict(torch.load("runs/final_model.pt", map_location="cpu"))
    model.eval()
    return tokenizer, model

def preprocess_input(smiles, temp, mw):
    # Example preprocessing
    return {"smiles": smiles, "temp": temp, "mw": mw}

def predict_property(inputs, model, tokenizer):
    tokens = tokenizer.encode(inputs["smiles"])
    # Dummy tensor shape; replace with actual model input formatting
    x = torch.tensor(tokens).unsqueeze(0)
    pred = model(x)  # assuming direct output
    return pred.item()