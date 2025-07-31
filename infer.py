import torch
import pickle
from model import TransPolymer
from PolymerSmilesTokenization import PolymerSmilesTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ['Eea', 'Egb', 'Egc', 'Ei', 'EPS', 'Nc']

def load_model_and_tokenizer(
    model_path="ckpt/poc_fast_model.pt",
    tokenizer_path="tokenizer.pt"
):
    # Load tokenizer dari fail pickle
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()

    return model, tokenizer

def predict_properties(model, tokenizer, smiles):
    tokens = tokenizer(smiles)
    input_ids = torch.tensor([tokens["input_ids"]]).to(device)
    attention_mask = torch.tensor([tokens["attention_mask"]]).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = outputs.cpu().numpy().flatten()

    return dict(zip(TARGETS, prediction.tolist()))
