class PolymerSmilesTokenizer:
    def encode(self, smiles):
        # Dummy tokenizer
        return [ord(c) % 50 for c in smiles]