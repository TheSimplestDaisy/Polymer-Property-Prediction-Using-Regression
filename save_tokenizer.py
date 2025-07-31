# src/save_tokenizer.py

import os
from PolymerSmilesTokenization import PolymerSmilesTokenizer

# Tentukan lokasi fail vocab dan merges (dari tokenizer asal)
vocab_file = os.path.abspath("ckpt/roberta-base/vocab.json")
merges_file = os.path.abspath("ckpt/roberta-base/merges.txt")

# Cipta tokenizer manual (bukan from_pretrained)
tokenizer = PolymerSmilesTokenizer(
    vocab_file=vocab_file,
    merges_file=merges_file,
    errors="replace",
    bos_token="<s>",
    eos_token="</s>",
    sep_token="</s>",
    cls_token="<s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

# Simpan tokenizer ke dalam folder model anda
save_path = "ckpt/poc_fast_model.pt"
tokenizer.save_pretrained(save_path)

print(f"✅ Tokenizer PolymerSmilesTokenizer berjaya disimpan ke: {save_path}")
