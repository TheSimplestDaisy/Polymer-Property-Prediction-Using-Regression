import json
import os
from functools import lru_cache
import regex as re

from transformers import PreTrainedTokenizer, AddedToken
import logging

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
}

@lru_cache()
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class PolymerSmilesTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs
    ):
        with open(vocab_file, encoding="utf-8") as vf:
            self.encoder = json.load(vf)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merges_file, encoding="utf-8") as mf:
            bpe_merges = mf.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # SMILES pattern
        pattern = r"(\-?[0-9]+\.?[0-9]*|\[|\]|SELF|Li|Be|Na|Mg|Al|K|Ca|Co|Zn|Ga|Ge|As|Se|Sn|Te|N|O|P|H|I|b|c|n|o|s|p|Br?|Cl?|Fe?|Ni?|Si?|\||\(|\)|\^|=|#|-|\+|\\|\/|@|\*|\.|\%|\$)"
        self.pat = re.compile(pattern)

        # Special tokens
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False)
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False)
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False)
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False)
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        if len(word) == 1:
            self.cache[token] = token
            return token

        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def _tokenize(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        text = "".join(tokens)
        return bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id, self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1, 1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return len([self.cls_token_id] + token_ids_0 + [self.sep_token_id]) * [0]
        return len([self.cls_token_id] + token_ids_0 + [self.sep_token_id, self.sep_token_id] + token_ids_1 + [self.sep_token_id]) * [0]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        vocab_file = os.path.join(save_directory, (filename_prefix or "") + "vocab.json")
        merges_file = os.path.join(save_directory, (filename_prefix or "") + "merges.txt")

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False)

        with open(merges_file, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for pair, idx in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
                f.write(" ".join(pair) + "\n")

        return vocab_file, merges_file
