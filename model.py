
import torch
import torch.nn as nn

class TransPolymer(nn.Module):
    def __init__(self, hidden_size=768, num_layers=6, num_heads=12, dropout=0.1, max_len=175):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(52000, hidden_size)  # default vocab size
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        # Regression head - assign externally (e.g., nn.Linear)
        self.regression_head = None

    def forward(self, input_ids, attention_mask=None):
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        if attention_mask is not None:
            x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        else:
            x = self.transformer(x)

        return x  # Output full sequence
