# models/mlp/mlp.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class MLPConfig:
    vocab_size: int
    context_size: int
    d_embed: int
    d_ff: int
    dropout: float


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_embed)

        self.mlp = nn.Sequential(
            nn.Linear(config.context_size * config.d_embed, config.d_ff),
            nn.LayerNorm(config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),

            nn.Linear(config.d_ff, config.d_ff // 2),
            nn.LayerNorm(config.d_ff // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),

            nn.Linear(config.d_ff // 2, config.vocab_size)
        )

    def forward(self, x):  # x: (batch_size, context_size)
        x = self.token_embedding(x)  # x: (batch_size, context_size, d_embed)
        # concatenate context words
        x = x.view(x.size(0), -1)  # x: (batch_size, context_size * d_embed)
        x = self.mlp(x)  # x: (batch_size, vocab_size)
        return x

    def loss(self, logits, targets):  # logits: (batch_size, vocab_size), targets: (batch_size, context_size)
        targets = targets[:, -1] # (batch_size)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_new_tokens, device, temperature=1.0):
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")

        self.eval()
        print(prompt)

        # Encode
        x = tokenizer.encode(prompt).to(device).unsqueeze(0)  # (batch_size=1, prompt_size)

        # Generation loop
        for _ in range(max_new_tokens):
            # Truncate
            context = x[:, -self.config.context_size:]  # (batch_size=1, context_size)

            # Forward
            logits = self.forward(context) / temperature  # (batch_size=1, vocab_size)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size=1, 1)

            # Concatenate
            x = torch.cat((x, next_token), dim=-1)  # (batch_size=1, context_size + 1)

            # Decode
            text = tokenizer.decode([next_token[0].item()])
            print(text, end='', flush=True)
