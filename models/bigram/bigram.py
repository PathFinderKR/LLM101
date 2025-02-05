# models/bigram/bigram.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class BigramConfig:
    vocab_size: int

class Bigram(nn.Module):
    def __init__(self, config: BigramConfig):
        super(Bigram, self).__init__()
        self.vocab_size = config.vocab_size
        self.probs = nn.Parameter(torch.randn(config.vocab_size, config.vocab_size))

    def forward(self, x):  # x: (batch_size, 1)
        logits = self.probs[x]  # (batch_size, 1, vocab_size)
        return logits

    def loss(self, logits, target):
        logits = logits.view(-1, self.vocab_size)  # (batch_size, vocab_size)
        target = target.view(-1)  # (batch_size)
        return F.cross_entropy(logits, target)

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
            context = x[:, -1:]  # (batch_size=1, 1)

            # Forward
            logits = self.forward(context)[:, -1, :] / temperature  # (batch_size=1, vocab_size)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size=1, 1)

            # Concatenate
            x = torch.cat([x, next_token], dim=-1)  # (batch_size=1, 2)

            # Decode
            text = tokenizer.decode([next_token[0].item()])
            print(text, end='', flush=True)
