# models/gpt/gpt.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    context_size: int
    n_layer: int
    n_head: int
    d_embed: int
    d_ff: int
    dropout: float

    def __post_init__(self):
        if self.d_embed % self.n_head != 0:
            raise ValueError("d_embed must be divisible by n_head")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")


class CasualSelfAttention(nn.Module):
    def __init__(self, d_embed: int, n_head: int, dropout: float):
        super(CasualSelfAttention, self).__init__()
        self.n_head = n_head
        self.d_head = d_embed // n_head
        self.scale = self.d_head ** -0.5
        self.dropout = dropout

        self.query = nn.Linear(d_embed, d_embed, bias=False)
        self.key = nn.Linear(d_embed, d_embed, bias=False)
        self.value = nn.Linear(d_embed, d_embed, bias=False)
        self.out = nn.Linear(d_embed, d_embed, bias=False)

    def forward(self, x):
        batch_size, context_size, _ = x.size()

        # Query, Key, Value
        q = self.query(x)  # (batch_size, context_size, d_embed)
        k = self.key(x)  # (batch_size, context_size, d_embed)
        v = self.value(x)  # (batch_size, context_size, d_embed)
        q = q.view(batch_size, context_size, self.n_head, self.d_head).transpose(1, 2)  # (batch_size, n_head, context_size, d_head)
        k = k.view(batch_size, context_size, self.n_head, self.d_head).transpose(1, 2)  # (batch_size, n_head, context_size, d_head)
        v = v.view(batch_size, context_size, self.n_head, self.d_head).transpose(1, 2)  # (batch_size, n_head, context_size, d_head)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, n_head, context_size, context_size)

        # Masking
        mask = torch.triu(torch.ones(context_size, context_size, device=x.device), diagonal=1).bool()  # (context_size, context_size)
        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float('-inf'))  # (batch_size, n_head, context_size, context_size)

        # Softmax
        attn_scores = F.softmax(attn_scores, dim=-1)  # (batch_size, n_head, context_size, context_size)

        # Dropout
        attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.training)  # (batch_size, n_head, context_size, context_size)

        # Weighted Sum
        attn_output = torch.matmul(attn_scores, v)  # (batch_size, n_head, context_size, d_head)

        # Concatenation
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, context_size, self.n_head * self.d_head)  # (batch_size, context_size, d_embed)

        # Output Linear Layer
        x = self.out(attn_output)  # (batch_size, context_size, d_embed)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_embed: int, d_ff: int, dropout: float):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_embed, d_ff)
        self.fc2 = nn.Linear(d_ff, d_embed)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_embed: int, n_head: int, d_ff: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attention = CasualSelfAttention(d_embed, n_head, dropout)
        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.feed_forward = FeedForward(d_embed, d_ff, dropout)
        self.layer_norm2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.positional_embedding = nn.Embedding(config.context_size, config.d_embed)
        self.layers = nn.ModuleList([DecoderLayer(config.d_embed, config.n_head, config.d_ff, config.dropout) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.d_embed)
        self.linear = nn.Linear(config.d_embed, config.vocab_size, bias=False)

    def forward(self, x):  # x: (batch_size, context_size)
        batch_size, context_size = x.size()
        assert context_size <= self.config.context_size, \
            f"context_size should be less than or equal to {self.config.context_size}, but got {context_size}"

        # Embedding
        token_embed = self.token_embedding(x)  # (batch_size, context_size, d_embed)
        pos_idx = torch.arange(context_size, device=x.device)  # (context_size)
        pos_embed = self.positional_embedding(pos_idx)  # (batch_size, context_size, d_embed)
        x = token_embed + pos_embed  # (batch_size, context_size, d_embed)

        # Decoder layers
        for layer in self.layers:
            x = layer(x)  # (batch_size, context_size, d_embed)

        # Output
        x = self.layer_norm(x)
        x = self.linear(x)  # (batch_size, context_size, vocab_size)
        return x

    def loss(self, logits, target):
        logits = logits.view(-1, self.config.vocab_size)  # (batch_size * context_size, vocab_size)
        target = target.view(-1)  # (batch_size * context_size)
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
            context = x[:, -self.config.context_size:]  # (batch_size=1, context_size)

            # Forward
            logits = self.forward(context)[:, -1, :] / temperature  # (batch_size=1, vocab_size)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size=1, 1)

            # Concatenate
            x = torch.cat((x, next_token), dim=-1)  # (batch_size=1, context_size + 1)

            # Decode
            text = tokenizer.decode([next_token[0].item()])
            print(text, end='', flush=True)
