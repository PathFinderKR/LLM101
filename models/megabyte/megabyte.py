# models/megebyte/megabyte.py

from einops import rearrange
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


@dataclass
class MegabyteConfig:
    vocab_size: int
    pad_id: int
    patch_size: int
    patch_num: int
    global_config: GPTConfig
    local_config: GPTConfig

    def __post_init__(self):
        if self.patch_size <= 2:
            raise ValueError("patch_size cannot be less than 2")

        # Global
        self.global_config = GPTConfig(
            vocab_size=self.vocab_size,
            context_size=self.patch_size * self.patch_num,
            n_layer=self.global_config.n_layer,
            n_head=self.global_config.n_head,
            d_embed=self.global_config.d_embed,
            d_ff=self.global_config.d_ff,
            dropout=self.global_config.dropout
        )
        # Local
        self.LocalConfig = GPTConfig(
            vocab_size=self.vocab_size,
            context_size=self.patch_size,
            n_layer=self.local_config.n_layer,
            n_head=self.local_config.n_head,
            d_embed=self.local_config.d_embed,
            d_ff=self.local_config.d_ff,
            dropout=self.local_config.dropout
        )


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

    def forward(self, x): # g_x=(batch_size, patch_num, patch_size * d_embed)
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


class GlobalModel(nn.Module):
    def __init__(self, vocab_size: int, patch_size: int, patch_num: int, n_layer: int, n_head: int, d_embed: int, d_ff: int, dropout: float):
        super(GlobalModel, self).__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.patch_num = patch_num

        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.positional_embedding = nn.Embedding(patch_size * patch_num, d_embed)
        self.layers = nn.ModuleList([DecoderLayer(patch_size * d_embed, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(patch_size * d_embed)

    def embed(self, x):  # g_x=(batch_size, patch_num * patch_size)
        batch_size, context_size = x.size()
        assert context_size <= self.patch_size * self.patch_num, \
            f"context_size should be less than or equal to {self.config.context_size}"

        # Embedding
        token_embed = self.token_embedding(x)  # g_x=(batch_size, patch_num * patch_size, d_embed)
        pos_idx = torch.arange(context_size, device=x.device)  # g_x=(batch_size, patch_num * patch_size)
        pos_embed = self.positional_embedding(pos_idx)   # g_x=(batch_size, patch_num * patch_size, d_embed)
        return token_embed + pos_embed

    def forward(self, x):  # g_x=(batch_size, patch_num, patch_size * d_embed)
        # Decoder layers
        for layer in self.layers:
            x = layer(x)  # g_x=(batch_size, patch_num, patch_size * d_embed)

        # Output
        x = self.layer_norm(x)
        return x

class LocalModel(nn.Module):
    def __init__(self, vocab_size: int, patch_size: int, n_layer: int, n_head: int, d_embed: int, d_ff: int, dropout: float):
        super(LocalModel, self).__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size

        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.positional_embedding = nn.Embedding(patch_size, d_embed)
        self.layers = nn.ModuleList([DecoderLayer(d_embed, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_embed)
        self.linear = nn.Linear(d_embed, vocab_size, bias=False)

    def embed(self, x):  # l_x=(batch_size * patch_num, patch_size)
        batch_size, context_size = x.size()
        assert context_size <= self.patch_size, \
            f"context_size should be less than or equal to {self.config.context_size}"

        # Embedding
        token_embed = self.token_embedding(x)  # l_x=(batch_size * patch_num, patch_size, d_embed)
        pos_idx = torch.arange(context_size, device=x.device)  # l_x=(batch_size * patch_num, patch_size)
        pos_embed = self.positional_embedding(pos_idx)  # l_x=(batch_size * patch_num, patch_size, d_embed)
        return token_embed + pos_embed

    def forward(self, x):  # l_x=(batch_size * patch_num, patch_size, d_embed)
        # Decoder layers
        for layer in self.layers:
            x = layer(x)  # l_x=(batch_size * patch_num, patch_size, d_embed)

        # Output
        x = self.layer_norm(x)
        x = self.linear(x)  # l_x=(batch_size * patch_num, patch_size, vocab_size)
        return x


class MEGABYTE(nn.Module):
    def __init__(self, config: MegabyteConfig):
        super(MEGABYTE, self).__init__()
        self.config = config

        self.global_model = GlobalModel(
            vocab_size=config.vocab_size,
            patch_size=config.patch_size,
            patch_num=config.patch_num,
            n_layer=config.global_config.n_layer,
            n_head=config.global_config.n_head,
            d_embed=config.global_config.d_embed,
            d_ff=config.global_config.d_ff,
            dropout=config.global_config.dropout
        )
        self.local_model = LocalModel(
            vocab_size=config.vocab_size,
            patch_size=config.patch_size,
            n_layer=config.local_config.n_layer,
            n_head=config.local_config.n_head,
            d_embed=config.local_config.d_embed,
            d_ff=config.local_config.d_ff,
            dropout=config.local_config.dropout
        )

    def prepare_input(self, bytes):  # bytes: (batch_size, patch_num * patch_size)
        """
        Prepare global and local input by padding the input bytes.
        """
        # Global
        paddig_global = bytes.new(bytes.shape[0], self.config.patch_size).fill_(self.config.pad_id)  # (batch_size, patch_size)
        # bytes[:, :-self.patch_size] = (batch_size, (patch_num -1) * patch_size)
        bytes_global = torch.cat((paddig_global, bytes[:, :-self.config.patch_size]), -1)  # (batch_size, patch_num * patch_size)

        # Local
        bytes_input = rearrange(bytes, "b (t p) -> (b t) p", p=self.config.patch_size)  # (batch_size * patch_num, patch_size)
        paddig_local = bytes_input.new(bytes_input.shape[0], 1).fill_(self.config.pad_id)  # (batch_size * patch_num, 1)
        # bytes_input[:, :-1] = (batch_size * patch_num, patch_size - 1)
        bytes_local = torch.cat((paddig_local, bytes_input[:, :-1]), -1)  # (batch_size * patch_num, patch_size)

        return bytes_global, bytes_local

    def forward(self, bytes):  # bytes: (batch_size, patch_num * patch_size)
        bytes_global, bytes_local = self.prepare_input(bytes)
        # bytes_global = (batch_size, patch_num * patch_size)
        # bytes_local = (batch_size * patch_num, patch_size)

        global_bytes_embedded = self.global_model.embed(bytes_global)  # (batch_size, patch_num * patch_size, d_embed)
        global_in = rearrange(
            global_bytes_embedded,
            "b (t p) e -> b t (p e)",
            p=self.config.patch_size
        )  # (batch_size, patch_num, patch_size * d_embed)
        global_output = self.global_model(global_in)  # (batch_size, patch_num, patch_size * d_embed)

        global_output_reshaped = rearrange(
            global_output,
            "b t (p e) -> (b t) p e",
            p=self.config.patch_size
        )  # (batch_size * patch_num, patch_size, vocab_size)
        local_bytes_embedded = self.local_model.embed(bytes_local)  # (batch_size * patch_num, patch_size, d_embed)
        local_in = local_bytes_embedded + global_output_reshaped  # (batch_size * patch_num, patch_size, d_embed)
        local_output = self.local_model(local_in)  # (batch_size * patch_num, patch_size, vocab_size)

        batch_size = bytes_global.shape[0]
        x = rearrange(local_output, "(b t) l v -> b (t l) v", b=batch_size)  # (batch_size, patch_num * patch_size, vocab_size)
        return x

    def loss(self, logits, target):
        logits = logits.view(-1, self.config.vocab_size)  # (batch_size * patch_num * patch_size, vocab_size)
        target = target.view(-1)  # (batch_size * patch_num * patch_size)
        return F.cross_entropy(logits, target)

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_new_tokens, device, temperature=1.0):
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")

        self.eval()
        print(prompt)

        # Encode
        x = tokenizer.encode(prompt).to(device).unsqueeze(0)  # (batch_size=1, prompt_size)

        # if prompt is shorter than the context size, pad the prompt
        if x.size(1) < self.config.patch_num * self.config.patch_size:
            pad_size = self.config.patch_num * self.config.patch_size - x.size(1)
            padding = torch.full((1, pad_size), self.config.pad_id, dtype=torch.long, device=device)
            # pad on the left
            x = torch.cat((padding, x), dim=-1)  # (batch_size=1, patch_num * patch_size)

        # Generation loop
        for _ in range(max_new_tokens):
            # Truncate
            context = x[:, -self.config.patch_num * self.config.patch_size:]  # (batch_size=1, patch_num * patch_size)

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


