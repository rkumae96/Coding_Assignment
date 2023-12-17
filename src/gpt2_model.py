import torch
import torch.nn as nn
import math

class GPT2Config:
    vocab_size = 50257
    max_position_embeddings = 1024
    num_layers = 12
    embed_size = 768
    num_heads = 12
    forward_expansion = 4
    dropout_rate = 0.1

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embed size needs to be divisible by num_heads"

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def apply_rotary_emb(qk, cos, sin):
        return torch.einsum("bhid,bhjd->bhij", qk * cos, qk * sin)

    def rotary_embedding(seq_len, dim):
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len).type_as(inv_freq)
        sinusoid_inp = torch.einsum('i,j->ij', t, inv_freq)
        return torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        # Apply Rotary Positional Embedding
        seq_length = query.size(1)
        rotary_emb = rotary_embedding(seq_length, self.head_dim)
        sin, cos = rotary_emb.unbind(dim=-1)
        query, key = apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example with each head
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class GPT2(nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__()
        self.embed_size = config.embed_size
        self.word_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.embed_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config.embed_size, config.num_heads, config.forward_expansion, config.dropout_rate) for _ in range(config.num_layers)]
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc_out = nn.Linear(config.embed_size, config.vocab_size)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)
        return out

    def to(self, device):
        self.device = device
        super().to(device)
        return self
