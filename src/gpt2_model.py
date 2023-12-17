import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Config:
    vocab_size = 50257
    max_position_embeddings = 1024
    n_layers = 12
    n_heads = 12
    n_embd = 768

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, input_ids):
        token_embeddings = self.embed_tokens(input_ids)  # [batch_size, seq_len, n_embd]
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_embeddings = self.embed_positions(position_ids)  # [seq_len, n_embd]

        x = token_embeddings + position_embeddings

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)  # [batch_size, seq_len, n_embd]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.out = nn.Linear(config.n_embd, config.n_embd)
        self.n_heads = config.n_heads

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_heads)**0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# Example usage
if __name__ == "__main__":
    # Define the model
    config = GPT2Config()
    model = GPT2Model(config)

    # Dummy input (batch size 1, sequence length 5)
    input_ids = torch.tensor([[464, 3290, 423, 534, 29]], dtype=torch.long)

    # Forward pass
    outputs = model(input_ids)

    # The output is a tuple with the token representations
    print(outputs.shape)  # (batch_size, sequence_length, config.n_embd)
