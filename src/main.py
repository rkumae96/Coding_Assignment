import torch
from gpt2_model import GPT2, GPT2Config

def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')
    return checkpoint

def main():
    # Initialize configuration and model
    config = GPT2Config()
    model = GPT2(config).to('cpu')

    # Load the original GPT-2 125M model checkpoints
    checkpoint = load_checkpoint("path_to_gpt2_checkpoint.pth")
    model.load_state_dict(checkpoint)

    # Prepare input data. Here we need real input handling with tokenization and padding
    # For demonstration purposes, we use dummy data
    input_ids = torch.tensor([[50256] * config.max_position_embeddings])  # Example token IDs
    mask = torch.ones_like(input_ids).unsqueeze(1).unsqueeze(2)  # Dummy attention mask

    # Forward pass through the model
    with torch.no_grad():
        predictions = model(input_ids, mask)

    print(predictions.shape)  # Expected shape: [batch_size, seq_length, vocab_size]

if __name__ == '__main__':
    main()
