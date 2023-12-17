import torch
from gpt2_model import GPT2Model, GPT2Config

def load_checkpoint(model, checkpoint_path):
    # Load the weights from the checkpoint file
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # If you need to remap keys in the state dict, you can do it here
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load state dict into the model
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully from", checkpoint_path)

def main():
    # Initialize GPT-2 configuration
    config = GPT2Config()
    
    # Initialize GPT-2 model
    model = GPT2Model(config)
    model.eval()  # Set the model to evaluation mode
    
    # Load pre-trained weights (this assumes you have downloaded them)
    # Note: Replace 'path_to_checkpoint.pt' with the actual file path
    checkpoint_path = 'path_to_checkpoint.pt'
    load_checkpoint(model, checkpoint_path)
    
    # Prepare a sample input
    # This should be a batch of token IDs, for demonstration let's assume [50256] * 5
    # 50256 is often the token ID for <|endoftext|> in GPT-2's tokenizer
    input_ids = torch.tensor([[50256] * 5], dtype=torch.long)
    
    # Get predictions (logits) from the model
    with torch.no_grad():
        predictions = model(input_ids)
    
    # Convert predictions to token IDs (greedy approach for simplicity)
    predicted_token_ids = predictions.argmax(-1)
    
    # Print out the predicted token IDs
    print("Predicted token IDs:", predicted_token_ids)

if __name__ == "__main__":
    main()
