import torch
from gpt2_model import GPT2, GPT2Config

def load_checkpoint(filepath, model):
    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

def main():
    # Initialize configuration and model
    config = GPT2Config()
    model = GPT2(config)

    # Load the original GPT-2 125M model checkpoints
    checkpoint_path = "path_to_gpt2_checkpoint.pth"  # Update this path to your checkpoint file
    load_checkpoint(checkpoint_path, model)

    # Prepare input data manually (e.g., using a dictionary that maps tokens to IDs)
    # For demonstration purposes, let's assume the following tokens and IDs:
    tokenizer = {'The': 1212, 'quick': 12345, 'brown': 54321, ... , '<|endoftext|>': 50256}

    # Encode a sample input text
    input_text = "The quick brown fox"
    input_ids = [tokenizer.get(word, tokenizer['<|endoftext|>']) for word in input_text.split()]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension

    # Set model to eval mode and run prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]  # Get the last hidden states

    # Get the predicted token ID (greedy decoding for simplicity)
    predicted_token_id = torch.argmax(predictions[:, -1, :], dim=-1).item()
    
    # Assuming you have a reverse tokenizer (ID to token)
    reverse_tokenizer = {1212: 'The', 12345: 'quick', 54321: 'brown', ... , 50256: '<|endoftext|>'}
    predicted_token = reverse_tokenizer.get(predicted_token_id, '<|unknown|>')

    print(f"Input text: {input_text}")
    print(f"Predicted token: {predicted_token}")

if __name__ == '__main__':
    main()
