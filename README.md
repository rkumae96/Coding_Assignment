Task 1: GPT-2 Model Implementation & Checkpoints
Overview
This part of the project focuses on the implementation of the GPT-2 small model (125 million parameters) using Python and PyTorch. Key aspects of this task include the development of multi-head self-attention mechanisms, feed-forward networks, and positional encoding, following the original GPT-2 architecture. The implementation strictly avoids the use of pre-built transformer libraries to deepen the understanding of the transformer model's inner workings.

Implementation Details
Model Architecture: The GPT-2 small model comprises 12 layers of transformer blocks, each including a multi-head self-attention mechanism and a point-wise feed-forward network.
Token and Positional Embeddings: We utilize both token and positional embeddings in the model, as per the GPT-2 design. The token embeddings translate input tokens into vectors, while positional embeddings provide context about the position of tokens in the sequence.
Custom Layers: Custom implementations of the transformer layers, including the multi-head self-attention and feed-forward network, are provided.
Normalization: Layer normalization is applied within each transformer block and after the final output layer.

Code Structure
src/gpt2_model.py: Contains the complete implementation of the GPT-2 model, including the transformer block, attention mechanisms, and embedding layers.
src/main.py: Demonstrates how to initialize the model, load checkpoints, and run a sample prediction.

Loading Checkpoints
Pre-trained Checkpoints: The model is compatible with the original GPT-2 125M model checkpoints. Instructions for downloading these checkpoints and loading them into the model are provided.
Checkpoint Loading: The main.py script includes an example of how to load the pre-trained GPT-2 checkpoints into our custom model implementation. This process validates the correct architecture and functioning of the model.

Running a Sample Prediction
The main.py file demonstrates how to run a sample text through the model to generate predictions.
Instructions for preprocessing input text, passing it through the model, and interpreting the output are included in the script.
