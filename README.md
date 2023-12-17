## Task 1 | GPT-2 Model & Checkpoints (20 Points)

### Objective
The goal for Task 1 is to implement the `GPT2-small` model, which is a smaller version of the Generative Pretrained Transformer 2 (GPT-2) with approximately 125 million parameters. The implementation is done in Python using the PyTorch framework, adhering closely to the original design and architecture as described in the GPT-2 paper.

### Key Implementation Details

- **Token and Positional Embeddings**: The model includes token embeddings that convert input tokens into vectors and positional embeddings that provide sequence position information to the model, following the GPT-2 design philosophy.

- **Transformer Layers**: Each transformer layer in our model is composed of a multi-head self-attention mechanism and a point-wise feed-forward network. These layers are custom-built, without relying on pre-built transformer libraries, to facilitate a deeper understanding of their inner workings.

- **Self-Attention Mechanism**: The multi-head self-attention mechanism allows the model to weigh input token importance differently, depending on the task at hand.

- **Feed-Forward Networks**: After attention aggregation, a feed-forward network is applied to each position's output separately and identically.

### Code Structure

- `src/gpt2_model.py`: This file contains the core GPT-2 model implementation, including the embedding layers and transformer blocks.

- `src/utils.py`: Utility functions for model initialization and other helper functions.

- `checkpoints/`: Directory containing the original GPT-2 125M model checkpoints.

- `src/main.py`: A script to demonstrate loading the model, initializing weights, and running a sample prediction to validate the implementation.

### Validation

To confirm the correct functioning of our model, we performed the following steps:

1. Loaded the original GPT-2 125M model checkpoints into our implementation.
2. Ran a series of sample inputs through the model to generate predictions.
3. Compared the outputs against expected results to ensure consistency with the original GPT-2 outputs.

### References

- The original GPT-2 paper can be found [here](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Andrej Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT) repository provided inspiration for the implementation approach.
- The [makemore](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&feature=shared) series on YouTube offered valuable insights into the transformer model's architecture and functionality.

### Deliverables

The completed Python code for the GPT-2 model is available in the `src` directory, along with scripts to verify its functioning through testing with the original GPT-2 125M checkpoints.
