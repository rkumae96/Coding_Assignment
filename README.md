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

# Task 2 | Transformer Architectural Changes (40 Points)

---

Enhance the GPT-2 model by implementing and assessing transformative architectural modifications. This task challenges you to integrate advanced embedding and attention mechanisms to potentially improve the model's performance.

## Modifications

### Rotary Positional Embedding (15 Points)
- Swap the standard positional embeddings with Rotary embeddings, as described by [Su et. al. RoFormer](https://arxiv.org/pdf/2104.09864.pdf).
- Assess the impact on the model's ability to understand and generate contextually relevant text.

### Group Query Attention (10 Points)
- Apply the Group Query Attention mechanism, taking inspiration from [Ainslie et. al. GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/pdf/2305.13245v2.pdf).
- Evaluate how this modification influences the model's attention and processing capabilities.

### Sliding Window Attention (15 Points)
- Integrate Sliding Window Attention as detailed in [Beltagy et. al. Longformer](https://arxiv.org/pdf/2004.05150v2.pdf).
- Determine its effects on the model's efficiency and accuracy in handling longer sequences.

## Deliverables

Submit Python code that incorporates one, two, or all the proposed changes. With each integration, critically analyze:

- The alteration in model size and computational demands.
- Any potential challenges introduced with the changes.
- Improvements or regressions in model performance and applicability.

Points will be awarded based on the successful implementation of the features.

## Evaluation Criteria

The feature implementations will be scored as follows:

- **Rotary Positional Embedding**: 15 points
- **Group Query Attention**: 10 points
- **Sliding Window Attention**: 15 points

Your work will be evaluated on the accuracy of the implementation, the depth of your analysis, and your ability to articulate the effects of these changes on the model's performance.

