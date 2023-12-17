# GPT-2 Model Enhancements and Training Infrastructure

This repository showcases a series of enhancements to the GPT-2 model architecture, along with a robust training infrastructure that accommodates various GPU setups.

## Task 1 | GPT-2 Model & Checkpoints (20 Points)

### Objective
We've implemented a scaled-down version of the GPT-2 model, referred to as `GPT2-small`, with the intention to match the original GPT-2's architecture and capabilities. This implementation focuses on understanding and customizing the transformer model within the PyTorch framework.

### Key Implementation Details
- **Embeddings**: The model includes token and positional embeddings, with the latter being crucial for the model to capture the sequence order of the input tokens.
- **Transformer Layers**: Our GPT-2 model has multiple transformer layers, each comprising a multi-head self-attention mechanism and a position-wise feed-forward network, aligning with the standard transformer architecture.
- **Attention Mechanism**: The self-attention component is designed to weigh the influence of different parts of the input sequence, which is central to the transformer's ability to handle various NLP tasks.

### Deliverables
- `src/gpt2_model.py`: Contains the GPT-2 model's class definition, including the structure of the transformer blocks and embedding layers.
- `src/main.py`: A sample script that demonstrates how to load pre-trained GPT-2 checkpoints into our model and perform inference.

## Task 2 | Transformer Architectural Changes (40 Points)

### Modifications
We introduced three significant architectural modifications to the GPT-2 model, each with the potential to enhance the model's performance in different aspects.

#### Rotary Positional Embedding (15 Points)
- Replaced the original positional embeddings with Rotary Positional Embeddings to provide a more nuanced understanding of word positions within the input sequence. This change aimed to improve the model's capacity to generate contextually appropriate text.

#### Group Query Attention (10 Points)
- Introduced Group Query Attention, which processes queries in batches rather than individually. This approach is designed to optimize the computational efficiency of the attention mechanism by sharing computations across queries.

#### Sliding Window Attention (15 Points)
- Implemented Sliding Window Attention to limit the model's focus to a fixed-size window of surrounding tokens. This modification is expected to enhance the model's ability to process longer texts by emphasizing local context.

### Evaluation Criteria
We assessed the impact of each modification on the model's size, computational demands, and overall performance. Points were awarded based on the successful integration and functionality of these changes.

## Task 3 | Training Loop Implementation (40 Points)

### Features

#### Single GPU Training Loop
- Developed a basic training loop (`train_single_gpu.py`) for models that can be accommodated on a single GPU. This implementation is intended to serve as a baseline and starting point for more complex training setups.

#### Distributed Data Parallel (DDP)
- Expanded the training loop to a DDP-compatible version (`train_ddp.py`), allowing the model to be trained on multiple GPUs in parallel. This setup is particularly useful for accelerating the training process or handling larger models that exceed the memory capacity of a single GPU.

#### Fully Sharded Data Parallel (FSDP)
- For training at an even larger scale, we integrated FSDP into our training loop (`train_fsdp.py`). FSDP shards the model parameters, gradients, and optimizer states across all available GPUs, drastically reducing the memory requirements per GPU and enabling the training of significantly larger models.

### Deliverables
Each training setup is encapsulated in a distinct Python script, with comprehensive documentation detailing the code's adaptation to each GPU environment.

### Evaluation Scheme
The feature implementations were scored on their functionality and GPU compatibility, with the FSDP setup being weighted more due to its complexity and the substantial benefits it offers for large-scale model training.

## Setup and Usage
To run these training loops, ensure that the environment is properly set up with the necessary libraries, distributed computing configurations, and datasets. The README in the repository provides detailed instructions on how to execute each training loop for different GPU environments.
