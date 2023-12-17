# GPT-2 Model and Training Implementations

This repository contains custom implementations for modifying the GPT-2 model architecture and developing a flexible training loop that supports single GPU, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP) setups.

## Task 1 | GPT-2 Model & Checkpoints (20 Points)

### Objective
Implemented a `GPT2-small` model from scratch, closely following the original GPT-2 architecture. The model was adapted to work with pre-trained weights and tested for correct functionality.

### Key Implementation Details
- Custom token and positional embeddings, transformer layers, and a self-attention mechanism without using pre-built libraries.
- The model was verified by loading original GPT-2 125M checkpoints and running sample predictions.

### Deliverables
- `gpt2_model.py`: Core GPT-2 model implementation.
- `main.py`: Script for model inference and validation.
- Checkpoint directory for storing pre-trained weights.

## Task 2 | Transformer Architectural Changes (40 Points)

### Modifications
Implemented the following architectural enhancements to the GPT-2 model:

#### Rotary Positional Embedding (15 Points)
- Integrated Rotary Positional Embedding, improving the model's contextual understanding.

#### Group Query Attention (10 Points)
- Added Group Query Attention to process queries in batches, potentially enhancing efficiency.

#### Sliding Window Attention (15 Points)
- Employed Sliding Window Attention to better handle longer sequences.

### Evaluation Criteria
Each feature was evaluated based on its successful implementation and impact on the model's performance and computational requirements.

## Task 3 | Training Loop Implementation (40 Points)

### Features

#### Single GPU Training Loop
- Developed a straightforward training loop for single GPU setups (`train_single_gpu.py`).

#### Distributed Data Parallel (DDP)
- Extended the training loop to support multi-GPU training using DDP (`train_ddp.py`).

#### Fully Sharded Data Parallel (FSDP)
- Implemented an FSDP training loop, enabling efficient training of large models by sharding parameters across GPUs (`train_fsdp.py`).

### Deliverables
Three Python scripts for each training setup, complete with documentation on how to execute the training loop in different environments.

### Evaluation Scheme
Points were awarded based on the functionality and compatibility of the training loop with each GPU setup:

- **Single GPU**: 10 points
- **DDP**: 10 points
- **FSDP**: 20 points

## Setup and Usage
To utilize these implementations, ensure your environment is configured correctly, including any necessary library installations and distributed computing setups. Refer to individual scripts for detailed usage instructions.
