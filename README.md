# Eedi Competition Solution

This repository contains my solution for the Eedi competition. If you encounter any issues with the setup/code or have questions, please feel free to contact me.

## Repository Structure

- `train_code/` - Code to rebuild models from scratch
- `comp_data/` - Competition data

## Hardware Requirements

The original solution was developed using:
- Ubuntu 22.04 LTS
- 2 x NVIDIA Tesla A100-40G

## Software Requirements

- Python 3.10
- CUDA 11.8
- Additional Python packages are listed in `requirements.txt`

## Training Pipeline

### 1. Data Preparation for Embedding Model

Run the following notebook to generate hard negative data for embedding model training:
```bash
./train_code/FlagEmbedding/embed_data_maker.ipynb
```

### 2. Train the Embedding Model

Execute the training script:
```bash
./train_code/stage1_train.sh
```

### 3. Train the Stage 2 Reranker Model

Example command:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29504 stage2_train.py
```