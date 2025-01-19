Hello!

Below you can find a outline of how to reproduce my solution for the Eedi competition.
If you run into any trouble with the setup/code or have any questions please contact me.

#ARCHIVE CONTENTS
kaggle_model.tgz          : original kaggle model upload - contains original code, additional training examples, corrected labels, etc
comp_etc                     : contains ancillary information for prediction - clustering of training/test examples
comp_mdl                     : model binaries used in generating solution
comp_preds                   : model predictions
train_code                  : code to rebuild models from scratch
predict_code                : code to generate predictions from model binaries

#HARDWARE: (The following specs were used to create the original solution)
Ubuntu 22.04 LTS
2 x NVIDIA Tesla A100-40G

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.10
CUDA 11.8

#DATA PROCESSING
## 1. Prepare data for embedding model training
run ./train_code/FlagEmbedding/embed_data_maker.ipynb to generate the hard negative data for the embedding model training

## 2. Train the embedding model
run ./train_code/stage1_train.sh to train the embedding model

## 3. Train the stage2 reranker model
sample command:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29504 stage2_train.py
