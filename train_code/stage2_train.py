import os
os.environ["WANDB_PROJECT"] = "eedi_stage2"
import re
import json
import random
from tqdm import tqdm, trange
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.model_selection import KFold
from peft import PeftModel, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from liger_kernel.transformers import apply_liger_kernel_to_qwen2

apply_liger_kernel_to_qwen2()

I_FOLD = 0
MODEL_PATH = "Qwen/Qwen2.5-32B-Instruct-AWQ"
SEED = 777

PROMPT_TEMPLATE = """Task:
As a Mathematics teacher, your goal is to analyze a student's incorrect answer of a mathematics question, identify their fundamental conceptual misunderstanding, and select the single most appropriate misconception number from the given misconception options.

Here is a mathematics question about {construct}({subject}).
Question:
{question}

Correct Answer:
{correct_content}

Incorrect Answer:
{wrong_content}

Carefully analyze the incorrect answer and select a single most appropriate misconception number from the given misconceptions.

Here are the retrieved misconceptions:
{misconception_list_txt}

Only output the code of the selected misconception.
Don't output any other words.
"""

MIS_CODES = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
]

class Stage2TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data, mis_dict, tokenizer, do_tokenize=True):
        self.data = data
        self.mis_dict = mis_dict
        self.tokenizer = tokenizer
        self.do_tokenize = do_tokenize
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        
        gt_mis_text = d['gt_mis_text']
        n_neg = random.randint(20, 34)
        sampled_negs = random.sample(d['hard_neg'][:200], n_neg)
        assert not d['gt_mis_id'] in sampled_negs
        
        cand_mis_ids = sampled_negs + [d['gt_mis_id']]
        
        candidate_mis_txts = [self.mis_dict[m] for m in cand_mis_ids]
        random.shuffle(candidate_mis_txts)
        
        gt_code = MIS_CODES[candidate_mis_txts.index(gt_mis_text)]
        cand_list_txt = '\n'.join([f'{mc}. {mt}' for mc, mt in zip(MIS_CODES, candidate_mis_txts)])
        
        prompt_text = PROMPT_TEMPLATE.format(
            subject=d['subject'],
            construct=d['construct'],
            question=d['question'],
            choice_a=d['choice_a'],
            choice_b=d['choice_b'],
            choice_c=d['choice_c'],
            choice_d=d['choice_d'],
            correct_choice=d['correct_choice'],
            correct_content=d['correct_content'],
            wrong_choice=d['wrong_choice'],
            wrong_content=d['wrong_content'],
            misconception_list_txt=cand_list_txt,
        )
        
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": gt_code},
        ]
        
        if self.do_tokenize:
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=True)
            return {'input_ids': tokens}
        else:
            return messages

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    mis_map = pd.read_csv('../comp_data/misconception_mapping.csv')
    mis_dict = dict(zip(mis_map['MisconceptionId'], mis_map['MisconceptionName'].str.strip()))

    # Load hard negative data
    with open('../comp_data/stage2_train_data_all_folds.json') as f:
        stage2_all_data = json.load(f)
        
    kfold = KFold(n_splits=10, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(stage2_all_data)):
        if fold != I_FOLD:
            continue
        stage2_train_data = [stage2_all_data[i] for i in train_idx]
        # stage2_val_data = [stage2_all_data[i] for i in val_idx]

    # Create dataset
    stage2_train_dataset = Stage2TrainDataset(
        stage2_train_data,
        mis_dict,
        tokenizer,
        do_tokenize=True
    )
    
    # Load model
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device = 'cuda' if local_rank == -1 else f'cuda:{local_rank}'
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation='flash_attention_2',
        trust_remote_code=True,
    )

    # Training arguments
    training_args = SFTConfig(
        output_dir="./models/stage2/qwen_32b_ep5_fold_${I_FOLD}",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        
        fp16=True,
        optim="adamw_8bit",
        learning_rate=1.0e-4,
        warmup_ratio=0.06,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        
        logging_steps=2,
        eval_steps=50,
        save_strategy='epoch',
        
        # use_liger_kernel=True,
        max_seq_length=4096,
        dataset_num_proc=8,
        
        report_to=['wandb'],
        
        ddp_find_unused_parameters=False
    )

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # LoRA config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=stage2_train_dataset,
        args=training_args,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_seq_length=4096,
        data_collator=collator,
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    main() 