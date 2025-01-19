# export WANDB_MODE=disabled
export WANDB_PROJECT="Eedi_Embedder"

I_FOLD=0
train_data="./FlagEmbedding/stage1_data/fold_${I_FOLD}_train_minedHN.jsonl"
output_dir="./models/stage1/qwen_14b_fold_${I_FOLD}_ep3"
model_name="Qwen/Qwen2.5-14B-Instruct-AWQ"

# set large epochs and small batch size for testing
num_train_epochs=5
per_device_train_batch_size=2
gradient_accumulation_steps=8

# set num_gpus to 2 for testing
num_gpus=1

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path $model_name \
    --cache_dir $HF_HUB_CACHE \
    --use_lora True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --save_merged_lora_model False \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len 2048 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Given a math question with options, retrieve the most relevant misconceptions for the incorrect answers.' \
    --query_instruction_format 'Instruct: {}\n\nQuery: {}' \
    --knowledge_distillation False \
"

training_args="\
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./FlagEmbedding/examples/finetune/ds_stage1.json \
    --logging_steps 1 \
    --save_strategy epoch \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    --use_liger_kernel True 
    "

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.decoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
