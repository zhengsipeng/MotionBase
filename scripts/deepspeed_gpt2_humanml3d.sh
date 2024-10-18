export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32'

export EPOCH=300
export DATANAME="humanml3d"
export TRAIN_TARGET="t2m"
export LEARNING_RATE=1e-4
export WEIGHT_DECAY=0.1
export RUN_NAME="gpt2-$DATANAME-$TRAIN_TARGET-$EPOCH-$LEARNING_RATE-$WEIGHT_DECAY"
export LOG_DIR="outputs/$DATANAME/log/$RUN_NAME.log"
export OUTPUT_DIR="outputs/$DATANAME/$RUN_NAME"

export TRAIN_SPLIT_NAME="split/$DATANAME/train.txt"
export VAL_SPLIT_NAME="split/$DATANAME/val.txt"

export VQ_MODEL="pretrained/$DATANAME/VQVAE/net_last.pth"
export VQ_NAME="VQVAE/$DATANAME/VQVAE_all"

export META_DIR="./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
export VAL_META_DIR="./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
export TEXT_MOT_MATCH="./checkpoints/t2m/text_mot_match/model/finest.tar"

deepspeed --master_port=29506 --include localhost:0,1,2,3,4,5,6,7 main.py \
--model_name_or_path /share/pretrain/llm/gpt2-medium \
--output_dir $OUTPUT_DIR \
--num_train_epochs $EPOCH \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "steps" \
--eval_steps 1000 \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 10 \
--preprocessing_num_workers 64 \
--learning_rate $LEARNING_RATE \
--weight_decay $WEIGHT_DECAY \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_dir "./logs/" \
--deepspeed ds_config/stage1.json \
--bf16 \
--gradient_checkpointing 1 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--report_to "wandb" \
--run_name $RUN_NAME \
--logging_steps 1 \
--train_target $TRAIN_TARGET \
--nb_code 512 \
--resume_pth $VQ_MODEL \
--vq_name $VQ_NAME \
--train_split_file $TRAIN_SPLIT_NAME \
--train_meta_dir $META_DIR \
--val_split_file $VAL_SPLIT_NAME \
--val_meta_dir $VAL_META_DIR \
--text_mot_match_path $TEXT_MOT_MATCH \
--min_motion_len 40 \
--max_motion_len 196 \
--dataname $DATANAME \
--down_t 2 \
--depth 3 \
--dilation_growth_rate 3 \
--vq_act relu