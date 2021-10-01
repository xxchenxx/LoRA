CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8281 src/gpt2_ft.py \
    --train_data ./data/dart/train.jsonl \
    --valid_data ./data/dart/valid.jsonl \
    --train_batch_size 2 \
    --grad_acc 1 \
    --valid_batch_size 1 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 1 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M_bilateral_rank1_seed110_1/dart \
    --random_seed 110 \
    --compress_step 1 \
    --save_interval 50000 > 0903_bilateral_rank1_1.out &

CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8282 src/gpt2_ft.py \
    --train_data ./data/dart/train.jsonl \
    --valid_data ./data/dart/valid.jsonl \
    --train_batch_size 2 \
    --grad_acc 1 \
    --valid_batch_size 1 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 1 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M_bilateral_rank1_seed110_10/dart \
    --random_seed 110 \
    --compress_step 10 \
    --save_interval 50000 > 0903_bilateral_rank1_10.out &