
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=4 --master_port=18285 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
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
    --lora_dim 6 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank6_seed110/e2e \
    --random_seed 110 \
    --save_interval 10000 > 0929_bilateral_smooth_rank6.out &

CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=4 --master_port=18286 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
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
    --lora_dim 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank8_seed110/e2e \
    --random_seed 110 \
    --save_interval 10000 > 0929_bilateral_smooth_rank8.out &

CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=4 --master_port=18281 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
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
    --lora_dim 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank16_seed110/e2e \
    --random_seed 110 \
    --save_interval 10000 > 0929_bilateral_smooth_rank16.out &

CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=4 --master_port=18282 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
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
    --lora_dim 32 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank32_seed110/e2e \
    --random_seed 110 \
    --save_interval 10000 > 0929_bilateral_smooth_rank32.out &
