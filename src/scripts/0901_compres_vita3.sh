python -m torch.distributed.launch --nproc_per_node=1 --master_port=56781 src/gpt2_compress.py     --train_data ./data/e2e/train.jsonl     --valid_data ./data/e2e/valid.jsonl     --train_batch_size 2     --grad_acc 1     --valid_batch_size 1     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.01     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 1000     --lora_dim 1     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_M_bilateral_rank1_1500/e2e     --random_seed 1     --save_interval 10000 --compress_step 1500 