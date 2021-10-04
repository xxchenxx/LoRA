CUDA_VISIBLE_DEVICES=4 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=5678 src/gpt2_ft.py     --train_data ./data/dart/train.jsonl     --valid_data ./data/dart/valid.jsonl     --train_batch_size 2     --grad_acc 1     --valid_batch_size 1     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.00     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 10000     --lora_dim 4     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_M/dart_rank4_bilateral_smooth     --random_seed 110 --compress_step 100 > 0902_dart_rank4_bilateral_smooth.out &

CUDA_VISIBLE_DEVICES=5 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=5681 src/gpt2_ft.py     --train_data ./data/dart/train.jsonl     --valid_data ./data/dart/valid.jsonl     --train_batch_size 2     --grad_acc 1     --valid_batch_size 1     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.00     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 10000     --lora_dim 3     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_M/dart_rank3_bilateral_smooth     --random_seed 110 --compress_step 100 > 0902_dart_rank3_bilateral_smooth.out &

CUDA_VISIBLE_DEVICES=6 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=5679 src/gpt2_ft.py     --train_data ./data/dart/train.jsonl     --valid_data ./data/dart/valid.jsonl     --train_batch_size 2     --grad_acc 1     --valid_batch_size 1     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.00     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 10000     --lora_dim 2     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_M/dart_rank2_bilateral_smooth     --random_seed 110 --compress_step 100 > 0902_dart_rank2_bilateral_smooth.out &

CUDA_VISIBLE_DEVICES=7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=5682 src/gpt2_ft.py     --train_data ./data/dart/train.jsonl     --valid_data ./data/dart/valid.jsonl     --train_batch_size 2     --grad_acc 1     --valid_batch_size 1     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.00     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 10000     --lora_dim 1     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_M/dart_rank1_bilateral_smooth     --random_seed 110 --compress_step 100 > 0902_dart_rank1_bilateral_smooth.out &