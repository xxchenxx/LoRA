CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=4678 src/gpt2_ft.py     --train_data ./data/webnlg_challenge_2017/train.jsonl     --valid_data ./data/webnlg_challenge_2017/valid.jsonl     --train_batch_size 2     --grad_acc 1     --valid_batch_size 1     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.00     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 10000     --lora_dim 4     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017_rank4_bilateral_smooth     --random_seed 110 --compress_step 100 > 0902_webnlg_challenge_2017_rank4_bilateral_smooth.out &

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=4681 src/gpt2_ft.py     --train_data ./data/webnlg_challenge_2017/train.jsonl     --valid_data ./data/webnlg_challenge_2017/valid.jsonl     --train_batch_size 2     --grad_acc 1     --valid_batch_size 1     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.00     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 10000     --lora_dim 3     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017_rank3_bilateral_smooth     --random_seed 110 --compress_step 100 > 0902_webnlg_challenge_2017_rank3_bilateral_smooth.out &

CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=4679 src/gpt2_ft.py     --train_data ./data/webnlg_challenge_2017/train.jsonl     --valid_data ./data/webnlg_challenge_2017/valid.jsonl     --train_batch_size 2     --grad_acc 1     --valid_batch_size 1     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.00     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 10000     --lora_dim 2     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017_rank2_bilateral_smooth     --random_seed 110 --compress_step 100 > 0902_webnlg_challenge_2017_rank2_bilateral_smooth.out &

CUDA_VISIBLE_DEVICES=3 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=4682 src/gpt2_ft.py     --train_data ./data/webnlg_challenge_2017/train.jsonl     --valid_data ./data/webnlg_challenge_2017/valid.jsonl     --train_batch_size 2     --grad_acc 1     --valid_batch_size 1     --seq_len 512     --model_card gpt2.md     --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin     --platform local     --clip 0.0     --lr 0.0002     --weight_decay 0.00     --correct_bias     --adam_beta2 0.999     --scheduler linear     --warmup_step 500     --max_epoch 5     --save_interval 10000     --lora_dim 1     --lora_alpha 32     --lora_dropout 0.1     --label_smooth 0.1     --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017_rank1_bilateral_smooth     --random_seed 110 --compress_step 100 > 0902_webnlg_challenge_2017_rank1_bilateral_smooth.out &