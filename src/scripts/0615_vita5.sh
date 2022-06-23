NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8283 src/gpt2_ft.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_4_1 --lora_dim 4 --lora_alpha 32 --lora_dropout 0.1 --random_seed 110 --save_interval 10000 > 1.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8284 src/gpt2_ft.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_4_2 --lora_dim 4 --lora_alpha 32 --lora_dropout 0.1 --random_seed 109 --save_interval 10000 > 2.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8285 src/gpt2_ft.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_2_1 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --random_seed 110 --save_interval 10000 > 3.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=3 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8286 src/gpt2_ft.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_2_2 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --random_seed 109 --save_interval 10000 > 4.out &





NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8287 src/gpt2_ft.py --train_data ./data/dart/train.jsonl --valid_data ./data/dart/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_dart_rank_4_1 --lora_dim 4 --lora_alpha 32 --lora_dropout 0.1 --random_seed 110 --save_interval 10000 > 5.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=5 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8288 src/gpt2_ft.py --train_data ./data/dart/train.jsonl --valid_data ./data/dart/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_dart_rank_4_2 --lora_dim 4 --lora_alpha 32 --lora_dropout 0.1 --random_seed 109 --save_interval 10000 > 6.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=6 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8289 src/gpt2_ft.py --train_data ./data/dart/train.jsonl --valid_data ./data/dart/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_dart_rank_2_1 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --random_seed 110 --save_interval 10000 > 7.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8280 src/gpt2_ft.py --train_data ./data/dart/train.jsonl --valid_data ./data/dart/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_dart_rank_2_2 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --random_seed 109 --save_interval 10000 > 8.out &






CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8283 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_e2e_rank_4_1/model.105155.pt --platform local --lora_dim 4 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_e2e_rank_4_1 --output_file predict.105155.b10p08.jsonl > 1.out &

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8284 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_e2e_rank_4_2/model.105155.pt --platform local --lora_dim 4 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_e2e_rank_4_2 --output_file predict.105155.b10p08.jsonl > 2.out &

CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8285 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_e2e_rank_2_1/model.105155.pt --platform local --lora_dim 2 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_e2e_rank_2_1 --output_file predict.105155.b10p08.jsonl > 3.out &

CUDA_VISIBLE_DEVICES=3 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8286 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_e2e_rank_2_2/model.105155.pt --platform local --lora_dim 2 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_e2e_rank_2_2 --output_file predict.105155.b10p08.jsonl > 4.out &


python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M_e2e_rank_2_2/predict.105155.b10p08.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p





NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8283 src/gpt2_ft.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_4_1 --lora_dim 4 --lora_alpha 32 --lora_dropout 0.1 --random_seed 111 --save_interval 10000 > 1.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8284 src/gpt2_ft.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_4_2 --lora_dim 4 --lora_alpha 32 --lora_dropout 0.1 --random_seed 112 --save_interval 10000 > 2.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8285 src/gpt2_ft.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_2_1 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --random_seed 111 --save_interval 10000 > 3.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=3 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8286 src/gpt2_ft.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_2_2 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --random_seed 112 --save_interval 10000 > 4.out &




CUDA_VISIBLE_DEVICES=4 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8287 src/gpt2_beam.py --data ./data/dart/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_dart_rank_4_1/model.156650.pt --platform local --lora_dim 4 --lora_alpha 32 --beam 10 --length_penalty 0.9 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_dart_rank_4_1 --output_file predict.156650.b10p09.jsonl > 1.out &

CUDA_VISIBLE_DEVICES=5 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8288 src/gpt2_beam.py --data ./data/dart/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_dart_rank_4_2/model.156650.pt --platform local --lora_dim 4 --lora_alpha 32 --beam 10 --length_penalty 0.9 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_dart_rank_4_2 --output_file predict.156650.b10p09.jsonl > 2.out &

CUDA_VISIBLE_DEVICES=6 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8289 src/gpt2_beam.py --data ./data/dart/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_dart_rank_2_1/model.156650.pt --platform local --lora_dim 2 --lora_alpha 32 --beam 10 --length_penalty 0.9 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_dart_rank_2_1 --output_file predict.156650.b10p09.jsonl > 3.out &

CUDA_VISIBLE_DEVICES=7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8290 src/gpt2_beam.py --data ./data/dart/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_dart_rank_2_2/model.156650.pt --platform local --lora_dim 2 --lora_alpha 32 --beam 10 --length_penalty 0.9 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_dart_rank_2_2 --output_file predict.156650.b10p09.jsonl > 4.out &


python src/gpt2_decode.py \
        --vocab ./vocab \
        --sample_file ./trained_models/GPT2_M_dart_rank_2_1/predict.156650.b10p09.jsonl \
        --input_file ./data/dart/test_formatted.jsonl \
        --ref_type dart \
        --ref_num 6 \
        --output_ref_file eval/GenerationEval/data/references_dart \
        --output_pred_file eval/GenerationEval/data/hypothesis_dart \
        --tokenize --lower

cd ./eval/GenerationEval/
python eval.py \
    -R data/references_dart/reference \
    -H data/hypothesis_dart \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..