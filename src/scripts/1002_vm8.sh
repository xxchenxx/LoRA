CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8761 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_bilateral_smooth_rank32_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 32 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank32_seed110/e2e \
    --output_file predict.26289.b10p08.jsonl &


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8762 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_bilateral_smooth_rank16_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 16 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank16_seed110/e2e \
    --output_file predict.26289.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8763 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_bilateral_smooth_rank8_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 8 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank8_seed110/e2e \
    --output_file predict.26289.b10p08.jsonl

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8764 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_bilateral_smooth_rank6_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 6 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank6_seed110/e2e \
    --output_file predict.26289.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_bilateral_smooth_rank4_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank4_seed110/e2e \
    --output_file predict.26289.b10p08.jsonl &

git checkout main

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8766 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_rank4_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_rank4_seed110/e2e \
    --output_file predict.26289.b10p08.jsonl

