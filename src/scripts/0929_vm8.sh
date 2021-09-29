CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=8181 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_bilateral_smooth_rank1_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 1 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank1_seed110/e2e \
    --output_file predict.105155.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=8182 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_bilateral_smooth_rank2_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 2 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank2_seed110/e2e \
    --output_file predict.105155.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=8183 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_bilateral_smooth_rank3_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 3 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_bilateral_smooth_rank3_seed110/e2e \
    --output_file predict.105155.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=8184 src/gpt2_beam.py \
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
    --output_file predict.105155.b10p08.jsonl &
    
