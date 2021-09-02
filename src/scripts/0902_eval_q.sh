CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8761 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/webnlg_challenge_2017_rank4_bilateral_step1000/model.45065.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.9 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017_rank4_bilateral_step1000 \
    --output_file predict.45065.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8762 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/webnlg_challenge_2017_rank3_bilateral_step1000/model.45065.pt \
    --platform local \
    --lora_dim 3 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.9 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017_rank3_bilateral_step1000 \
    --output_file predict.45065.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8763 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/webnlg_challenge_2017_rank2_bilateral_step1000/model.45065.pt \
    --platform local \
    --lora_dim 2 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.9 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017_rank2_bilateral_step1000 \
    --output_file predict.45065.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=3 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8764 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/webnlg_challenge_2017_rank1_bilateral_step1000/model.45065.pt \
    --platform local \
    --lora_dim 1 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.9 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017_rank1_bilateral_step1000 \
    --output_file predict.45065.b10p08.jsonl &

