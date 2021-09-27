CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8181 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_low_rank/e2e_rank_1/model.105155.pt --platform local --lora_dim 1 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_low_rank/e2e_rank_1 --output_file predict.105155.b10p08.jsonl & 

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8182 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_low_rank/e2e_rank_2/model.105155.pt --platform local --lora_dim 2 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_low_rank/e2e_rank_2 --output_file predict.105155.b10p08.jsonl & 

CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8183 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_low_rank/e2e_rank_3/model.105155.pt --platform local --lora_dim 3 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_low_rank/e2e_rank_3 --output_file predict.105155.b10p08.jsonl & 

CUDA_VISIBLE_DEVICES=7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8184 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_low_rank/e2e_rank_4/model.105155.pt --platform local --lora_dim 4 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_low_rank/e2e_rank_4 --output_file predict.105155.b10p08.jsonl & 

CUDA_VISIBLE_DEVICES=4 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8185 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_low_rank/e2e_rank_6/model.105155.pt --platform local --lora_dim 6 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_low_rank/e2e_rank_6 --output_file predict.105155.b10p08.jsonl & 

CUDA_VISIBLE_DEVICES=5 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8186 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_low_rank/e2e_rank_8/model.105155.pt --platform local --lora_dim 8 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_low_rank/e2e_rank_8 --output_file predict.105155.b10p08.jsonl & 

CUDA_VISIBLE_DEVICES=6 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8187 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_low_rank/e2e_rank_16/model.105155.pt --platform local --lora_dim 16 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_low_rank/e2e_rank_16 --output_file predict.105155.b10p08.jsonl & 

CUDA_VISIBLE_DEVICES=7 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8188 src/gpt2_beam.py --data ./data/e2e/test.jsonl --batch_size 1 --seq_len 512 --eval_len 64 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_low_rank/e2e_rank_32/model.105155.pt --platform local --lora_dim 32 --lora_alpha 32 --beam 10 --length_penalty 0.8 --no_repeat_ngram_size 4 --repetition_penalty 1.0 --eos_token_id 628 --work_dir ./trained_models/GPT2_M_low_rank/e2e_rank_32 --output_file model.105155.pt &