CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 6789 src/gpt2_beam.py     --data ./data/dart/test.jsonl     --batch_size 1     --seq_len 512     --eval_len 64     --model_card gpt2.md     --init_checkpoint ./trained_models/GPT2_M/dart_rank1_bilateral_step1000/model.100000.pt     --platform local     --lora_dim 1     --lora_alpha 32     --beam 10     --length_penalty 0.9     --no_repeat_ngram_size 4     --repetition_penalty 1.0     --eos_token_id 628     --work_dir ./trained_models/GPT2_M/dart_rank1_bilateral_step1000     --output_file predict.100000.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 6790 src/gpt2_beam.py     --data ./data/dart/test.jsonl     --batch_size 1     --seq_len 512     --eval_len 64     --model_card gpt2.md     --init_checkpoint ./trained_models/GPT2_M/dart_rank2_bilateral_step1000/model.100000.pt     --platform local     --lora_dim 2     --lora_alpha 32     --beam 10     --length_penalty 0.9     --no_repeat_ngram_size 4     --repetition_penalty 1.0     --eos_token_id 628     --work_dir ./trained_models/GPT2_M/dart_rank2_bilateral_step1000     --output_file predict.100000.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=2 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 6791 src/gpt2_beam.py     --data ./data/dart/test.jsonl     --batch_size 1     --seq_len 512     --eval_len 64     --model_card gpt2.md     --init_checkpoint ./trained_models/GPT2_M/dart_rank3_bilateral_step1000/model.100000.pt     --platform local     --lora_dim 3     --lora_alpha 32     --beam 10     --length_penalty 0.9     --no_repeat_ngram_size 4     --repetition_penalty 1.0     --eos_token_id 628     --work_dir ./trained_models/GPT2_M/dart_rank3_bilateral_step1000     --output_file predict.100000.b10p08.jsonl &

CUDA_VISIBLE_DEVICES=3 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 6793 src/gpt2_beam.py     --data ./data/dart/test.jsonl     --batch_size 1     --seq_len 512     --eval_len 64     --model_card gpt2.md     --init_checkpoint ./trained_models/GPT2_M/dart_rank4_bilateral_step1000/model.100000.pt     --platform local     --lora_dim 4     --lora_alpha 32     --beam 10     --length_penalty 0.9     --no_repeat_ngram_size 4     --repetition_penalty 1.0     --eos_token_id 628     --work_dir ./trained_models/GPT2_M/dart_rank4_bilateral_step1000     --output_file predict.100000.b10p08.jsonl &

