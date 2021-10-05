NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8293 src/gpt2_ft_prune.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_e2e_rank_2/model.105155.pt --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 3 --save_interval 1000 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_2_unstructure --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --num_sparse 128 --random_seed 110 --save_interval 10000 --pruning_ratio 0.5 > 1005_prune_0.5.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8294 src/gpt2_ft_prune.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_e2e_rank_2/model.105155.pt --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 3 --save_interval 1000 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_2_unstructure_0.3 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --num_sparse 128 --random_seed 110 --save_interval 10000 --pruning_ratio 0.3 > 1005_prune_0.3.out &

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8293 src/gpt2_ft_prune.py --train_data ./data/e2e/train.jsonl --valid_data ./data/e2e/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./trained_models/GPT2_M_e2e_rank_2/model.105155.pt --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 1 --save_interval 1000 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_e2e_rank_2_unstructure_0.5 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --num_sparse 128 --random_seed 110 --save_interval 10000 --pruning_ratio 0.5 &



CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=8766 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_e2e_rank_2_unstructure/model.21031.pt \
    --platform local \
    --lora_dim 2 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_e2e_rank_2_unstructure \
    --output_file predict.21031.b10p08.jsonl

CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8767 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_e2e_rank_2_unstructure_0.3/model.42062.pt \
    --platform local \
    --lora_dim 2 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_e2e_rank_2_unstructure_0.3 \
    --output_file predict.21031.b10p08.jsonl &

    CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8768 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_e2e_rank_2_unstructure_0.5/model.42062.pt \
    --platform local \
    --lora_dim 2 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_e2e_rank_2_unstructure_0.5 \
    --output_file predict.42062.b10p08.jsonl &



python src/gpt2_decode.py     --vocab ./vocab     --sample_file ./trained_models/GPT2_M_e2e_rank_2_unstructure/predict.21031.b10p08.jsonl     --input_file ./data/e2e/test_formatted.jsonl     --output_ref_file e2e_ref.txt     --output_pred_file e2e_pred.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p