NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8293 src/gpt2_ft_prune.py --train_data ./data/webnlg_challenge_2017/train.jsonl --valid_data ./data/webnlg_challenge_2017/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint webnlg_rank2_64/model.45065.pt --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 1 --save_interval 1000 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_webnlg_challenge_2017_rank_2_unstructure_0.1 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --num_sparse 128 --random_seed 110 --save_interval 10000 --pruning_ratio 0.1 &


NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8292 src/gpt2_ft_prune.py --train_data ./data/webnlg_challenge_2017/train.jsonl --valid_data ./data/webnlg_challenge_2017/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint webnlg_rank2_64/model.45065.pt --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 2 --save_interval 1000 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_webnlg_challenge_2017_rank_2_unstructure_0.5 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --num_sparse 128 --random_seed 110 --save_interval 10000 --pruning_ratio 0.5 &


NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8294 src/gpt2_ft_prune.py --train_data ./data/webnlg_challenge_2017/train.jsonl --valid_data ./data/webnlg_challenge_2017/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint webnlg_rank2_64/model.45065.pt --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 2 --save_interval 1000 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_webnlg_challenge_2017_rank_2_unstructure_0.3 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --num_sparse 128 --random_seed 110 --save_interval 10000 --pruning_ratio 0.3  > 1.out &  

CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8766 src/gpt2_beam.py \
    --data ./data/webnlg_challenge_2017/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_webnlg_challenge_2017_rank_2_unstructure_0.5/model.18026.pt \
    --platform local \
    --lora_dim 2 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.9 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_webnlg_challenge_2017_rank_2_unstructure_0.5 \
    --output_file predict.18026.b10p08.jsonl &


CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8766 src/gpt2_beam.py \
    --data ./data/webnlg_challenge_2017/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_webnlg_challenge_2017_rank_2_unstructure_0.3/model.18026.pt \
    --platform local \
    --lora_dim 2 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.9 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_webnlg_challenge_2017_rank_2_unstructure_0.3 \
    --output_file predict.18026.b10p08.jsonl &




NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8294 src/gpt2_ft_prune.py --train_data ./data/dart/train.jsonl --valid_data ./data/dart/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint dart_2/model.39160.pt --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 2 --save_interval 1000 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_dart_rank_2_unstructure_0.3 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1 --num_sparse 128 --random_seed 110 --save_interval 10000 --pruning_ratio 0.3  > 1.out & 




python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M_webnlg_challenge_2017_rank_2_unstructure_0.5/predict.18026.b10p08.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg \
    --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_webnlg \
    --output_pred_file eval/GenerationEval/data/hypothesis_webnlg \
    --tokenize --lower

cd ./eval/GenerationEval/
python eval.py \
    -R data/references_webnlg/reference \
    -H data/hypothesis_webnlg \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..



python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M_webnlg_challenge_2017_rank_2_unstructure_0.3/predict.18026.b10p08.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg \
    --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_webnlg \
    --output_pred_file eval/GenerationEval/data/hypothesis_webnlg \
    --tokenize --lower

cd ./eval/GenerationEval/
python eval.py \
    -R data/references_webnlg/reference \
    -H data/hypothesis_webnlg \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..




CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8766 src/gpt2_beam.py \
    --data ./data/dart/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint GPT2_M_dart_rank_2_unstructure_0.3/model.62660.pt \
    --platform local \
    --lora_dim 2 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.9 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir GPT2_M_dart_rank_2_unstructure_0.3 \
    --output_file predict.62660.b10p08.jsonl &