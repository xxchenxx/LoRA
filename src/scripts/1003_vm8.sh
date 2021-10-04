CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=8283 src/gpt2_ft.py --train_data ./data/dart/train.jsonl --valid_data ./data/dart/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --save_interval 1000 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_dart_rank_2 --lora_dim 2 --lora_alpha 32 --lora_dropout 0.1  --random_seed 110 --save_interval 1000  &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=8284 src/gpt2_ft.py --train_data ./data/dart/train.jsonl --valid_data ./data/dart/valid.jsonl --train_batch_size 2 --grad_acc 1 --valid_batch_size 1 --seq_len 512 --model_card gpt2.md --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin --platform local --clip 0.0 --lr 0.0002 --weight_decay 0.01 --correct_bias --adam_beta2 0.999 --scheduler linear --warmup_step 500 --max_epoch 5 --save_interval 1000 --label_smooth 0.1 --work_dir ./trained_models/GPT2_M_dart_rank_3 --lora_dim 3 --lora_alpha 32 --lora_dropout 0.1  --random_seed 110 --save_interval 1000  &



CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=8766 src/gpt2_beam.py \
    --data ./data/dart/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_dart_rank_3/model.39160.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_dart_rank_3 \
    --output_file predict.39160.b10p08.jsonl &

    CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=8767 src/gpt2_beam.py \
    --data ./data/dart/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_dart_rank_2/model.39160.pt \
    --platform local \
    --lora_dim 2 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_dart_rank_2 \
    --output_file predict.39160.b10p08.jsonl &





python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M_dart_rank_2/predict.39160.b10p08.jsonl \
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