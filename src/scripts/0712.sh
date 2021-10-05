CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=8484 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 2 \
    --grad_acc 1 \
    --valid_batch_size 1 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 0 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M_original/e2e \
    --random_seed 110 > original.out &


python -m torch.distributed.launch --nproc_per_node=8 src/gpt2_beam.py      --data ./data/e2e/test.jsonl      --batch_size 1      --seq_len 512      --eval_len 64      --model_card gpt2.md      --init_checkpoint ./trained_models/GPT2_M/e2e/model.13145.pt      --platform local      --lora_dim 4      --lora_alpha 32      --beam 10      --length_penalty 0.8      --no_repeat_ngram_size 4      --repetition_penalty 1.0      --eos_token_id 628      --work_dir ./trained_models/GPT2_M/e2e      --output_file predict.13145.b10p08.jsonl

python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.13145.b10p08.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
