NCCL_P2P_DISABLE=1 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=8184 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M_baseline_rank4_seed110/e2e/model.26289.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M_baseline_rank4/e2e \
    --output_file predict.26289.b10p08.jsonl &


python src/gpt2_decode.py     --vocab ./vocab     --sample_file ./trained_models/GPT2_M_baseline_rank1_seed110/e2e/predict.105155.b10p08.jsonl     --input_file ./data/e2e/test_formatted.jsonl     --output_ref_file e2e_ref.txt     --output_pred_file e2e_pred.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p