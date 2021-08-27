nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=5678 src/gpt2_ft.py \
    --train_data ./data/webnlg_challenge_2017/train.jsonl \
    --valid_data ./data/webnlg_challenge_2017/valid.jsonl \
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
    --lora_dim 1 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017 \
    --random_seed 110 \
    --save_interval 10000 > 0822_train_lora_1.out &


python -m torch.distributed.launch --nproc_per_node=4 src/gpt2_beam.py \
    --data ./data/webnlg_challenge_2017/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/webnlg_challenge_2017/model.11265.pt \
    --platform local \
    --lora_dim 1 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/webnlg_challenge_2017 \
    --output_file predict.11265.b10p08.jsonl

    python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/webnlg_challenge_2017/predict.11265.b10p08.jsonl \
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