python src/gpt2_decode.py          --vocab ./vocab          --sample_file ./trained_models/GPT2_M/dart_rank4/predict.100000.b10p08.jsonl          --input_file ./data/dart/test_formatted.jsonl          --ref_type dart          --ref_num 6          --output_ref_file eval/GenerationEval/data/references_dart          --output_pred_file eval/GenerationEval/data/hypothesis_dart          --tokenize --lower

cd ./eval/GenerationEval/
python eval.py      -R data/references_dart/reference      -H data/hypothesis_dart      -nr 6      -m bleu,meteor,ter 
cd ../..