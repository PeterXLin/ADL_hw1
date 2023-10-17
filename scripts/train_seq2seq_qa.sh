output_dir=$1

python ./code/run_seq2seq_qa.py \
  --model_name_or_path uer/t5-v1_1-base-chinese-cluecorpussmall \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir ./checkpoint/qa/$output_dir/ \
#     --max_train_samples 100 \
#   --max_eval_samples 100 \