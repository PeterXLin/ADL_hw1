output_dir=$1

python ./code/run_qa_no_trainer.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 5 \
  --learning_rate 3e-5 \
  --num_warmup_steps 0 \
  --max_answer_length 32 \
  --doc_stride 32 \
  --output_dir ./checkpoint/qa/$output_dir/ \
  --checkpointing_steps epoch \
  # --with_tracking \
  # --resume_from_checkpoint ./checkpoint/qa/roberta_final/epoch_3 \
  # --max_train_samples 10 \
  # --max_eval_samples 10 \
  # --model_type bert \
  # --tokenizer_name bert-base-chinese \
 