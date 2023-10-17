output_dir=$1

python ./code/run_qa_no_trainer.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 6 \
  --learning_rate 5e-5 \
  --num_warmup_steps 0 \
  --max_answer_length 32 \
  --doc_stride 32 \
  --output_dir ./checkpoint/qa/$output_dir/ \
  --checkpointing_steps epoch 
  # --max_train_samples 10 \
  # --max_eval_samples 10 \
    # --with_tracking \
  # --model_type bert \
  # --tokenizer_name bert-base-chinese \
  # --resume_from_checkpoint ./checkpoint/qa/roberta/epoch_19 \
