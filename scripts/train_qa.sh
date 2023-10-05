output_dir=$1

python run_qa_no_trainer.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 40 \
  --learning_rate 3e-5 \
  --output_dir ./checkpoint/qa/$output_dir/ \
  --checkpointing_steps epoch \
  --doc_stride 128 \
  --with_tracking \
  --resume_from_checkpoint ./checkpoint/qa/roberta/epoch_19 \
  # --max_train_samples 100 \
  # --max_eval_samples 100 \