output_dir=$1

python ./code/run_swag_no_trainer.py \
  --model_name_or_path hfl/chinese-xlnet-mid \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir ./checkpoint/paragraph_select/$output_dir/ \
  --checkpointing_steps epoch \
  --gradient_accumulation_steps 2 \
  --train_file "./data/train.json" \
  --validation_file "./data/valid.json" \
  # --with_tracking \
  #  --resume_from_checkpoint ./checkpoint/paragraph_select/bert_chinese/epoch_3 \
  # "--resume_from_checkpoint",