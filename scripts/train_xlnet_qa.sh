output_dir=$1

python run_qa_beam_search.py \
    --model_name_or_path hfl/chinese-xlnet-base \
    --checkpointing_steps epoch \
    --learning_rate 3e-5 \
    --num_train_epochs 40 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir ./checkpoint/qa/$output_dir/ \
    --per_device_eval_batch_size=16  \
    --per_device_train_batch_size=24   \
    --gradient_accumulation_steps 2 \
    --with_tracking \
    # --max_train_samples 100 \
    # --max_eval_samples 100 \