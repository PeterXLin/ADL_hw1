# perform inference using your trained models and output predictions on test.json
# if there is CUDA out of memory error, please try to Reduce batch size

context_path=$1
test_path=$2
output_path=$3

python ./code/two_stage_qa_predict.py \
    --context_path $context_path \
    --test_path $test_path \
    --output_path $output_path \
    --paragraph_select_model_path "checkpoint_for_predict/paragraph_select" \
    --question_answering_model_path "checkpoint_for_predict/qa" \
    --batch_size 4 \
    --max_answer_length 32 \
