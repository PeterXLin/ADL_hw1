# perform inference using your trained models and output predictions on test.json

# example
# bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv

# context_path=$1
# test_path=$2
# output_path=$3

python ./code/select_paragraph_predict.py --context_path "./data/context.json" --test_path "./data/test.json" --output_path "./data/first_predict.csv"
