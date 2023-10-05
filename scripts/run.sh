# perform inference using your trained models and output predictions on test.json

# example
# bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv

context_path=$1
test_path=$2
output_path=$3

python predict.py 