from datasets import load_dataset
# from accelerate import Accelerator
import argparse
import torch
import os
import json
from itertools import chain
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    default_data_collator,
    AutoTokenizer,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering
)
from accelerate import Accelerator
from utils_qa import postprocess_qa_predictions
import pandas as pd
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--context_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--paragraph_select_model_path",
        type=str,
        default=None,
        help="The directory where model config, tokenizer config, model bin saved. (e.g. checkpoint/paragraph_select/xlnet)",
    )
    parser.add_argument(
        "--question_answering_model_path",
        type=str,
        default=None,
        help="The directory where model config, tokenizer config, model bin saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default = 30,
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    with open(args.context_path) as f:
        context_data = json.load(f)

    qa_dataset = paragraph_select(args, context_data)
    final_prediction = question_answering(args, context_data, qa_dataset)
    df = pd.DataFrame.from_records(final_prediction)
    df.to_csv(args.output_path, index=False)

    return 

def paragraph_select(args, context_data):
    # accelerator = Accelerator()
    # accelerator.wait_for_everyone()
    # prompt = "How the weather today?"
    # candidate2 = "My name is Pekora."
    # candidate1 = "It's sunny outside."
    if args.paragraph_select_model_path != None:
        model_path = os.path.join(os.getcwd(), args.paragraph_select_model_path)
    else:
        model_path = "/nfs/nas-6.1/whlin/ADL/2023_ADL_HW1/checkpoint/for_predict/paragraph/"

    print("paragraph_select_model_path: ", model_path)
    # -------------------------- prepare dataset

    # load raw dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_path})

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load context data

    def turn_to_swag_format(example):
        data = dict()
        # there are four possible context for an question
        for i in range(4):
            data['sentence{}'.format(str(i))] = [context_data[paragraph_ids[i]] for paragraph_ids in example['paragraphs']]

        return data

    ending_names = [f"sentence{i}" for i in range(4)]
    context_name = "question"

    def preprocess_function(examples):
            # queston * 4
            first_sentences = [[context] * 4 for context in examples[context_name]]
            second_sentences = [
                [f"{examples[end][i]}" for end in ending_names] for i in range(len(examples[context_name]))
            ]
            # Flatten out
            first_sentences = list(chain(*first_sentences))
            second_sentences = list(chain(*second_sentences))

            # Tokenize
            tokenized_examples = tokenizer(
                first_sentences,
                second_sentences,
                max_length=args.max_seq_length,
                padding="max_length",
                truncation=True,
            )
            # Un-flatten
            tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
            return tokenized_inputs

    swag_format_datasets = raw_datasets.map(turn_to_swag_format, batched = True, remove_columns=raw_datasets['test'].column_names.remove("question"))
    processed_datasets = swag_format_datasets.map(
        preprocess_function, batched=True, remove_columns=swag_format_datasets['test'].column_names
    )


    test_dataset = processed_datasets['test']
    test_dataloader = DataLoader(test_dataset, collate_fn = default_data_collator, batch_size=args.batch_size)


    # ----------------- prepare model --------------
    # load model
    model = AutoModelForMultipleChoice.from_pretrained(model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu");
    model.to(device)
    # model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # --------------- predict ---------------
    model.eval()
    all_prediction = list()
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        # predictions= accelerator.gather_for_metrics((predictions))
        predictions= predictions.detach().cpu().clone().numpy()
        all_prediction.extend(predictions)

    
    # postprocess
    output_dataset = raw_datasets['test'].add_column("relevant_idx", all_prediction)

    def postprocess(examples):
        return {'relevant': [examples['paragraphs'][idx][label] for idx, label in enumerate(examples['relevant_idx'])]}

    output_dataset = output_dataset.map(postprocess, batched = True, remove_columns=['paragraphs', 'relevant_idx'])
    output_dataset.to_json("./data/paragraph_select_result.json", force_ascii=False) 

    return output_dataset


# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def question_answering(args, context_data, raw_datasets):
    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    if args.question_answering_model_path != None:
        model_path = os.path.join(os.getcwd(), args.question_answering_model_path)
    else:
        model_path = "/nfs/nas-6.1/whlin/ADL/2023_ADL_HW1/checkpoint/for_predict/qa/"
    
    print(model_path)

    # prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_on_right = tokenizer.padding_side == "right"
    question_column_name = "question" 
    context_column_name = "context" 
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def turn_context_id_to_context(example):
        data = dict()
        data['context'] = context_data[example['relevant']]  
        return data
    
    raw_datasets = raw_datasets.map(turn_context_id_to_context, remove_columns=['relevant'])
    
    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=20,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        # if args.version_2_with_negative:
        #     formatted_predictions = [
        #         {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        #     ]
        # else:
        formatted_predictions = [{"id": k, "answer": v} for k, v in predictions.items()]

        return formatted_predictions

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))


    column_names = raw_datasets.column_names
    predict_examples = raw_datasets
    
    with accelerator.main_process_first():
        predict_dataset = predict_examples.map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file= True,
            desc="Running tokenizer on prediction dataset",
        )


    predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
    predict_dataloader = DataLoader(
        predict_dataset_for_model, collate_fn=data_collator, batch_size=args.batch_size
    )

    # ------------------- do predict ----------------------
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model, predict_dataloader = accelerator.prepare(model, predict_dataloader)
    
    all_start_logits = []
    all_end_logits = []
    model.eval()
    for batch in tqdm(predict_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)

    return prediction

if __name__ == '__main__':
    main()