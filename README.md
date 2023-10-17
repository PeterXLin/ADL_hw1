# 2023 ADL HW1

## How to use it 
step-by-step instruction on how to train this model with provided codes/scripts

**if there is CUDA out of memory error, please try to Reduce batch size. (modify in each scripts)**
### Train

- train paragraph select model
```shell
# run this commend in /2023_ADL_HW1
bash ./scripts/train_paragraph.sh /which/folder/in/checkpint/paragraph_select/you/want/to/save/the/model
# for example 
bash ./scripts/train_paragraph.sh xlnet_mid # in this case, model checkpiont will be saved in /checkpoint/paragraph_select/xlnet_mid
```

- trian span selection model

```shell
# run this commend in /2023_ADL_HW1
bash ./scripts/train_qa.sh /which/folder/in/checkpint/qa/you/want/to/save/the/model
# for example 
bash ./scripts/train_qa.sh roberta_large # in this case, model checkpiont will be saved in /checkpoint/qa/roberta_large
```

### Inference
```shell
# run this commend in /2023_ADL_HW1
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
# for example
bash ./run.sh ./data/context.json ./data/test.json ./data/test_data_predict.csv
```



