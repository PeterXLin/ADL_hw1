# Report

## Data processing
所有問題都有答案
max start index => 1254
max answer lenght => 118
### Tokenizer
Q : Describe in detail about the tokenization algorithm you use. You need to explain what it does in your own ways.
A : WordPiece

### Answer Span
Q : How did you convert the answer span start/end position on characters to position on tokens after BERT tokenization?
A: 
tokenized後，context部份可能會因為太長而被截斷，假設原本有n句context，tokenized過後可能變成m句，m>=n。

我們需要確認這m句句子裡的每一句有沒有完整包含答案，若有，則需要紀錄答案在該句tokenized後句子裡的start_token_index及end_token_index。為了達到此目的，我們要對每一句tokenized後的句子做以下這些事。
1. 找出tokenized句子裡context的部份（因為每一句句子的前面都是問題，後面才是被截斷的context）
2. 確認context的部份對應到original sentences的範圍內有沒有完整包含到答案。
3. 若沒有，本句句子的label就是（start_index: 0, end_index: 0）。
3. 若有，則需要利用offset_mapping，取得answer span在該句tokenized sentence中的start_index與end_index並記錄。

以下為簡化版的preprocess function (實際的code還會考慮padding加在左邊還是右邊的問題)
```python
max_length = 512
stride = 128

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]] # 移除前後的空白
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,# 一個句子最多多少token
        truncation="only_second",
        stride=stride,   # 被截斷的部分要重複多少
        return_overflowing_tokens=True,  
        return_offsets_mapping=True, # 每一個token對應的char index
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping") # token 對應到的 char index
    sample_map = inputs.pop("overflow_to_sample_mapping") # map that tells us which sentence each of the results corresponds to
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping): # 每一組句子的offset mapping
        sample_idx = sample_map[i] # 原本對應到的句子的index
        answer = answers[sample_idx] # 對應的答案
        start_char = answer["answer_start"][0] # answer start index (in orginal sentence)
        end_char = answer["answer_start"][0] + len(answer["text"][0]) # answer end index (in original sentence)
        sequence_ids = inputs.sequence_ids(i) # 一次進來可以是一個list, return 他是list中的第幾個
				# 這裡的話就是問題是0，文章是1							

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
		# 回傳的是答案在tokenized後的句子裡的start index & end index
		# e.g. original sentence was divied to 3 subsentences，start_poistion might be
    # e.g. start_positions = [0, 0, 5] end_positions = [0, 0, 10]
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
```

Q : After your model predicts the probability of answer span start/end position, what rules did you apply to determine the final start/end position?
A : 
- 首先，先把tokenized後的句子(稱為feature)與原本的句子配對
- 接著，for 每一個問題
    - for對應的context所encode出的每一個feature
        - 把每一個feature裡被預測最有可能是開頭與最有可能是結尾的n個index找出來
        - for n 個start index和n個 end index 所有可能的配對結果 (n*n種可能)
            - 檢查是否有可能是答案(去除答案out-of-scope、答案長度小於0或大於max_answer_length的)
            - 若有可能是答案，將這一組組合加入可能為答案的集合中，這一組答案的分數為start_logit與end_logit相加
    - 這一個問題的答案為剛才加入的答案候選人裡分數最高的那一個
## Modeling with BERTs and their variants

### Describe
Your model.
The performance of your model.
The loss function you used.
The optimization algorithm (e.g. Adam), learning rate and batch size.
https://github.com/google-research/albert

###  Multiple choice
- loss function: CrossEntropyLoss
- optimizer: AdamW

|Model                  | lr   | batch size | accuracy |
|-----------------------|------|------------|--------- |
|bert-base-chinese      |3e-5  |   8        |  0.9601  | 
|chinese-roberta-wwm-ext|3e-5  |   16       |  0.9661  |
|chinese-roberta-wwm-ext-large||||
|albert-base-chinese    ||||
|chinese-xlnet-base     ||||

### Question answering 

loss function : (max_start + max_end) / 2 = ( max(Softmax(S*Ti)) + max(Softmax(E*Ti)) ) / 2

|Model                  | lr   | batch size | accuracy |
|-----------------------|------|------------|--------- |
|bert-base-chinese      |3e-5  |   8        |          | 
|chinese-roberta-wwm-ext|5e-5  |   16       |    1     |
|chinese-roberta-wwm-ext-large| |||
|xlm-roberta-base       |      |          ||
|albert-base-chinese    |      |||
|chinese-xlnet-base     ||||

### Try another type of pre-trained LMs and describe
Your model.
The performance of your model.
The difference between pre-trained LMs (architecture, pretraining loss, etc.)
For example, BERT -> xlnet, or BERT -> BERT-wwm-ext. You can find these models in the huggingface’s Model Hub.

## Curves (後面那一個)
Plot the learning curve of your span selection (extractive QA) model. Please make sure there are at least 5 data points in each curve.
Learning curve of the loss (0.5%)
Learning curve of the EM (0.5%)

## Pre-trained vs Not Pre-trained
Train a transformer-based model (you can choose either paragraph selection or span selection) from scratch (i.e. without pretrained weights).選一個就好
Describe
The configuration of the model and how do you train this model (e.g., hyper-parameters).
The performance of this model v.s. BERT.
Hint
You can use the same training code, just skip the part where you load the pretrained weights.
The model size configuration for BERT might be too large for this problem, if you find it hard to train a model of the same size, try to reduce model size (e.g. num_layers, hidden_dim, num_heads).

##  End to End model
- Instead of the paragraph selection + span selection pipeline approach, train a end-to-end transformer-based model and describe
Your model.
    - The performance of your model.
    - The loss function you used.
    - The optimization algorithm (e.g. Adam), learning rate and batch size.
- Hint: Try models that can handle long input (e.g., models that have a larger context windows).

xlnet