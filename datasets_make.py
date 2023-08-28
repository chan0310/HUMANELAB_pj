from datasets import load_dataset, load_metric

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")

from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
tokenizer.add_special_tokens({"bos_token": "<bos>"})

en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence)
with tokenizer.as_target_tokenizer():
    targets = tokenizer(fr_sentence)

wrong_targets = tokenizer(fr_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(targets["input_ids"]))

max_input_length = 128
max_target_length = 128


def preprocess_function(examples):
    inputs = ["<bos> "+ex["en"] for ex in examples["translation"]]
    targets = ["<bos> "+ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # 타겟을 위한 토크나이저 셋업
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

train_dataset = tokenized_datasets["train"]
validation_dataset = tokenized_datasets["validation"]

X_train = train_dataset["input_ids"][0:15000]
y_train = train_dataset["labels"][0:15000]

X_val = validation_dataset["input_ids"][:2000]
y_val = validation_dataset["labels"][:2000]

import pandas as pd
df_train = pd.DataFrame({
    "X_train": X_train,
    "y_train": y_train,
})

df_val = pd.DataFrame({
    "X_val": X_val,
    "y_val": y_val,
})

tokenizer.save_pretrained("Datatset2/tokenizer")

df_train.to_csv("Datatset2/train.csv",index=False)
df_val.to_csv("Datatset2/val.csv",index=False)
