from Transformer.Config import Config
from Transformer.Model import Transformer
import torch.nn as nn
from torch import optim
from Seq2SeqModel.Seq2SeqModel import Seq2SeqModel
import pandas as pd
from transformers import AutoTokenizer

# CSV 파일에서 데이터 불러오기
df_train = pd.read_csv("Datatset/train.csv")
df_val = pd.read_csv("Datatset/val.csv")
tokenizer= AutoTokenizer.from_pretrained("Datatset/tokenizer")

def str_to_list(input_string):
    cleaned_string = input_string.strip("[]").replace(" ", "")

    number_strings = cleaned_string.split(",")
    number_list = [int(num) for num in number_strings]
    return number_list

# 데이터 추출
X_train = df_train["X_train"].values.tolist()
X_train=[str_to_list(i) for i in X_train]
y_train = df_train["y_train"].values.tolist()
y_train=[str_to_list(i) for i in y_train]


X_val = df_val["X_val"].values.tolist()
X_val=[str_to_list(i) for i in X_val]
y_val = df_val["y_val"].values.tolist()
y_val=[str_to_list(i) for i in y_val]

tokenizer= AutoTokenizer.from_pretrained("Datatset/tokenizer")
config=Config(3)
model=Transformer(config)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
batchsize=64
print("Set")
module=Seq2SeqModel( tokenizer,optimizer,loss_fn,
                    X_train,y_train,X_val,y_val)
print("train")
module.train_main(5,4) 
