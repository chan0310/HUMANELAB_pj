import torch.nn as nn
import torch
from Seq2SeqModel.Seq2SeqModel import Seq2SeqModel
import pandas as pd
from transformers import AutoTokenizer
import torch.nn.init as init

# CSV 파일에서 데이터 불러오기
df_train = pd.read_csv("Datatset2/train.csv")
df_val = pd.read_csv("Datatset2/val.csv")
tokenizer= AutoTokenizer.from_pretrained("Datatset2/tokenizer")

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

device = torch.device('mps')

print(f'Using {device}')

tokenizer= AutoTokenizer.from_pretrained("Datatset/tokenizer")
model=None
loss_fn = nn.CrossEntropyLoss()
optimizer=None
batchsize=64
print("Set")
module=Seq2SeqModel( model,tokenizer,optimizer,loss_fn,
                    X_train,y_train,X_val,y_val,batch_size=batchsize)

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.Linear):
        init.kaiming_uniform_(submodule.weight, mode='fan_in', nonlinearity='relu')
        if submodule.bias is not None:
            init.constant_(submodule.bias, 0.0)

module.model.apply(weight_init_xavier_uniform)


print("train")
module.train_main(20) 
