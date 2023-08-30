import torch.nn as nn
import torch
from Seq2SeqModel.Seq2SeqModel import Seq2SeqModel
import pandas as pd
from transformers import AutoTokenizer
import torch.nn.init as init

# CSV 파일에서 데이터 불러오기
df_train = pd.read_csv("Datatset3/train.csv")
df_val = pd.read_csv("Datatset3/val.csv")
tokenizer= AutoTokenizer.from_pretrained("Datatset3/tokenizer")

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

device = torch.device('cpu')

tokenizer= AutoTokenizer.from_pretrained("Datatset/tokenizer")
model=None
class_weights = torch.ones(tokenizer.vocab_size+1).to(device=device)
class_weights[tokenizer.pad_token_id]=0.1
class_weights[tokenizer.pad_token_id-1]=0.1
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01,weight=class_weights)
optimizer=None
batchsize=64
lr=0.005
print(len(tokenizer.get_vocab()),len(X_train),len(y_train))
print("Set")
module=Seq2SeqModel( tokenizer,loss_fn,lr,
                    X_train,y_train,X_val,y_val,batch_size=batchsize,device=torch.device("cpu"))
print(module.config.n_dec_seq)

module.train_main(4,100)
print("train")
module.load_model("saved16.pth")
