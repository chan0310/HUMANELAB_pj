from Transformer.Config import Config
import torch
import torch.nn as nn
from torch import optim

from Transformer.Trainer.Tokenizer import TokenizerPlus
from Transformer.Trainer.Trainer import Trainer
from Transformer.Trainer.decoding import greedy_decoding
from Transformer.Config import Config
from Transformer.Model import Transformer

import requests

# ﻿﻿《밑바닥부터 시작하는 딥러닝2》역자 깃허브에서 데이터를 가져옵니다.
url = "https://raw.githubusercontent.com/WegraLee/deep-learning-from-scratch-2/master/dataset/date.txt"
r = requests.get(url)

questions, answers = [], []
for line in r.text.strip().split('\n'):
    idx = line.find('_')
    questions.append(line[:idx].strip())
    answers.append(line[idx:].strip())

tokenizer = TokenizerPlus(char_level=True, filters='')
tokenizer.fit_on_texts(questions)
tokenizer.fit_on_texts(answers)

src = tokenizer.to_tensor(questions)
tgt = tokenizer.to_tensor(answers)

config=Config(len(tokenizer.word_index)+1)
config.n_enc_seq=64
config.n_dec_seq=64
config.d_hidn=64
config.d_ff=128
config.d_head=64
config.n_layer=2
config=Config(len(tokenizer.word_index)+1)
model = Transformer(config)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

trainer = Trainer(model, loss_fn, optimizer,dec_fnc=greedy_decoding,tokenizer=tokenizer)
trainer.train(src, tgt,src[:20],tgt[:20], max_epoch=2)

def translate(input):
    input=tokenizer.to_tensor("august 10, 1994").view(1,-1)
    input = torch.cat([torch.zeros(1, src.size(1) - input.size(1)) , input], dim=1).to(torch.int64)

    output,att=greedy_decoding(model,input,"_")
    print(output)
