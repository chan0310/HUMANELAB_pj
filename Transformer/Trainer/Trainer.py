
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys

def overwrite_previous_line(text):
    sys.stdout.write('\r' + text)
    sys.stdout.flush()

def greedy_decoding(trm, src, detokenizer,num,start_token="_"):
    start_token=detokenizer.pad_token_id
    start_token=torch.tensor([start_token])
    N = src.size(0)
    preds =torch.tensor( [start_token]*N).view(-1,1)
    with torch.no_grad():
        for _ in range(num):
            y_pred,a,b,c = trm(src,preds)
            t_pred = torch.argmax(y_pred[:,-1,:], axis=-1, keepdims=True)
            preds = torch.cat([preds, t_pred], axis=1)            
        return preds
    
class Trainer:
    def __init__ (self, model, loss_fn, optimizer, dec_fnc, tokenizer, pad_id=0, start_token='_'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.pad_id = pad_id
        self.start_token = start_token
        self.losses = []
        self.current_epoch = 0
        self.dec_fnc=greedy_decoding
        self.tokenizer=tokenizer
        
    def train(self, src, tgt, val_src, val_tgt, max_epoch=1, batch_size=64):
        X1_train = src
        X2_train = tgt[:, :-1]
        y_train = tgt[:, 1:]
        ds = TensorDataset(X1_train, X2_train, y_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        for epoch in range(1, max_epoch+1):
            self.model.train()
            running_loss = 0
            self.current_epoch += 1
            print("EPOCH: %s :: " %self.current_epoch, end='')
            for i, (x1, x2, yy) in enumerate(loader):
                y_pred,ea,de,eda = self.model(x1, x2)
                loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1)), yy.view(-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                text='(Epoch{:4d}/{})   Batch:{}/{}   Cost:{:.6f}'.format(epoch, max_epoch, i+1, len(loader),loss.item())
                overwrite_previous_line(text)
                break
            self.model.eval()
            running_loss = round(running_loss / (i+1), 3)
            self.losses.append(running_loss)
            acc = self.evaluate(val_src, val_tgt[:,1:])
            print("        Running_Loss: %s" %(running_loss), "VAL_ACC: %s" %acc)

    def evaluate(self, src, y):
        num=10
        pred = np.array(self.dec_fnc(self.model,src, self.tokenizer,num))
        y_text = np.array(y[:,:num+1])
        print(pred.shape,y_text.shape,pred.size)
        acc = (pred == y_text).sum() / y_text.size
        return acc