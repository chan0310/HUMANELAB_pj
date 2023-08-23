from Transformer.Models import Decoder, Encoder
from Transformer.Config import Config
from Transformer.Model import Transformer
import sys

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
import torch.utils.data as data
import torchvision
from torch.nn.utils.rnn import pad_sequence

def overwrite_previous_line(text):
    sys.stdout.write('\r' + text)
    sys.stdout.flush()

class Seq2SeqModel(nn.Module):
    def __init__(self,
                 model, tokenizer, optimizer, loss_fn
                 ,X_train, y_train, X_val, y_val,batch_size=64):
        super().__init__()
        self.tokenizer=tokenizer
        self.config=Config(len(self.tokenizer.get_vocab()))
        self.loss_fn=loss_fn
        self.optimizer=optimizer

        self.batch_size=batch_size
        self.train_data_loader, self.config.n_enc_seq, self.config.n_dec_seq=self.text_to_DataLoader(X_train,y_train, batch_size=self.batch_size)
        self.val_data_loader,*_=self.text_to_DataLoader(X_val,y_val, batch_size=self.batch_size)
        self.config.d_hidn=self.config.n_enc_seq
        self.config.d_ff=self.config.n_enc_seq*2

        self.model=Transformer(self.config)
        self.losses=[]
        
    def text_to_DataLoader(self,X,y, batch_size=64):
        x1=self.text_preprocess_for_list(X)
        y=self.text_preprocess_for_list(y)
        x2=y[:,:-1]
        y=y[:,1:]
        ds = TensorDataset(x1,x2 , y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        return loader, x1.size(-1),y.size(-1)+1


    def text_preprocess_for_list(self,list): #list-seq-wordindex list*n_seq
        list = [torch.tensor(seq) for seq in list]
        list= pad_sequence([seq.flip(0) for seq in list], batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(1)
        return list

    def train_each_batch(self,x1,x2,yy,i,epoch,max_epoch):
            y_pred,ea,de,eda = self.model(x1, x2)
            loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1)), yy.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss = loss.item()
            text='(Epoch{:4d}/{})   Batch:{}/{}   Cost:{:.6f}'.format(epoch, max_epoch,i+1, len(self.train_data_loader),loss.item())
            overwrite_previous_line(text)
            return running_loss
            
    def train_main(self,max_epoch,check=False):
        self.model.train()
        running_loss = 0
        for epoch in range(1, max_epoch+1):
            for i, (x1, x2, yy) in enumerate(self.train_data_loader):
                running_loss+=self.train_each_batch(x1,x2,yy,i,epoch,max_epoch,)
                if check and check<i:
                        break
            self.model.eval()
            running_loss = round(running_loss / (i+1), 3)
            self.losses.append(running_loss)
            acc = self.evaluate()
            print("        Running_Loss: %s" %(running_loss), "VAL_ACC: %s" %acc)
    
            
    def evaluate(self):
        return 0