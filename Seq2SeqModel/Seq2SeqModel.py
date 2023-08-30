from Transformer.Config import Config
from Transformer.Model import Transformer
from tqdm import tqdm
from transformers import AutoTokenizer

import random

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence

class Seq2SeqModel:
    def __init__(self,
                  tokenizer,  loss_fn,lr
                 ,X_train, y_train, X_val, y_val,batch_size=64,device=torch.device("mps"),
                 n_head=3,n_layer=4):
        self.tokenizer=tokenizer
        self.config=Config(len(self.tokenizer.get_vocab()))

        self.config.dropout=0.2
        self.config.n_head=n_head; self.config.n_layer=n_layer #for overfitting
        self.batch_size=batch_size
        self.train_data_loader, self.config.n_enc_seq, self.config.n_dec_seq=self.text_to_DataLoader(X_train,y_train, batch_size=self.batch_size)
        self.val_data_loader,*_=self.text_to_DataLoader(X_val,y_val, batch_size=1)
        self.config.d_hidn=self.config.n_enc_seq
        self.config.d_head=self.config.d_hidn
        self.config.d_ff=self.config.n_enc_seq*2

        self.device=device
        self.model=Transformer(self.config).to(device=self.device)
        
        print(f'Using {self.device}')
        self.losses=[]
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)
        self.loss_fn=loss_fn
        
    def text_to_DataLoader(self,X,y, batch_size=256):
        x1=self.text_preprocess_for_list(X)
        y=self.text_preprocess_for_list(y)
        x2=y[:,:-1]
        y=y[:,1:]
        ds = TensorDataset(x1,x2 , y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        return loader, x1.size(-1),y.size(-1)+1


    def text_preprocess_for_list(self,list:list): #list-seq-wordindex list*n_seq
        list = [torch.tensor(seq) for seq in list]
        list= pad_sequence(list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return list
    
    # for eval
    def seq_pading(self,seq:torch.tensor,e=0):
        seq=list(seq)+[self.tokenizer.pad_token_id for _ in range(self.config.n_enc_seq-len(seq)-e)]
        print("seq_padd: ",len(seq))
        return torch.tensor(seq)
    
    def seq_to_seq_process(self,seq,seqd): #tensor to tensor vector(d=vocab) to list(d=vocab)
        seq=self.seq_pading(seq)
        dec_input=self.seq_pading(torch.tensor([self.tokenizer.bos_token_id]))
        dec_input=self.seq_pading(seqd[:-1],1)
        if self.device:
            seq=seq.to(self.device)
            dec_input=dec_input.to(self.device)
        output,*_=self.model(seq.view(1,-1),dec_input.view(1,-1))
        output=[torch.argmax(i).item() for i in output.view(output.size(1),-1)]
        return output,seq,dec_input
    #./for eval


    def train_each_batch(self,x1,x2,yy):
            y_pred,ea,de,eda = self.model(x1, x2)
            loss = self.loss_fn(y_pred.view(y_pred.size(0)*y_pred.size(1),y_pred.size(2))
                                ,yy.view(-1))

            self.optimizer.zero_grad()
            

            loss.backward()
            self.optimizer.step()
            running_loss = loss.item()
            attentions={"enc_at":ea,"dec_at":de,"enc_dec_at":eda}
            return running_loss,attentions
            
    def train_main(self,max_epoch,check=False,save=False,curLossstep=34):   
        accs=[]
        for epoch in range(1, max_epoch+1):
            self.model.train()
            running_loss = 0 
            
            bar_format = "{desc}|{bar:20}|{percentage:3.2f}%"
            total=len(self.train_data_loader)
            if check:
                total=check

            epT=tqdm(enumerate(self.train_data_loader),bar_format=bar_format,
                     total=total,unit="batches")
            cur_losses=[]    

            for i, (x1, x2, yy) in epT:
                # yy=torch.zeros(yy.size(0),yy.size(1), self.config.n_dec_vocab).scatter_(2, yy.unsqueeze(2), 1)
                if self.device:
                    x1=x1.to(self.device)
                    x2=x2.to(self.device)
                    yy=yy.to(self.device)
                current_loss,attentions=self.train_each_batch(x1,x2,yy)
                running_loss+=current_loss
                if i%curLossstep==0:
                    cur_losses.append(cur_losses)
                
                epT.set_description(f"Epoch: {epoch}/{max_epoch}   Batch: {i+1}/{total}   cost: {current_loss}")
                if check and check<=i+1:
                        break

            self.model.eval()
            running_loss = round(running_loss / (i+1), 7)
            self.losses.append(running_loss)
            acc = self.evaluate(running_loss)
            accs.append(acc)

            epT.update(1)
            epT.close() 
            if save:
                self.save_model_and_config(str(epoch),cur_losses,current_loss)

        file_path="savedModel/acc_total"
        with open(file_path, "w") as file:
            for acc in accs:
                file.write(f"{acc}\n")
        
    
    def evaluate(self,running_loss): #easy eval
        cor_cnt=0
        epT=tqdm(enumerate(self.val_data_loader))
        for i, (x1, x2, yy) in epT:
            if self.device:
                x1=x1.to(self.device)
                x2=x2.to(self.device)
                yy=yy.to(self.device)

            self.model.eval()
            predict,*attentions=self.model(x1,x2)

            random.seed(42)
            r_idx=random.randint(0, predict.size(0))
            predict=predict.max(2)[1]

            if(predict[r_idx].shape!=yy[r_idx].shape):
                print("오류")

            if (predict[0,r_idx].item()==yy[0,r_idx].item()):
                cor_cnt+=1
            epT.set_description(f"{i}/{len(self.val_data_loader)}Running_Loss: %s {running_loss}   VAL_ACC: %s {cor_cnt/(i+1)}")
        return cor_cnt/(i+1)
    
    def save_model_and_config(self,name,cur_losses,enu_loss):
        torch.save(self.model.state_dict(), "savedModel/saved"+str(name)+".pth")
        file_path="savedModel/losses"+str(name)+".txt"
        with open(file_path, "w") as file:
            for loss in cur_losses:
                file.write(f"{loss}\n")
            file.write(f"{enu_loss}")

    def load_model(self,path):
        model_checkpoint = path
        self.model.load_state_dict(torch.load(model_checkpoint,map_location=self.device))

        # config_checkpoint = "savedTk" + "/savedconfig.pth"
        # self.config = torch.load(config_checkpoint)

        # # 토크나이저를 로드합니다.
        # tokenizer_checkpoint ="savedTk" + "/savedtokenizer"
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)