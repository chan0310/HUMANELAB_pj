from Transformer.Models import Decoder, Encoder
from Transformer.Config import Config
from Transformer.Model import Transformer
import sys
from tqdm import tqdm
import torch
from transformers import AutoTokenizer


from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence

def overwrite_previous_line(text):
    sys.stdout.write('\r' + text)
    sys.stdout.flush()

class Seq2SeqModel(nn.Module):
    def __init__(self
                 ,tokenizer, optimizer, loss_fn
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

    def seq_list_to_text_list(self,seqli):
        seqli=self.text_preprocess_for_list(seqli)
        stringli=[self.tokenizer.convert_tokens_to_string([i for i in self.tokenizer.convert_ids_to_tokens(seq)])for seq in seqli]
        return stringli

    def text_preprocess_for_list(self,list): #list-seq-wordindex list*n_seq
        list = [torch.tensor(seq) for seq in list]
        list= pad_sequence([seq.flip(0) for seq in list], batch_first=False, padding_value=self.tokenizer.pad_token_id).flip(1)
        return list

    def train_each_batch(self,x1,x2,yy):
            y_pred,ea,de,eda = self.model(x1, x2)
            loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1)), yy.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss = loss.item()
            attentions={"enc_at":ea,"dec_at":de,"enc_dec_at":eda}
            return running_loss,attentions
            
    def train_main(self,max_epoch,check=False):   
        for epoch in range(1, max_epoch+1):
            self.model.train()
            running_loss = 0 
            
            bar_format = "{desc}|{bar:20}|{percentage:3.2f}%"
            total=len(self.train_data_loader)
            if check:
                total=check

            epT=tqdm(enumerate(self.train_data_loader),bar_format=bar_format,
                     total=total,unit="batches")
                
            for i, (x1, x2, yy) in epT:
                current_loss,attentions=self.train_each_batch(x1,x2,yy)
                running_loss+=current_loss
                epT.set_description(f"Epoch: {epoch}/{max_epoch}   Batch: {i+1}/{total}   cost: {current_loss:.4f}")
                if check and check<=i+1:
                        break
            epT.update(1)
            epT.close()

            self.model.eval()
            running_loss = round(running_loss / (i+1), 3)
            self.losses.append(running_loss)
            acc = self.evaluate()


            print("        Running_Loss: %s" %(running_loss), "VAL_ACC: %s" %acc,end=" ")
            
            self.save_model_and_config(str(epoch))
            
    def evaluate(self):
        return 0
    
    def save_model_and_config(self,name):
        torch.save(self.model.state_dict(), "savedModel/saved"+str(name)+"/model.pth")
        torch.save(self.config,"savedModel/saved"+str(name)+"/config.pth")
        self.tokenizer.save_pretrained("savedModel/saved"+str(name)+"/tokenizer")

    def load_model(self,path):
        model_checkpoint = path + "/model.pth"
        self.model.load_state_dict(torch.load(model_checkpoint))

        config_checkpoint = path + "/config.pth"
        self.config = torch.load(config_checkpoint)

        # 토크나이저를 로드합니다.
        tokenizer_checkpoint =path + "/tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)