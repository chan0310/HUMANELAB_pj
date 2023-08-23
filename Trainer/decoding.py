import torch
import sys
def overwrite_previous_line(text):
    sys.stdout.write('\r' + text)
    sys.stdout.flush()

def greedy_decoding(trm, src, detokenizer,start_token="_"):
    N = src.size(0)
    preds =torch.tensor( [[detokenizer.convert_tokens_to_ids(start_token)] for _ in range(N)])
    with torch.no_grad():
        for _ in range(128):
            print("\n----")
            print(str(_),end="/")
            y_pred,a,b,c = trm(src,preds)
            t_pred = torch.argmax(y_pred[:,-1,:], axis=-1, keepdims=True)
            preds = torch.cat([preds, t_pred], axis=1)            
        return preds