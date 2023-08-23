from Transformer.Models import Decoder, Encoder
from Transformer.Config import Config
import torch.nn as nn

class Seq2SeqModel(nn.Model):
    def __init__(self,model,max_seq_len):

        super().__init__()

        self.model=model
        self.n_enc_seq=model.config.n_enc_seq
        self.n_dec_seq=model.config.n_dec_seq
    
    def forward(self, enc_inputs, dec_inputs):
        # (batchs, n_enc_seq, d_hidn), [(batchs, n_head, n_enc_seq, n_enc_seq)]
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs)
        # (batchs, n_seq, d_hidn), [(batchs, n_head, n_dec_seq, n_dec_seq)], [(batchs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # (batchs, n_dec_seq, n_dec_vocab)
        dec_outputs=self.projection(dec_outputs)
        # (batchs, n_dec_seq, n_dec_vocab), [(batchs, n_head, n_enc_seq, n_enc_seq)], [(batchs, n_head, n_dec_seq, n_dec_seq)], [(batchs, n_head, n_dec_seq, n_enc_seq)]
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs