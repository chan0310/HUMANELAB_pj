import torch.nn as nn
from .Models import Encoder, Decoder
import torch.nn.functional as F
import torch
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.projection = nn.Linear(self.config.d_hidn, self.config.n_dec_vocab)
        self.soft=nn.Softmax(dim=-1)

    def forward(self, enc_inputs, dec_inputs):
        # (batchs, n_enc_seq, d_hidn), [(batchs, n_head, n_enc_seq, n_enc_seq)]
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs)
        # (batchs, n_seq, d_hidn), [(batchs, n_head, n_dec_seq, n_dec_seq)], [(batchs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # (batchs, n_dec_seq, n_dec_vocab)
        dec_outputs=self.projection(dec_outputs)
        dec_outputs=self.soft(dec_outputs)
        # (batchs, n_dec_seq, n_dec_vocab), [(batchs, n_head, n_enc_seq, n_enc_seq)], [(batchs, n_head, n_dec_seq, n_dec_seq)], [(batchs, n_head, n_dec_seq, n_enc_seq)]
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
    
