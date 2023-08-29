from .Layers import EncoderLayer, DecoderLayer
import torch.nn as nn
import torch
import numpy as np


class Embeding(nn.Module):
    def get_sinusoid_encoding_table(self,n_seq, d_hidn):
        def cal_angle(position, i_hidn):
            return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
        def get_posi_angle_vec(position):
            return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

        sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

        return sinusoid_table
    
    def __init__(self,config,n_vocab,pos_seq):
        super().__init__()
        self.config=config
        self.n_vocab=n_vocab
        self.pos_seq=pos_seq

        self.emb=nn.Embedding(self.n_vocab, config.d_hidn)
        sinusoid_table=torch.FloatTensor(self.get_sinusoid_encoding_table(pos_seq+1,self.config.d_hidn))
        self.pos_emb=nn.Embedding.from_pretrained(sinusoid_table,freeze=True)

    
    def forward(self,inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)

        # (batchs, n_enc_seq, d_hidn)
        outputs = self.emb(inputs) + self.pos_emb(positions)
        return outputs

def get_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return pad_attn_mask

def get_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask


class Encoder(nn.Module):
    def __init__(self, config,scale_emb=False):
        super().__init__()
        self.config = config

        self.emb=Embeding(self.config,self.config.n_enc_vocab,self.config.n_enc_seq)
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
        self.dropout = nn.Dropout(p=config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_hidn, eps=config.layer_norm_epsilon)
        self.scale_emb = scale_emb
        self.d_model = config.d_hidn

        self.encoder_batch_norms = nn.ModuleList([nn.BatchNorm1d(self.config.d_hidn) for _ in range(self.config.n_layer)])


    def forward(self, inputs):
        # (batchs, n_enc_seq, d_hidn)
        outputs = self.emb(inputs)
        if self.scale_emb:
            outputs *= self.d_model ** 0.5
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs)

        # (batchs, n_enc_seq, n_enc_seq)
        attn_mask = get_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []
        for layer, batch_norm_layer in zip(self.layers, self.encoder_batch_norms):
            # (batchs, n_enc_seq, d_hidn), (batchs, n_head, n_enc_seq, n_enc_seq)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)

            outputs =batch_norm_layer(outputs)

        # (batchs, n_enc_seq, d_hidn), [(batchs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, attn_probs
    
class Decoder(nn.Module):
    def __init__(self, config,scale_emb=False):
        super().__init__()
        self.config = config

        self.emb=Embeding(self.config,self.config.n_dec_vocab,self.config.n_dec_seq)
        self.dropout = nn.Dropout(p=config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])
        self.scale_emb = scale_emb
        self.d_model = config.d_hidn
        self.layer_norm = nn.LayerNorm(config.d_hidn, eps=config.layer_norm_epsilon)

        self.decoder_batch_norms = nn.ModuleList([nn.BatchNorm1d(self.config.d_hidn-1) for _ in range(self.config.n_layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
       
        # (batchs, n_dec_seq, d_hidn)
        dec_outputs = self.emb(dec_inputs)
        if self.scale_emb:
            dec_outputs *= self.d_model ** 0.5
        dec_outputs = self.dropout(dec_outputs)
        dec_outputs = self.layer_norm(dec_outputs)

        # (batchs, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = get_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (batchs, n_dec_seq, n_dec_seq)
        dec_attn_decoder_mask = get_decoder_mask(dec_inputs)
        # (batchs, n_dec_seq, n_dec_seq)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        # (batchs, n_dec_seq, n_enc_seq)
        dec_enc_attn_mask = get_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)

        self_attn_probs, dec_enc_attn_probs = [], []

        for layer, batch_norm_layer in zip(self.layers, self.decoder_batch_norms):
            # (batchs, n_dec_seq, d_hidn), (batchs, n_dec_seq, n_dec_seq), (bs, n_dec_seq, n_enc_seq)
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)

            # 배치 정규화를 적용
            dec_outputs = batch_norm_layer(dec_outputs)

        # (batchs, n_dec_seq, d_hidn), [(batchs, n_dec_seq, n_dec_seq)], [(bs, n_dec_seq, n_enc_seq)]
        return dec_outputs, self_attn_probs, dec_enc_attn_probs