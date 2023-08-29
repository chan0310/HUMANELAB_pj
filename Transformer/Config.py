import json
class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    def __init__(self,vocabLen=3000):
        self.n_enc_vocab=vocabLen
        self.n_dec_vocab=vocabLen
        self.n_enc_seq=256
        self.n_dec_seq=256
        self.n_layer=4
        self.d_hidn=256
        self.i_pad=0
        self.d_ff=512
        self.n_head=4
        self.d_head=256
        self.dropout=0.5
        self.layer_norm_epsilon=1e-12
    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)
        
