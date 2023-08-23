from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
class TokenizerPlus(Tokenizer):
        
    def to_tensor(self, texts, **kwargs):
        sequences = self.texts_to_sequences(texts)
        padded = pad_sequences(sequences, **kwargs)
        return torch.tensor(padded, dtype=torch.int64)
    
    def to_string(self, tensor):
        texts = self.sequences_to_texts(tensor.data.numpy())
        return [t[::2] for t in texts]
    
