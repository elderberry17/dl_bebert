import math
import torch

class PositionalEmbedding(torch.nn.Module):
    '''
    На нечетных позициях эмбеддинга sin, на четных cos
    Каждая точка зависит от pos-токена и ind в эмбеддинге
    '''
    def __init__(self, d_model, max_len=128):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe.unsqueeze(0)

    def forward(self, input_ids):
        return self.pe[:, :input_ids.shape[1], :]
    
class SemanticEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        self.linear = torch.nn.Linear(vocab_size, embed_dim)

    def forward(self, input_ids):
        attention_mask = (input_ids == self.padding_idx)
        ohe = torch.nn.functional.one_hot(input_ids.long(), num_classes=self.vocab_size).float()
        embeddings = self.linear(ohe)
        padding_embedding = torch.full((self.embed_dim,), self.padding_idx, device=input_ids.device)
        embeddings = torch.where(attention_mask.unsqueeze(-1) == 0, padding_embedding, embeddings)

        return embeddings


class BeBertEmbedding(torch.nn.Module):
    """
    комбинируем семантические и простые позиционные эмбеддинги
    """

    def __init__(self, vocab_size, embed_size, seq_len=128, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.token = SemanticEmbedding(vocab_size, embed_size, padding_idx=0)
        self.segment = SemanticEmbedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, input_ids, segment_label):
        x = self.token(input_ids).to(input_ids.device) + self.position(input_ids) + self.segment(segment_label)
        return self.dropout(x)
