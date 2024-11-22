import torch


class BeAttentionGPT(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(BeAttentionGPT, self).__init__()
        self.hidden_dim = hidden_dim

        # инициализируем свои K, Q, V для каждого аттеншна
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim)

        # scale factor
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

    def forward(self, x, attention_mask=None):
        # получаем K, Q, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # считаем скор
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # the difference in attention calculation!!
        # i cant see further tokens
        scores = scores.masked_fill(torch.tril(scores) == 0, float('-inf'))

        # # work with padding. pad_token = 0 (hard coded)
        if attention_mask is not None:
            attention_mask_pad = attention_mask[:, :, None] & attention_mask[:, None, :]
            scores = scores.masked_fill(attention_mask_pad == 0, float('-inf')) 

        # to avoid nans
        scores = scores.masked_fill(scores == float('-inf'), -1e9)

        # считаем sm
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        # получаем выход
        output = torch.matmul(attn_weights, V)

        return output
    
class BeMultiHeadAttentionGPT(torch.nn.Module):
    def __init__(self, n_heads, hidden_dim):
        super(BeMultiHeadAttentionGPT, self).__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.heads = torch.nn.ModuleList(BeAttentionGPT(self.head_dim) for _ in range(self.n_heads))
        self.heads_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, X, attention_mask):
        attention_inputs = torch.cat([self.heads[ind](X[:, :, ind*self.head_dim: (ind+1)*self.head_dim], attention_mask) for ind in range(self.n_heads)], dim=-1)
        output = self.heads_proj(attention_inputs)

        return output