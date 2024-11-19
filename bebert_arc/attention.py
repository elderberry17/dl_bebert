import torch


class BeAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(BeAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # инициализируем свои K, Q, V для каждого аттеншна
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim)

        # scale factor
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

    def forward(self, x):
        # получаем K, Q, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # считаем скор
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # считаем sm
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        # получаем выход
        output = torch.matmul(attn_weights, V)

        return output
    
class BeMultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads, hidden_dim):
        super(BeMultiHeadAttention, self).__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.heads = torch.nn.ModuleList(BeAttention(self.head_dim) for _ in range(self.n_heads))
        self.heads_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, X):
        attention_inputs = torch.cat([self.heads[ind](X[:, :, ind*self.head_dim: (ind+1)*self.head_dim]) for ind in range(self.n_heads)], dim=-1)
        output = self.heads_proj(attention_inputs)

        return output