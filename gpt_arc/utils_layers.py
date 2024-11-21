import torch


class BeFF(torch.nn.Module):
    def __init__(self, d_model, middle_dim, dropout=0.1):
        super(BeFF, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class BeLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size:int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate the squared mean E[X^2] of all elements
        mean_x2 = (x ** 2).mean(dim=-1, keepdim=True)
        # Variance of all element Var[X] = E[X^2] - E[X]^2
        var = mean_x2 - mean ** 2
        # |var| : (batch_size, seq_len, 1)

        # Normalize x. Layernorm[X] = weight * (X - E[X])/(Var[X] + eps)^0.5 + bias
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Elementwise affine transformation
        x_norm = x_norm * self.weight + self.bias
        # |x_norm| : (batch_size, seq_len, d_model)

        return x_norm
