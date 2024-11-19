import torch

class NSPModel(torch.nn.Module):
    def __init__(self, bert_model, hidden_dim):
        """
        все крайне просто
        """
        super().__init__()
        self.bert_model = bert_model
        self.linear = torch.nn.Linear(hidden_dim, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, inputs_ids, segm_ids):
        # using CLS to train
        sentence_embed = self.bert_model(inputs_ids, segm_ids)
        return self.softmax(self.linear(sentence_embed[:, 0]))