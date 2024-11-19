import torch

class MLMModel(torch.nn.Module):
    def __init__(self, bert_model, hidden_dim, vocab_size):
        """
        все крайне просто
        """
        super().__init__()
        self.bert_model = bert_model
        self.vocab_size = vocab_size
        self.linear = torch.nn.Linear(hidden_dim, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, inputs_ids, segm_ids):
        sentence_embed = self.bert_model(inputs_ids, segm_ids)
        return self.softmax(self.linear(sentence_embed))