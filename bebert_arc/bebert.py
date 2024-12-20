import torch
from bebert_arc.utils_layers import BeLayerNorm, BeFF
from bebert_arc.attention import BeMultiHeadAttention
from bebert_arc.embeddings import BeBertEmbedding
import os
import json


class BeBertBlock(torch.nn.Module):
    def __init__(
        self, model_dim, n_heads, ff_hidden, dropout=0.1):
        super(BeBertBlock, self).__init__()

        self.layernorm = BeLayerNorm(model_dim)
        self.self_multihead = BeMultiHeadAttention(n_heads, model_dim)
        self.feed_forward = BeFF(model_dim, middle_dim=ff_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings):
        # embeddings: (batch_size, max_len, d_model)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded

class BeBertEncoder(torch.nn.Module):
    """
    BeBert model : Bezgin's Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, model_dim=128, n_blocks=3, n_heads=2, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads

        self.feed_forward_hidden = self.model_dim * 4

        self.embedding = BeBertEmbedding(vocab_size=vocab_size, embed_size=self.model_dim)

        self.encoder_blocks = torch.nn.ModuleList(
            [BeBertBlock(self.model_dim, self.n_heads, self.model_dim * 4, dropout) for _ in range(n_blocks)])

    def forward(self, input_ids, segm_ids=None):
        segm_ids = torch.zeros(input_ids.shape).to(input_ids.device)
        x = self.embedding(input_ids, segm_ids)

        for encoder in self.encoder_blocks:
            x = encoder.forward(x)

        return x

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        config_path = os.path.join(save_directory, "config.json")

        torch.save(self.state_dict(), model_path)

        config = {
            "vocab_size": self.embedding.vocab_size,
            "model_dim": self.model_dim,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "dropout": 0.1
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, load_directory):
        config_path = os.path.join(load_directory, "config.json")
        model_path = os.path.join(load_directory, "pytorch_model.bin")

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(torch.load(model_path))
        return model
