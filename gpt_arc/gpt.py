import torch
from gpt_arc.utils_layers import BeLayerNorm, BeFF
from gpt_arc.attention import BeMultiHeadAttentionGPT
from gpt_arc.embeddings import BeBertEmbedding
import os
import json


class GPTBlock(torch.nn.Module):
    def __init__(
        self, model_dim, n_heads, ff_hidden, dropout=0.1):
        super(GPTBlock, self).__init__()

        self.layernorm = BeLayerNorm(model_dim)
        self.self_multihead = BeMultiHeadAttentionGPT(n_heads, model_dim)
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

class BeBertDecoder(torch.nn.Module):
    """
    BeBert model : Bezgin's Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, model_dim=128, n_blocks=3, n_heads=2, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.vocab_size = vocab_size

        self.feed_forward_hidden = self.model_dim * 4

        self.embedding = BeBertEmbedding(vocab_size=vocab_size, embed_size=self.model_dim)

        self.encoder_blocks = torch.nn.ModuleList(
            [GPTBlock(self.model_dim, self.n_heads, self.model_dim * 4, dropout) for _ in range(n_blocks)])
        
        # linear output - classify the next token + softmax for probabilities
        self.linear_output = torch.nn.Linear(self.model_dim, self.vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    # dont need segment_ids anymore
    def forward(self, input_ids):
        x = self.embedding(input_ids)

        for encoder in self.encoder_blocks:
            x = encoder.forward(x)

        # add the last layer for probability forecasting
        x = self.softmax(self.linear_output(x))

        return x

    def generate(self, input_ids, max_length=10):
        # we make forward for max_length times
        # each time we add new token
        # TODO: should check the context window len and eos-token
        for _ in range(max_length):
            output_probs = self.forward(input_ids)
            # next token for the whole input sequence
            new_token_ids = output_probs[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat((input_ids, new_token_ids.unsqueeze(1)), dim=-1)

        return input_ids

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
