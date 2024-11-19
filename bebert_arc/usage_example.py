from bebert_arc.bebert import BeBertEncoder
from transformers import AutoTokenizer


if __name__ == "__main__":
    # haven't trained my own tokenizer though :(
    tokenizer = AutoTokenizer.from_pretrained('elderberry17/silly-sentence-bert')
    model = BeBertEncoder(tokenizer.vocab_size, model_dim=256, n_blocks=6, n_heads=4)

    inputs = tokenizer(['hey! is it your own bert?', 'hey! its my own bert!'], return_tensors='pt', padding=True)
    outputs = model(inputs['input_ids'], inputs['token_type_ids'])
    print(outputs.shape)
