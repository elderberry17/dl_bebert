from gpt_arc.gpt import BeBertDecoder
from transformers import AutoTokenizer


if __name__ == "__main__":
    # haven't trained my own tokenizer though :(
    tokenizer = AutoTokenizer.from_pretrained('ai-forever/rugpt3medium_based_on_gpt2')
    model = BeBertDecoder(tokenizer.vocab_size, model_dim=256, n_blocks=6, n_heads=4)

    # it's like "hey! how are you doing" and "i am doing good" or a sort of
    inputs_start = ['hey! how are you', 'i am doing', 'can you tell me something about transformers?']
    inputs = tokenizer(inputs_start, return_tensors='pt', padding=True, truncation=True)

    # now it's bullshit because there is not training
    generated = model.generate(inputs['input_ids'], max_length=3)
    generated_str = tokenizer.batch_decode(generated, skip_special_tokens=True)
    print(generated_str)

    # example of loss calculation for train-loop
    last_tokens, loss = model(inputs['input_ids'], mode='train')
    print(loss)