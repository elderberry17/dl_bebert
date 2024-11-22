from gpt_train.data_utils import CLMDataset, collate_fn_clm 
from torch.utils.data import DataLoader   
from gpt_arc.gpt import BeBertDecoder
from gpt_train.train_utils import train_val_loop

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.optim import AdamW



if __name__ == "__main__":
    # init model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('ai-forever/rugpt3medium_based_on_gpt2')
    model = BeBertDecoder(tokenizer.vocab_size, model_dim=768, n_blocks=6, n_heads=4)

    device = 'cuda:0'
    model.to(device)

    texts_ru = open('yandex_translate_corpus/texts_ru.txt', 'r').read()
    texts_en = open('yandex_translate_corpus/texts_en.txt', 'r').read()
    
    sentences = texts_ru.split('\n') + texts_en.split('\n')
    sentences = list(filter(lambda x: len(x) > 0, sentences))
    sentences_train, sentences_val = train_test_split(sentences, random_state=42, train_size=0.8)

    dataset_train = CLMDataset(sentences_train)
    loader_train = DataLoader(dataset_train, collate_fn=lambda x: collate_fn_clm(x, tokenizer, max_len=512), batch_size=64, shuffle=True)

    dataset_val = CLMDataset(sentences_val)
    loader_val = DataLoader(dataset_val, collate_fn=lambda x: collate_fn_clm(x, tokenizer, max_len=512), batch_size=64, shuffle=False)

    num_epochs = 10
    optimizer = AdamW(params=model.parameters(), lr=10e-6)

    train_val_loop(model, loader_train, loader_val, optimizer, num_epochs, checkpoints_dir='checkpoints_clm', device=device)
