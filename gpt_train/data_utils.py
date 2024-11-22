from torch.utils.data import Dataset


class CLMDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)

    
def collate_fn_clm(sentences, tokenizer, max_len):
    return tokenizer(sentences, max_length=max_len, padding=True, truncation=True, return_tensors='pt')