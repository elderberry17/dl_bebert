import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import random


class MLMDataset(Dataset):
    def __init__(self, seqs, tokenizer, max_seq_len=128):
        self.seqs = seqs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenized_seqs = [
            self.tokenizer(seq, truncation=True, max_length=self.max_seq_len)['input_ids'] for seq in seqs
        ]
        
    def __getitem__(self, i):
        return self.tokenized_seqs[i]

    def __len__(self):
        return len(self.seqs)
    
    
class NSPDataset(Dataset):
    def __init__(self, data_pair, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.get_sent(item)
        t1_tokenized = self.tokenizer(t1, max_length=self.max_seq_len // 2, truncation=True)['input_ids']
        t2_tokenized = self.tokenizer(t2, max_length=self.max_seq_len // 2, truncation=True)['input_ids']

        t1 = [self.tokenizer.vocab['[CLS]']] + t1_tokenized + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_tokenized + [self.tokenizer.vocab['[SEP]']]

        segment_label = ([1] * len(t1) + [2] * len(t2))
        bert_input = (t1 + t2)

        output = {
            "bert_input": torch.tensor(bert_input[:self.max_seq_len]),
            "segment_label": torch.tensor(segment_label[:self.max_seq_len]),
            "is_next": torch.tensor(is_next_label),
        }

        return output
    

    def get_sent(self, index):
        '''return random sentence pair'''
        t1, t2 = self.get_corpus_line(index)

        # возвращает тройку для NSP
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        '''return sentence pair'''
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        '''return random single sentence'''
        return self.lines[random.randrange(len(self.lines))][1]
    

class LengthSortedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        # Sort indices by sequence length
        sorted_indices = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]))
        self.batches = [sorted_indices[i:i + batch_size] for i in range(0, len(sorted_indices), batch_size)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    
def collate_fn_nsp(batch):
    bert_inputs = [item['bert_input'] for item in batch]
    segment_labels = [item['segment_label'] for item in batch]
    is_next_labels = torch.stack([item['is_next'] for item in batch])

    bert_inputs_padded = pad_sequence(bert_inputs, batch_first=True, padding_value=0)
    segment_labels_padded = pad_sequence(segment_labels, batch_first=True, padding_value=0)

    return {
        "bert_input": bert_inputs_padded,
        "segment_label": segment_labels_padded,
        "is_next": is_next_labels,
    }

