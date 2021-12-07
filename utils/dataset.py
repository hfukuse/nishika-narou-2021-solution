import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import re

def convert_examples_to_features(text, tokenizer, max_len=256):
    tok = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        padding='max_length')
    return tok

class NishikaNarouDataset(Dataset):
    def __init__(self, data, tokenizer, is_test=False):
        self.data = data
        self.excerpts = self.data.excerpt.tolist()
        if not is_test:
            self.targets = self.data.target.tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if not self.is_test:
            excerpt = self.excerpts[item]
            label = self.targets[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer, Config.max_len
            )
            return {
                'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float),
            }
        else:
            excerpt = self.excerpts[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer, Config.max_len
            )
            return {
                'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
            }
