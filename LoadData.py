import json
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, filename):
        self.data = []

        # Load data from the JSON file
        with open(filename, 'r') as file:
            for line in file:
                entry = json.loads(line)
                self.data.append({
                    'input_ids': torch.tensor(entry['encoded_text']['input_ids']),
                    'attention_mask': torch.tensor(entry['encoded_text']['attention_mask']),
                    'global_attention_mask': torch.tensor(entry['encoded_text']['global_attention_mask']),
                    'labels': torch.tensor(entry['label'])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]