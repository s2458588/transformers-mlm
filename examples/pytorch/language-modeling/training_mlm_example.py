import os
from transformers import BertTokenizer, BertForMaskedLM
import torch

wd_in = os.getcwd() + "/data/in/"
wd_out = os.getcwd() + "/data/out/"
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
model = BertForMaskedLM.from_pretrained('bert-base-german-cased')

with open(wd_in + "plenar.txt", 'r', encoding='utf8') as fp:
    text = fp.read().split('.')

inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
inputs['labels'] = inputs.input_ids.detach().clone()

rand = torch.rand(inputs.input_ids.shape)
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102)
selection = torch.flatten(mask_arr[0].nonzero()).tolist()
inputs.input_ids[0, selection] = 103

selection = []
for i in range(mask_arr.shape[0]):
    selection.append(
        torch.flatten(mask_arr[0].nonzero()).tolist()
    )

for i in range(mask_arr.shape[0]):
    inputs.input_ids[i, selection[i]] = 103

class GerParCorDS(torch.utils.data.Dataset):
    def __int__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


dataset = GerParCorDS(inputs)
dataloader = torch.utils.DataLoader(dataset, batch_size=16, shuffle=True)



db = True