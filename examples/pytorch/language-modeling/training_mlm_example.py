import os

from transformers import BertTokenizer, BertForMaskedLM
import torch

from transformers import AdamW
from tqdm import tqdm

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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device('cuda')
model.to(device)
model.train()


optim = AdamW(model.parameters(), lr=1e-5)

epochs = 2

# training
for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

db = True