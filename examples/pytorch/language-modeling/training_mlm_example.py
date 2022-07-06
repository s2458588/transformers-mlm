import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

wd_in = os.getcwd() + "/data/in/"
wd_out = os.getcwd() + "/data/out/"
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
model = BertForMaskedLM.from_pretrained('bert-base-german-cased')

print(torch.cuda.is_available())

with open(wd_in + "plenar.txt", 'r', encoding='utf8') as fp:
    text = fp.read() #.split('.')

short_text = "Das ist nur ein Test des Arbeitsspeichers. Die Familie der Entenvögel (Anatidae) ist die artenreichste aus der Ordnung der Gänsevögel (Anseriformes). Sie umfasst 47 Gattungen und etwa 150 Arten. Zu dieser Gruppe gehören so bekannte Typen von Wasservögeln wie die Enten, Gänse und Schwäne. Vielleicht abgesehen von den Hühnervögeln hat keine andere Vogelgruppe so zahlreiche Wechselbeziehungen zum Menschen: Allein fünf Arten wurden domestiziert. Entenvögel werden wegen ihres Fleisches, ihrer Eier und ihrer Federn gejagt und gehalten, und in vielerlei Form haben sie Eingang in Märchen, Sagen und Comics gefunden. Sprachlich bezeichnen die Begriffe Ente den weiblichen und Erpel oder Enterich den männlichen Vogel. Auffälligstes Unterscheidungsmerkmal ist das farbigere Prachtkleid der männlichen Entenvögel, der Erpel (siehe Erscheinungsbild ausgewachsener Stockenten)."

inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
inputs['labels'] = inputs.input_ids.detach().clone()

rand = torch.rand(inputs.input_ids.shape)
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

selection = []
for i in range(mask_arr.shape[0]):
    selection.append(
        torch.flatten(mask_arr[0].nonzero()).tolist()
    )

for i in range(mask_arr.shape[0]):
    inputs.input_ids[i, selection[i]] = 103


class GerParCorDS(Dataset):
    def __new__(cls, encodings, *args, **kwargs):
        print("Creating class instance")
        instance = super(GerParCorDS, cls).__new__(cls, *args, **kwargs)
        return instance

    def __init__(self, encodings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


dataset = GerParCorDS(inputs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda')
model.to(device)
model.train()

optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

epochs = 20

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
