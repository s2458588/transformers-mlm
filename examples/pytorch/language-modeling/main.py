import run_mlm as mlm

from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
model = BertForMaskedLM.from_pretrained('bert-base-german-cased')

text = ("Wenn Menschen zu viel essen werden sie zu träge. Der Kongress entschied am Vormittag über die neuen Bestimmungen "
        "zum Handelsabkommen mit den Balkanstaaten.")

inputs = tokenizer(text, return_tensors='pt')

inputs['labels'] = inputs.input_ids.detach().clone()

rand = torch.rand(inputs.input_ids.shape)

# mask 15% of input tokens and exclude special tokens like 101 and 102
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102)
selection = torch.flatten(mask_arr[0].nonzero()).tolist()
inputs.input_ids[0, selection] = 103

outputs = model(**inputs)
print(outputs.keys())
