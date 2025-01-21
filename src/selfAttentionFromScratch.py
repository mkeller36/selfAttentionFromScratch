# Embedding an input sentence 
sentence = 'Lift is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

import torch

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',','').split()])
print(sentence_int)

torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape)

# Defining the weight matrices 