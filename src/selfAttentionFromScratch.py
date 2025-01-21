sentence = 'Lift is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

import torch

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',','').split()])
print(sentence_int)