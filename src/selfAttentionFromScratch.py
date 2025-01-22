# setting up this because python doesn't have preprocessor defines :(
debugging=True

# Embedding an input sentence 
sentence = 'Lift is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}

if debugging==True:
    print(dc)

import torch

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',','').split()])
if debugging==True:
    print(sentence_int)

torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

if debugging==True:
    print('Embedded sentence info')
    print(embedded_sentence)
    print(embedded_sentence.shape)

# Defining the weight matrices 

torch.manual_seed(123)

d = embedded_sentence.shape[1]

d_q, d_k, d_v = 24, 24, 28

W_query = torch.nn.Parameter(torch.rand(d_q, d))
W_key   = torch.nn.Parameter(torch.rand(d_k, d))
W_value = torch.nn.Parameter(torch.rand(d_v, d))

if debugging==True:
    print(W_query.size())
    print(W_key.size())
    print(W_value.size())

# Computing un-normalized Attention Weights

if debugging==True:
    x_2 = embedded_sentence[1]
    query_2 = W_query.matmul(x_2)
    key_2 = W_key.matmul(x_2)
    value_2 = W_value.matmul(x_2)
    print('query2')
    print(query_2.shape)
    print(key_2.shape)
    print(value_2.shape)



query = torch.matmul(W_query,embedded_sentence.T)
key = torch.matmul(W_key,embedded_sentence.T).T
value = torch.matmul(W_value,embedded_sentence.T).T
print('query, key, value')
print(query.size())
print(key.size())
print(value.size())

omega = torch.matmul(query.T, key.T)

print(omega)

if debugging==True:
    omega_2 = query_2.matmul(key.T)
    print(omega_2)

# Computing Attention Scores

import torch.nn.functional as F

if debugging==True:
    attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim=0)
    print('Attention weights 2')
    print(attention_weights_2)

# I am not sure why the array attention weights 2 does not show up in the matrix of all attention weights
# Look into how softtmax works later
attention_weights = F.softmax(omega / d_k**0.5, dim=0)
print('attention weights')
print(attention_weights.size())

context_vec = torch.matmul(attention_weights, value)
print(context_vec)