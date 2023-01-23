from bigram_functions_classes import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


torch.manual_seed(1337)


# get data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))

# encoder decoder
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

# split the data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


xb, yb = get_batch('train', train_data, val_data)

model = BigramLanguageModel(vocab_size)
m = model.to(device)
logits, loss = m(xb, yb)


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# optim
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


losses_plot_dict = {'train': [], 'val': []}
for iter in range(max_iters):
    print(f'\riter: {iter}', end='')

    if iter % eval_interval == 0:
        losses = estimate_loss(m, train_data, val_data)
        losses_plot_dict['train'].append(losses["train"])
        losses_plot_dict['val'].append(losses["val"])
        print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    xb, yb = get_batch('train', train_data, val_data)

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # plot
    if iter % eval_interval == 0:
        plt.cla()
        plt.title(f'On the {device=}')
        plt.plot(losses_plot_dict["train"], label='train')
        plt.plot(losses_plot_dict["val"], label='val')
        plt.legend()
        plt.pause(0.01)


print(loss.item())
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
plt.show()




