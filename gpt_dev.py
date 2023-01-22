import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


def get_batch(split, b_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (b_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):

    def __init__(self, v_size):
        super().__init__()
        self.v_size = v_size
        # super(BigramLanguageModel, self).__int__()
        self.token_embedding_table = nn.Embedding(v_size, v_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is not None:
            B, T, C = logits.shape  # batch, time, channel
            # print('-----------------')
            # print(logits)
            logits = logits.view(B*T, C)
            # print('-----------------')
            # print(logits)
            # print('-----------------')
            # print(targets)
            targets = targets.view(B*T)
            # print('-----------------')
            # print(targets)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    # read it in to inspect it
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # print("length of dataset in characters: ", len(text))

    # let's look at the first 1000 characters
    # print(text[:1000])

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    # print(vocab_size)

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
    # print(encode('hii there'))
    # print(decode(encode('hii there')))

    # let's encode entire thing
    data = torch.tensor(encode(text), dtype=torch.long)
    # print(data.shape, data.type)
    # print(data[:1000])

    # split the data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    block_size = 8
    # print(train_data[:block_size + 1])

    x = train_data[:block_size]
    y = train_data[1:block_size+1]
    # for t in range(block_size):
    #     context = x[:t+1]
    #     target = y[t]
        # print(f'when input is {context} the target: {target}')

    # ----- #

    torch.manual_seed(1337)
    batch_size = 4
    block_size = 8

    xb, yb = get_batch('train', batch_size)
    # print('inputs:')
    # print(xb.shape)
    # print(xb)
    # print('targets:')
    # print(yb.shape)
    # print(yb)
    #
    # print('---')

    # for b in range(batch_size):
    #     for t in range(block_size):
    #         context = xb[b, :t+1]
    #         target = yb[b, t]
    #         print(f'when input is {context} the target: {target}')

    # print(xb)

    m = BigramLanguageModel(vocab_size)
    logits, loss = m(xb, yb)
    # print(logits.shape)
    # print(loss)

    idx = torch.zeros((1, 1), dtype=torch.long)

    print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

    # Let's train the model
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    batch_size = 32
    losses = []
    for steps in range(10000):
        print(f'\riter: {steps}', end='')
        xb, yb = get_batch('train', batch_size)

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(loss.item())
    print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
    plt.plot(losses)
    plt.show()









