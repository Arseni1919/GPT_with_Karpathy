# [GPT With Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)

## Math trick in self-attention

Instead of this:

```python
import torch
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b, t] = torch.mean(xprev, 0)
```

Do this:

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x
torch.allclose(xbow, xbow2)
# True
```

```python
import torch
torch.manual_seed(42)
# a = torch.ones(3, 3)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3,2)).float()
c = a @ b
print(f'{a=}')
print(f'{b=}')
print(f'{c=}')
```

## Self-Attention Head

```python
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# query - what am I looking for
# key - what do I contain

# a single head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)    # (B, T, 16)
q = query(x)  # (B, T, 16)
# wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) -> (B, T, T)
wei = q @ k.transpose(-2, -1) * head_size ** -0.5  # (B, T, 16) @ (B, 16, T) -> (B, T, T)
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
# out = wei @ x

out.shape
```

## The Full Model

### BigramLanguageModel Class

```python
class BigramLanguageModel(nn.Module):

    def __init__(self, v_size):
        super().__init__()
        self.v_size = v_size
        self.token_embedding_table = nn.Embedding(v_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, curr_n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, v_size)
        print('Created the model')

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is not None:
            B, T, C = logits.shape  # batch, time, channel
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

### Block Class

```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, curr_n_embd, curr_n_head):
        super().__init__()
        head_size = curr_n_embd // curr_n_head
        self.sa = MultiHeadAttention(curr_n_head, head_size)
        self.ffwd = FeedForward(curr_n_embd)
        self.ln1 = nn.LayerNorm(curr_n_embd)
        self.ln2 = nn.LayerNorm(curr_n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

### MultiHeadAttention Class

```python
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
```

### Head Class

```python
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # query - what am I looking for
        # key - what do I contain
        k = self.key(x)  # (B, T, 16)
        q = self.query(x)  # (B, T, 16)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
```

### FeedForward Class

```python
class FeedForward(nn.Module):

    def __init__(self, ff_n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ff_n_embd, 4 * ff_n_embd),
            nn.ReLU(),
            nn.Linear(4 * ff_n_embd, ff_n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

### Functions

```python
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            print('before model')
            logits, loss = model(X, Y)
            print('after model')
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

### Parameters

```python
batch_size = 64
block_size = 256
max_iters = 1000
eval_interval = 2
learning_rate = 3e-4
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = 'cpu'
eval_iters = 1
n_embd = 384  # number for embedding dimensions
n_head = 6  
n_layer = 6  
dropout = 0.2
```

### The Train Loop

```python
from bigram_functions_classes import *
import torch
import matplotlib.pyplot as plt

# get data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

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

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# optim
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

losses_plot_dict = {'train': [], 'val': []}
for iter in range(max_iters):
    print(f'\riter: {iter}', end='')

    if iter % eval_interval == 0:
        losses = estimate_loss(m, train_data, val_data)
        losses_plot_dict['train'].append(losses["train"])
        losses_plot_dict['val'].append(losses["val"])
        print(f'\n[{device}] step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

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
plt.show()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

## Credits

- [yt | Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)
- [dataset | Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- [colab | Karpathy](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=h5hjCcLDr2WC)
- [github | nanoGPT](https://github.com/karpathy/nanoGPT)