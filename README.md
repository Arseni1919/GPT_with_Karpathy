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

## Credits

- [yt | Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)
- [dataset | Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- [colab | Karpathy](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=h5hjCcLDr2WC)
- [github | nanoGPT](https://github.com/karpathy/nanoGPT)