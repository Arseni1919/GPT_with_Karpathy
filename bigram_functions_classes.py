import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# batch_size = 64  # !!!
batch_size = 32
block_size = 256
# max_iters = 1000
max_iters = 10
eval_interval = 2
learning_rate = 3e-4
device = 'cpu'
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     device = torch.device("mps")
# else:
#     device = 'cpu'
eval_iters = 1
n_embd = 384  # number for embedding dimensions
# n_head = 6  # !!!
# n_layer = 6  # !!!
n_head = 3
n_layer = 3
dropout = 0.2


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


class BigramLanguageModel(nn.Module):

    def __init__(self, v_size):
        super().__init__()
        self.v_size = v_size
        # super(BigramLanguageModel, self).__int__()
        self.token_embedding_table = nn.Embedding(v_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, curr_n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, curr_n_head=4),
        #     Block(n_embd, curr_n_head=4),
        #     Block(n_embd, curr_n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        self.lm_head = nn.Linear(n_embd, v_size)
        print('Created the model')

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B, T, C)
        # x = self.sa_head(x)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
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






