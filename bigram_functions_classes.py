import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

batch_size = 32
block_size = 8
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
device = 'cpu'
# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     device = torch.device("mps")
# else:
#     device = 'cpu'
eval_iters = 200


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
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
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
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






