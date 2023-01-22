class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

print(p1.name)
print(p1.age)

class BigramLanguageModel:

    def __init__(self, v_size):
        self.v_size = v_size
        # super(BigramLanguageModel, self).__int__()
        # self.token_embedding_table = nn.Embedding(v_size, v_size)
        self.token_embedding_table = {}

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        # loss = F.cross_entropy(logits, targets)
        return logits


m = BigramLanguageModel(12)
print(m)



