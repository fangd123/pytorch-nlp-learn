import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

class CBOW(nn.Module):
    def __init__(self,vocab_size,embedding_dim,context_size):
        super(CBOW,self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim * 2,128)
        self.linear2 = nn.Linear(128,vocab_size)

    def forward(self, inputs):
        embed = self.embed(inputs).view(1,-1)
        out = F.relu(self.linear1(embed))
        out = self.linear2(out)
        log_prob = F.log_softmax(out,dim=1)
        return log_prob


def context2idx(context,word_to_ix):
    return [word_to_ix[x] for x in context]

loss_function = nn.NLLLoss()

model = CBOW(vocab_size,10,CONTEXT_SIZE)

optimizer = optim.SGD(model.parameters(),0.001)

for epoch in range(10):
    for context,target in data:
        model.zero_grad()
        input = torch.tensor(context2idx(context,word_to_ix),dtype=torch.long)
        target = torch.tensor([word_to_ix[target]],dtype=torch.long)
        #print(input,target)
        log_probs = model(input)
        loss = loss_function(log_probs,target)
        loss.backward()
        optimizer.step()

        print(loss)