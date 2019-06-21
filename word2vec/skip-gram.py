"""
参考资料：
https://medium.com/district-data-labs/forward-propagation-building-a-skip-gram-net-from-the-ground-up-9578814b221
https://github.com/blackredscarf/pytorch-SkipGram/
https://blog.csdn.net/weixin_40759186/article/details/87857361

"""



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


class SkipGram(nn.model):
    def __init__(self,vocab_size,embedding_dim):
        super(SkipGram,self).__init__()
        # embedding 允许稀疏表示，意味着里边有很多的0，使用了indices、value的表示方法
        self.u_embeddings = nn.Embedding(vocab_size,embedding_dim,sparse=True)
        # 这里也可以用linear层，其实是等价的，但是为了后边方便计算负采样，所以用embedding，然后再使用了多个sigmoid代替整体的softmax计算，减小计算量
        # 如果这里用linear层，则后边使用负采样的时候，需要去除linear层中的weight参数，没有直接使用embedding方便
        self.v_embeddings = nn.Embedding(vocab_size,embedding_dim,sparse=True)
        self.embedding_dim = embedding_dim
        self.init_emb()
    def init_emb(self):
        """
        为什么这里需要这样初始化，可能是别人做过试验吧！
        :return:
        """
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange,initrange)
        self.v_embeddings.weight.data.uniform_(-0,0)

    def forward(self,u_pos,v_pos,v_neg,batch_size):
        """
        :param u_pos:中心词的索引
        :param v_pos: 上下文词的索引
        :param v_neg: 非上下文词的索引
        :param batch_size: 批次大小
        :return:
        """
        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)

        # 这里使用了对应位置相乘，得到了维度不变的向量
        # 为什么要这样做？论文中的公式用的应该是点乘才对？
        score = torch.mul(embed_u,embed_v)
        score = torch.sum(score,dim=1)
        # squeeze(): 去除size为1的维度，包括行和列，也就是转换为非向量的形式
        log_target = F.logsigmoid(score).squeeze()
        # 取出负样本
        neg_embed_v = self.v_embeddings(v_neg)

        neg_score = torch.bmm(neg_embed_v,embed_u.unqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score,dim=1)
        # 负样本尽可能小
        sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_target +sum_log_sampled

        return -1*loss.sum()/batch_size

    def input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()
    def save_embedding(self,file_name,id2word):
        embeds = self.u_embeddings.weight.data
        fo = open(file_name, 'w')
        for idx in range(len(embeds)):
            word = id2word(idx)
            embed = ' '.join(embeds[idx])
            fo.write(word + ' ' + embed + '\n')

