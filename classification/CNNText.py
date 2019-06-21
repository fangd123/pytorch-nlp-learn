"""
参考资料：
https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
"""
import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.autograd import Variable


class CNNText(nn.Module):
    def __init__(self,args):
        super(CNNText,self).__init__()
        self.args = args

        embed_num = args.embed_num
        embed_dim = args.embed_dim
        class_num = args.class_num
        channel_num = 1
        kernel_num = args.kernel_nul
        kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(embed_num,embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(channel_num,kernel_num,(K,embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        # 最后一个线性层

        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num,class_num)
        self.out = nn.LogSoftmax(class_num)

    def forward(self, *input):
        x = self.embed(input)
        if self.args.static:
            x = Variable(x)

        # 添加一个input_channel=1,方便后边进入卷积层，此时x_size=(N,1,word_num,embed_dim)
        x = x.unsqueeze(1)
        # 在经过卷积层后，最后一个维度即W_out值为1（因为W_in=embed_dim），所以可以用squeeze去掉最后一个维度
        # 这里用relu的理由？可能是减小后期的计算量吧（似乎论文中没有用relu在这里）
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # i.size(2)表示word_num，也就是句子宽度，从里边找出max值，因为out值为1，所以使用squeeze去掉这个维度
        x = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in x]
        # 将每个kernel_size拼接起来，按照行拼接，1表示行，因为前边还有个Batch维
        x = torch.cat(x,1)
        x = self.dropout(x)
        logit = self.fc1(x)
        out = self.out(logit)

        return out


def train():
    out = model(x)
    F.cross_entropy(out,y)