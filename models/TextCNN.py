import torch
from torch import nn
from models import BasicModule
import torch.nn.functional as F
import pandas as pd
import numpy as np

class TextCNN(BasicModule):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.word_vector = args.word_vector
        self.word_number = args.word_number
        self.label_number = args.label_number
        self.input_channel = args.input_channel
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(self.word_number, self.word_vector, padding_idx=0)

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.input_channel, self.kernel_num, (K, self.word_vector)) for K in self.kernel_sizes])

        self.dropout = nn.Dropout(args.dropout)

        self.fc = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.label_number)

        # 以下是attention机制参数
        self.attention = args.attention
        self.reduction_ratio = args.reduction_ratio

        self.fc_att = nn.Sequential(nn.Conv2d(self.kernel_num, self.kernel_num // self.reduction_ratio, 1, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(self.kernel_num // self.reduction_ratio, self.kernel_num, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        kernel_size = 7
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        x = self.embed(x)  # (N,H,W)

        x = x.unsqueeze(1)  # (N,Ci,H,W)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Kernel_num(C_out),W)

        if self.attention:
            for line in x:
                avg_out = self.fc_att(F.max_pool1d(line, int(line.size(2))).unsqueeze(3))
                max_out = self.fc_att(F.avg_pool1d(line, int(line.size(2))).unsqueeze(3))
                out = avg_out + max_out
                channel_att = self.sigmoid(out).squeeze(3)
                line = line * channel_att

                # spatial attention
                avg_out = torch.mean(line, dim=1, keepdim=True)
                max_out, _ = torch.max(line, dim=1, keepdim=True)
                line_both = torch.cat([avg_out, max_out], dim=1)
                spatital_att = self.conv1(line_both)
                line = line * spatital_att

        x = [F.max_pool1d(line, int(line.size(2))).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        x = torch.cat(x, 1)  # (N,Knum*len(Ks))
        x = self.dropout(x)
        out = self.fc(x)
        # out = F.softmax(out, dim=1)
        return out