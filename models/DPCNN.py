import torch
from models import BasicModule
from torch import nn
import torch.nn.functional as F


class DPCNN(BasicModule):
    def __init__(self, args):
        super(DPCNN, self).__init__()
        self.args = args

        self.channel_size = 250
        self.embed = nn.Embedding(self.args.word_number, self.args.word_vector)

        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, self.args.word_vector),
                                               stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(self.channel_size, self.args.label_number)

    def forward(self, x):
        batch = x.shape[0]

        # Region embedding
        x = self.embed(x)  # (N,H,W)

        x = x.unsqueeze(1)  # (N,Ci,H,W)
        x = self.conv_region_embedding(x)  # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)  # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[2] > 2:
            x = self._block(x)

        x = x.squeeze()
        x = self.linear_out(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

