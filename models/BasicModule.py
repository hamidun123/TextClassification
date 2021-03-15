from torch import nn
import torch
import time


class BasicModule(nn.Module):
    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 模型的默认名字

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

