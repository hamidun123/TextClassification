import torch
from models import BasicModule
from torch import nn
import torch.nn.functional as F

class LSTM_ATT(BasicModule):
    def __init__(self, args):
        super(LSTM_ATT, self).__init__()
        self.embedding = nn.Embedding(args.word_number, args.word_vector, padding_idx=0)
        self.lstm = nn.LSTM(args.word_vector, args.hidden_size, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(args.hidden_size * 2))
        self.fc = nn.Linear(args.hidden_size * 2, args.label_number)

    def forward(self, x):
        emb = self.embedding(x)
        H, _ = self.lstm(emb)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc(out)
        return out
