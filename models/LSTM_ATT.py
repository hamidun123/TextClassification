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
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(args.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(args.hidden_size * 2, args.hidden_size2)
        self.fc = nn.Linear(args.hidden_size2, args.label_number)

    def forward(self, x):
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
