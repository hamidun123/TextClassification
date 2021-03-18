import torch
from models import BasicModule
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM_ATT(BasicModule):
    def __init__(self, args):
        super(LSTM_ATT, self).__init__()
        self.embedding = nn.Embedding(args.word_number, args.word_vector, padding_idx=0)
        self.lstm = nn.LSTM(args.word_vector, args.hidden_size, args.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(args.hidden_size * 2))
        self.fc = nn.Linear(args.hidden_size * 2, args.label_number)

    def forward(self, x, length):
        emb = self.embedding(x)

        # pack & pad
        emb_pack = pack_padded_sequence(emb, length, batch_first=True, enforce_sorted=False)
        H, _ = self.lstm(emb_pack)
        H, _ = pad_packed_sequence(H, batch_first=True, total_length=65)

        # ATT
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)

        # linear
        out = F.relu(out)
        out = self.fc(out)
        return out
