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
        self.w_domain = nn.Parameter(torch.zeros(args.hidden_size * 2), requires_grad=True)
        self.fc_domain = nn.Linear(args.hidden_size * 2, args.domain_number)

        self.w_command = nn.Parameter(torch.zeros(args.hidden_size * 2), requires_grad=True)
        self.fc_command = nn.Linear(args.hidden_size * 4, args.command_number)

        self.w_value = nn.Parameter(torch.zeros(args.hidden_size * 2), requires_grad=True)
        self.fc_value = nn.Linear(args.hidden_size * 4, args.value_number)

    def forward(self, x, length):
        emb = self.embedding(x)

        # pack & pad
        emb_pack = pack_padded_sequence(emb, length, batch_first=True, enforce_sorted=False)
        H, _ = self.lstm(emb_pack)
        H, _ = pad_packed_sequence(H, batch_first=True, total_length=80)
        M = self.tanh1(H)

        # ATT
        # domain
        alpha0 = F.softmax(torch.matmul(M, self.w_domain), dim=1).unsqueeze(-1)
        out0 = H * alpha0
        out0 = torch.sum(out0, 1)
        out_domain = F.relu(out0)
        domain_out = self.fc_domain(out_domain)

        # command
        alpha1 = F.softmax(torch.matmul(M, self.w_command), dim=1).unsqueeze(-1)
        out1 = H * alpha1
        out1 = torch.sum(out1, 1)
        out_command = torch.cat((out1, out0), dim=1)
        out_command = F.relu(out_command)
        command_out = self.fc_command(out_command)

        # value
        alpha2 = F.softmax(torch.matmul(M, self.w_value), dim=1).unsqueeze(-1)
        out2 = H * alpha2
        out2 = torch.sum(out2, 1)
        out_value = torch.cat((out2, out1), dim=1)
        out_value = F.relu(out_value)
        value_out = self.fc_value(out_value)

        return [domain_out, command_out, value_out]
