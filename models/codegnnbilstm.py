import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GCNConv

class CodeGNNBiLSTM(nn.Module):
    def __init__(self, config, device):
        super.__init__()
        config['modeltype'] = 'codegnnbilstm'

        self.config = config
        self.code_vocab_size = int(config['model']['code_vocab_size'])
        self.com_vocab_size = int(config['model']['com_vocab_size'])

        self.embed_size= int(config['model']['embed_size'])
        self.hidden_size = int(config['model']['hidden_size'])

        self.SrcEmbed = nn.Embedding(self.code_vocab_size, self.embed_size)
        self.SumryEmbed = nn.Embedding(self.com_vocab_size, self.embed_size)
        self.ConvGNN = GCNConv(self.embed_size, self.hidden_size)
        self.GRU1 = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
        self.GRU2 = nn.LSTM(self.hidden_size, self.hidden_size/2, bidirectional=True, batch_first=True)
        self.GRU3 = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)
        self.FC = nn.Linear(3*self.hidden_size, self.hidden_size, bias=False)
        self.Relu = nn.ReLU()
        self.FCOut = nn.Linear(self.hidden_size, self.com_vocab_size, bias=False)


    def forward(self, data):
        seq, x, edge_index, y = data.seq, data.x, data.edge_index, data.y
        seq_embed = self.SrcEmbed(seq)
        node_embed = self.SrcEmbed(x)
        com_embed = self.SrcEmbed(y)
        astwork = node_embed
        # according to the paper, hop=2
        for i in range(2):
            astwork = self.ConvGNN(astwork, edge_index)

        seq_out, (hn1, cn1) = self.GRU1(seq_embed)
        ast_out, (hn2, cn2) = self.GRU2(astwork)
        com_out, (hn3, cn3) = self.GRU3(com_embed, (hn1, cn1))
        # [batch_size, com_len, hidden_size] * [batch_size, hidden_size, seq_size]
        seq_out_ = seq_out.permute(0,2,1)
        attn1 = torch.bmm(com_out, seq_out_)
        attn1 = f.softmax(attn1, -1)
        context1 = torch.bmm(attn1, seq_out)

        ast_out_ = ast_out.permute(0,2,1)
        attn2 = torch.bmm(com_out, ast_out_)
        attn2 = f.softmax(attn2, -1)
        context2 = torch.bmm(attn2, ast_out)

        context = torch.cat((context1, com_out, context2), -1)
        out = self.Relu(self.FC(context))
        out = torch.flatten(out)
        out = self.FCOut(out)
        out = f.softmax(out, -1)

        return out