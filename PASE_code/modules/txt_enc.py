import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import l2norm
from modules.adcap import AP
from modules.bert import BertModel as BertModel_
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adj_matrix):
        input = self.dropout(input)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj_matrix, support) + self.bias
        return output


class MultiLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, dropout=0.2):
        super(MultiLayerGCN, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # 添加第一层
        self.layers.append(GraphConvolutionLayer(in_features, hidden_features, dropout))

        # 添加中间层
        for _ in range(max(0, num_layers - 2)):
            self.layers.append(GraphConvolutionLayer(hidden_features, hidden_features, dropout))

        # 添加最后一层
        if num_layers > 1:
            self.layers.append(GraphConvolutionLayer(hidden_features, out_features, dropout))
        else:
            # 如果只有一层，直接从输入特征到输出特征
            self.layers[0] = GraphConvolutionLayer(in_features, out_features, dropout)

        self.activation = nn.ReLU()

    def forward(self, x, adj_matrix):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_matrix)
            if i < len(self.layers) - 1:  # 除最后一层外，都应用激活函数
                x = self.activation(x)
        return x

class triple_Transformer(nn.Module):
    def __init__(self,args):
        self.bert_type = args.bert_type
        super().__init__()
        self.model = BertModel_.from_pretrained(self.bert_type)
        self.linear = nn.Linear(768, 1024)
        self.length = 5

    def forward(self, head_inputs, relation_inputs, tail_inputs, attention_mask):


        text_emb = self.model(
            input_ids=head_inputs.cuda(),
            input_ids_rel=relation_inputs.cuda(),
            input_ids_tail=tail_inputs.cuda(),
            token_type_ids=head_inputs.cuda(),
            )

        text_emb = text_emb[0][:, 0, :].squeeze(1)
        text_emb = self.linear(text_emb)
        print(text_emb.shape)
        return text_emb

class triple_Bert(nn.Module):
    def __init__(self, args):
        self.bert_type = args.bert_type
        super().__init__()
        self.bert = BertModel.from_pretrained(self.bert_type)
        self.linear = nn.Linear(768, 1024)
        self.gcn = GraphConvolutionLayer(2048, 2048, 0.5)
    def forward(self, head_inputs, relation_inputs, tail_inputs,txt_emb):
        bs = head_inputs.shape[0]
        #128,1024
        head_mask = (head_inputs != 0).float()
        head_inputs = self.bert(head_inputs, head_mask).last_hidden_state
        head_inputs = torch.mean(head_inputs, dim=1)
        head_inputs = self.linear(head_inputs)
        # 128,1024
        rel_mask = (relation_inputs != 0).float()
        relation_inputs = self.bert(relation_inputs, rel_mask).last_hidden_state
        relation_inputs = torch.mean(relation_inputs, dim=1)
        relation_inputs = self.linear(relation_inputs)
        # 128,1024
        tail_mask = (tail_inputs != 0).float()
        tail_inputs = self.bert(tail_inputs, tail_mask).last_hidden_state
        tail_inputs = torch.mean(tail_inputs, dim=1)
        tail_inputs = self.linear(tail_inputs)

        inputs = torch.cat([head_inputs,tail_inputs], dim=-1)  # [batch_size, 2048]
        inputs = F.normalize(inputs, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)
        similarity_matrix=torch.mm(txt_emb,txt_emb.transpose(0, 1))
        # 计算相似度矩阵
        # similarity_matrix = torch.mm(inputs, inputs.transpose(0, 1))  # [batch_size, batch_size]
        # similarity_matrix = F.softmax(similarity_matrix, dim=-1)

        #128,2048
        gcn_all = self.gcn(inputs, similarity_matrix)

        head_inputs, tail_inputs = torch.split(gcn_all, 1024, dim=-1)

        head_inputs = F.normalize(head_inputs, dim=-1)
        relation_inputs = F.normalize(relation_inputs, dim=-1)
        tail_inputs = F.normalize(tail_inputs, dim=-1)

        # out = head_inputs + relation_inputs - tail_inputs
        out = (head_inputs + relation_inputs + tail_inputs) / 3
        out = F.normalize(out, dim=-1)
        #三元组融合以后的输出
        return out

# Bi-GRU based Language Enocder
class EncoderText_BiGRU(nn.Module):
    def __init__(self, vocab_size, word_dim, emb_size, num_layers,
                 use_bigru=False, no_txtnorm=False):
        super(EncoderText_BiGRU, self).__init__()
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.emb_size = emb_size
        self.no_txtnorm = no_txtnorm
        self.embed = nn.Embedding(self.vocab_size, self.word_dim)
        self.use_bigru = use_bigru
        self.rnn = nn.GRU(self.word_dim, self.emb_size, num_layers,
                          batch_first=True,
                          bidirectional=self.use_bigru)
        self.pool = AP(self.emb_size)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, txt, lengths):
        '''
           Extract sentence features
           input = [batch_size, seq_len]
           output = [batch_size, seq_len, emb_size]
        '''
        # embedding layer
        txt_emb = self.embed(txt)
        self.rnn.flatten_parameters()
        try:
            txt_emb = pack_padded_sequence(txt_emb, lengths, batch_first=True)
        except:
            txt_emb = pack_padded_sequence(txt_emb, lengths, enforce_sorted=False,
                                           batch_first=True)
        # bi rnn layers
        txt_emb, _ = self.rnn(txt_emb)
        txt_emb, length = pad_packed_sequence(txt_emb, batch_first=True)

        if self.use_bigru:
            txt_emb = (txt_emb[:, :, :txt_emb.size(2) // 2] + txt_emb[:, :, txt_emb.size(2) // 2:]) / 2

        # pooling operation
        txt_lengths = torch.LongTensor(length).to(txt_emb.device)
        txt_emb, _ = self.pool(txt_emb, txt_lengths)

        # normalization
        if not self.no_txtnorm:
            txt_emb = l2norm(txt_emb, dim=-1)

        return txt_emb, length


# BERT based Language Encoder
class EncoderText_BERT(nn.Module):
    def __init__(self, bert_type, emb_size, args, no_txtnorm=False):
        super(EncoderText_BERT, self).__init__()
        self.args = args
        self.bert_type = bert_type
        self.emb_size = emb_size
        self.no_txtnorm = no_txtnorm

        self.bert_1 = BertModel.from_pretrained(self.bert_type)
        self.bert_dim = self.bert_1.config.hidden_size

        # self.bert_2 = BertModel.from_pretrained(self.bert_type)
        # self.bert_dim = self.bert_2.config.hidden_size

        # self.bert_2 = triple_Transformer(args).cuda()
        self.bert_2 = triple_Bert(args).cuda()

        self.linear_1 = nn.Linear(self.bert_dim, 512)
        self.fc = nn.Linear(self.bert_dim, self.emb_size)
        self.fc_1 = nn.Linear(self.bert_dim, self.emb_size)
        self.pool = AP(self.emb_size)
        self.avgpool =Avg_Pooling_Variable()

    def forward(self, txt, lengths, et_txts, et_txt_lengths, sub_txts, rel_txts,  obj_txts):
        batch_size, max_length = txt.shape
        txt_mask = torch.ones((txt.shape),
                              dtype=torch.int)
        txt_mask = (txt != 0).float()
        txt_emb = self.bert_1(txt, txt_mask).last_hidden_state
        fg_txt_emb = self.linear_1(txt_emb)

        txt_emb = self.fc(txt_emb)


        # et_txt_emb = self.bert_2(sub_txts, rel_txts, obj_txts)

        # et_txts_mask = (et_txts != 0).float()
        # et_txt_emb = self.bert_2(et_txts, et_txts_mask).last_hidden_state
        # et_txt_emb = self.fc(et_txt_emb)

        # # pooling operation
        txt_lengths = torch.LongTensor(lengths).to(txt_emb.device)
        txt_emb, _ = self.pool(txt_emb, txt_lengths)
        et_txt_emb = self.bert_2(sub_txts, rel_txts, obj_txts, txt_emb)
        #
        # et_txt_lengths = torch.LongTensor(et_txt_lengths).to(et_txt_emb.device)
        # et_txt_emb, _ = self.pool(et_txt_emb, et_txt_lengths)

        # txt_emb = txt_emb + et_txt_emb

        # txt_emb = torch.cat((txt_emb.unsqueeze(1), et_txt_emb.unsqueeze(1)), dim=1)
        #
        # l = [torch.tensor(2) for _ in range(batch_size)]
        # l = torch.LongTensor(l).to(txt_emb.device)
        #
        # txt_emb, _ = self.pool(txt_emb, l)

        txt_emb = 0.9*txt_emb + 0.1 * et_txt_emb

        # normalization
        if not self.no_txtnorm:
            txt_emb = l2norm(txt_emb, dim=-1)
            fg_txt_emb = l2norm(fg_txt_emb, dim=-1)

        return txt_emb, fg_txt_emb, lengths


# 平均池化
def avg_pool1d_var(x, dim, lengths):
    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):
        # keep use all number of features
        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim - 1)[0]
        avg_i = tmp.mean(dim - 1)

        results.append(avg_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results

class Avg_Pooling_Variable(nn.Module):
    def __init__(self, dim=1):
        super(Avg_Pooling_Variable, self).__init__()

        self.dim = dim

    def forward(self, features, lengths):
        pool_weights = None
        pooled_features = avg_pool1d_var(features, dim=self.dim, lengths=lengths)

        return pooled_features, pool_weights