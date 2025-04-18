import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import ContrastiveLoss, InfoNCELoss, TripletLoss
from utils import l2norm


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout).to('cuda')
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout).to('cuda')
        # self.feedforward = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     nn.ReLU(),
        #     nn.Linear(d_model * 4, d_model)
        # )
        self.linear1 = nn.Linear(d_model, d_model * 4).to('cuda')
        self.linear2 = nn.Linear(d_model * 4, d_model).to('cuda')
        self.relu = nn.ReLU()

        self.norm1 = nn.LayerNorm(d_model).to('cuda')
        self.norm2 = nn.LayerNorm(d_model).to('cuda')
        self.norm3 = nn.LayerNorm(d_model).to('cuda')
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # 多头交叉注意力模块
        cross_attn_output, _ = self.cross_attn(query, key, value)
        cross_attn_output = self.dropout1(cross_attn_output)
        query = self.norm1(query + cross_attn_output)

        # 多头自注意力模块
        self_attn_output, _ = self.self_attn(query, query, query)
        self_attn_output = self.dropout2(self_attn_output)
        query = self.norm2(query + self_attn_output)

        # 前馈网络模块
        feedforward_output = self.linear2(self.relu(self.linear1(query)))
        feedforward_output = self.dropout3(feedforward_output)
        # feedforward_output = self.feedforward(query)
        # feedforward_output = self.dropout3(feedforward_output)
        output = self.norm3(query + feedforward_output)

        return output


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super(SelfAttentionLayer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=nhead, dim_feedforward=d_model
                                                        , dropout=dropout).to('cuda')
        self.self_attn = nn.TransformerEncoder(self.encoder_layer, num_layers=1).to('cuda')

    def forward(self, img_feat, txt_feat):
        bs = img_feat.shape[0]

        all_feat = torch.cat((img_feat, txt_feat), dim=0)
        all_enhanced_feat = self.self_attn(all_feat.unsqueeze(1)).squeeze(1)
        img_enhanced_feat, txt_enhanced_feat = torch.split(all_enhanced_feat, bs, dim=0)
        return img_enhanced_feat, txt_enhanced_feat


# 使用示例
# 假设图像特征为 image_features, 文本特征为 text_features
# image_features = torch.randn(batch_size, seq_len, d_model)
# text_features = torch.randn(batch_size, seq_len, d_model)
#
# cross_attn_layer = CrossAttentionLayer(d_model, nhead)
# enhanced_image_features = cross_attn_layer(image_features, text_features, text_features)
# enhanced_text_features = cross_attn_layer(text_features, image_features, image_features)

class CrossModalInteractionLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb_size = args.emb_size
        self.dropout = 0.3
        self.cross_modal_interaction_loss = TripletLoss(args)

        self.cross_attn_layer = CrossAttentionLayer(self.emb_size, 8, self.dropout)
        self.self_attn_layer = SelfAttentionLayer(self.emb_size, 8, self.dropout)
        self.iter_count = 0

    def forward(self, img_emb, txt_emb, img_ids):
        img_enhanced_feat = self.cross_attn_layer(img_emb, txt_emb, txt_emb)
        txt_enhanced_feat = self.cross_attn_layer(txt_emb, img_emb, img_emb)

        img_emb = l2norm(img_emb, dim=-1)
        txt_emb = l2norm(txt_emb, dim=-1)

        # img_enhanced_feat = l2norm(img_enhanced_feat, dim=-1)
        # txt_enhanced_feat = l2norm(txt_enhanced_feat, dim=-1)
        #
        # img_enhanced_feat = img_enhanced_feat + img_emb
        # txt_enhanced_feat = txt_enhanced_feat + txt_emb
        #
        # img_enhanced_feat = l2norm(img_enhanced_feat, dim=-1)
        # txt_enhanced_feat = l2norm(txt_enhanced_feat, dim=-1)

        img_enhanced_feat_, txt_enhanced_feat_ = self.self_attn_layer(img_enhanced_feat, txt_enhanced_feat)

        img_enhanced_feat = l2norm(img_enhanced_feat, dim=-1)
        txt_enhanced_feat = l2norm(txt_enhanced_feat, dim=-1)

        img_enhanced_feat_ = l2norm(img_enhanced_feat_, dim=-1)
        txt_enhanced_feat_ = l2norm(txt_enhanced_feat_, dim=-1)

        img_enhanced_feat_ = img_enhanced_feat_ + img_enhanced_feat
        txt_enhanced_feat_ = txt_enhanced_feat_ + txt_enhanced_feat

        if self.iter_count >= self.args.cross_warmup:
            cross_modal_interaction_loss_1 = \
                self.cross_modal_interaction_loss(img_emb, txt_enhanced_feat_, img_ids)[0]
            cross_modal_interaction_loss_2 = \
                self.cross_modal_interaction_loss(img_enhanced_feat_, txt_emb, img_ids)[0]

            cross_modal_interaction_loss = self.cross_modal_interaction_loss(img_enhanced_feat_
                                                                             , txt_enhanced_feat_, img_ids)[0]
            # cross_modal_interaction_loss = self.nce_loss(enhanced_image_features, enhanced_text_features)

            all_cross_modal_interaction_loss = (cross_modal_interaction_loss_1
                                                + cross_modal_interaction_loss_2
                                                + cross_modal_interaction_loss)

            print('cross_modal_interaction_loss_1', cross_modal_interaction_loss_1.item())
            print('cross_modal_interaction_loss_2', cross_modal_interaction_loss_2.item())
            print('cross_modal_interaction_loss', cross_modal_interaction_loss.item())
            print('all_cross_modal_interaction_loss', all_cross_modal_interaction_loss.item())
        else:
            all_cross_modal_interaction_loss = 0

        self.iter_count += 1

        return all_cross_modal_interaction_loss
