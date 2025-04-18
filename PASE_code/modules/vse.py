import torch
import torch.nn as nn
import numpy as np
from modules.txt_enc import EncoderText_BERT, EncoderText_BiGRU
from modules.img_enc import EncoderImage_ProjFeature, EncoderImage_BUTD, EncoderImage_VIT
from modules.resnet import ResNetFeatureExtractor
import math
import torch.nn.functional as F


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


########################################################
### For the dimensional selective mask, we design both heuristic and adaptive strategies.
### You can use this flag to control which strategy is selected. True -> heuristic strategy, False -> adaptive strategy

heuristic_strategy = False
########################################################


if heuristic_strategy:

    ### Heuristic Dimensional Selective Mask
    class Image_levels(nn.Module):
        def __init__(self, opt):
            super(Image_levels, self).__init__()
            self.sub_space = opt.embed_size
            self.kernel_size = int(opt.kernel_size)

            self.kernel_img_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_img_2 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_img_3 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_img_4 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_img_5 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_img_6 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_img_7 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_img_8 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)

        def get_image_levels(self, img_emb, batch_size, n_region):
            img_emb_1 = self.kernel_img_1(img_emb.reshape(-1, self.sub_space).unsqueeze(-2)).sum(1)
            img_emb_1 = l2norm(img_emb_1.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_1.size(-1)
            img_emb_2 = self.kernel_img_2(img_emb_1.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            img_emb_2 = l2norm(img_emb_2.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_2.size(-1)
            img_emb_3 = self.kernel_img_3(img_emb_2.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            img_emb_3 = l2norm(img_emb_3.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_3.size(-1)
            img_emb_4 = self.kernel_img_4(img_emb_3.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            img_emb_4 = l2norm(img_emb_4.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_4.size(-1)
            img_emb_5 = self.kernel_img_5(img_emb_4.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            img_emb_5 = l2norm(img_emb_5.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_5.size(-1)
            img_emb_6 = self.kernel_img_6(img_emb_5.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            img_emb_6 = l2norm(img_emb_6.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_6.size(-1)
            img_emb_7 = self.kernel_img_7(img_emb_6.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            img_emb_7 = l2norm(img_emb_7.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_7.size(-1)
            img_emb_8 = self.kernel_img_8(img_emb_7.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            img_emb_8 = l2norm(img_emb_8.reshape(batch_size, n_region, -1), -1)

            return torch.cat(
                [img_emb, img_emb_1, img_emb_2, img_emb_3, img_emb_4, img_emb_5, img_emb_6, img_emb_7, img_emb_8], -1)

        def forward(self, img_emb):
            batch_size, n_region, embed_size = img_emb.size(0), img_emb.size(1), img_emb.size(2)

            return self.get_image_levels(img_emb, batch_size, n_region)


    class Text_levels(nn.Module):

        def __init__(self, opt):
            super(Text_levels, self).__init__()
            self.sub_space = opt.embed_size
            self.kernel_size = int(opt.kernel_size)

            self.kernel_txt_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_txt_2 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_txt_3 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_txt_4 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_txt_5 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_txt_6 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_txt_7 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)
            self.kernel_txt_8 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size,
                                          stride=self.kernel_size, bias=False)

        def get_text_levels(self, cap_i, batch_size, n_word):
            cap_i_1 = self.kernel_txt_1(cap_i.reshape(-1, self.sub_space).unsqueeze(-2)).sum(1)
            cap_i_expand_1 = l2norm(cap_i_1.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_1.size(-1)
            cap_i_2 = self.kernel_txt_2(cap_i_1.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            cap_i_expand_2 = l2norm(cap_i_2.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_2.size(-1)
            cap_i_3 = self.kernel_txt_3(cap_i_2.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            cap_i_expand_3 = l2norm(cap_i_3.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_3.size(-1)
            cap_i_4 = self.kernel_txt_4(cap_i_3.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            cap_i_expand_4 = l2norm(cap_i_4.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_4.size(-1)
            cap_i_5 = self.kernel_txt_5(cap_i_4.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            cap_i_expand_5 = l2norm(cap_i_5.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_5.size(-1)
            cap_i_6 = self.kernel_txt_6(cap_i_5.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            cap_i_expand_6 = l2norm(cap_i_6.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_6.size(-1)
            cap_i_7 = self.kernel_txt_7(cap_i_6.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            cap_i_expand_7 = l2norm(cap_i_7.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_7.size(-1)
            cap_i_8 = self.kernel_txt_8(cap_i_7.reshape(-1, emb_size).unsqueeze(-2)).sum(1)
            cap_i_expand_8 = l2norm(cap_i_8.reshape(batch_size, n_word, -1), -1)

            return torch.cat(
                [cap_i, cap_i_expand_1, cap_i_expand_2, cap_i_expand_3, cap_i_expand_4, cap_i_expand_5, cap_i_expand_6,
                 cap_i_expand_7, cap_i_expand_8], -1)

        def forward(self, cap_i):
            batch_size, n_word, embed_size = cap_i.size(0), cap_i.size(1), cap_i.size(2)

            return self.get_text_levels(cap_i, batch_size, n_word)

else:

    #### Adaptive Dimensional Selective Mask
    class Image_levels(nn.Module):
        def __init__(self, opt):
            super(Image_levels, self).__init__()
            self.sub_space = opt.embed_size
            self.kernel_size = int(opt.kernel_size)
            self.out_channels = 2

            self.masks_1 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 1)),
                                              int(opt.embed_size / math.pow(self.kernel_size,
                                                                            0)))  # num_embedding, dims_input
            self.masks_2 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 2)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 1)))
            self.masks_3 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 3)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 2)))
            self.masks_4 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 4)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 3)))
            self.masks_5 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 5)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 4)))
            self.masks_6 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 6)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 5)))
            self.masks_7 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 7)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 6)))
            self.masks_8 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 8)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 7)))

        def get_image_levels(self, img_emb, batch_size, n_region):

            sub_space_index = torch.tensor(torch.linspace(0, 1024, steps=1024, dtype=torch.int)).cuda()
            Dim_learned_mask_1 = l2norm(
                self.masks_1(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 1))]), dim=-1)
            Dim_learned_mask_2 = l2norm(
                self.masks_2(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 2))]), dim=-1)
            Dim_learned_mask_3 = l2norm(
                self.masks_3(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 3))]), dim=-1)
            Dim_learned_mask_4 = l2norm(
                self.masks_4(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 4))]), dim=-1)
            Dim_learned_mask_5 = l2norm(
                self.masks_5(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 5))]), dim=-1)
            Dim_learned_mask_6 = l2norm(
                self.masks_6(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 6))]), dim=-1)
            Dim_learned_mask_7 = l2norm(
                self.masks_7(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 7))]), dim=-1)
            Dim_learned_mask_8 = l2norm(
                self.masks_8(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 8))]), dim=-1)

            if Dim_learned_mask_1.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_1.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_1.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_1 = (Dim_learned_mask_1 >= Dim_learned_range).float() * Dim_learned_mask_1

            if Dim_learned_mask_2.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_2.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_2.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_2 = (Dim_learned_mask_2 >= Dim_learned_range).float() * Dim_learned_mask_2

            if Dim_learned_mask_3.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_3.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_3.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_3 = (Dim_learned_mask_3 >= Dim_learned_range).float() * Dim_learned_mask_3

            if Dim_learned_mask_4.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_4.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_4.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_4 = (Dim_learned_mask_4 >= Dim_learned_range).float() * Dim_learned_mask_4

            if Dim_learned_mask_5.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_5.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_5.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_5 = (Dim_learned_mask_5 >= Dim_learned_range).float() * Dim_learned_mask_5

            if Dim_learned_mask_6.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_6.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_6.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_6 = (Dim_learned_mask_6 >= Dim_learned_range).float() * Dim_learned_mask_6

            if Dim_learned_mask_7.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_7.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_7.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_7 = (Dim_learned_mask_7 >= Dim_learned_range).float() * Dim_learned_mask_7

            if Dim_learned_mask_8.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_8.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_8.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_8 = (Dim_learned_mask_8 >= Dim_learned_range).float() * Dim_learned_mask_8

            img_emb_1 = img_emb.reshape(-1, self.sub_space) @ Dim_learned_mask_1.t()
            img_emb_1 = l2norm(img_emb_1.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_1.size(-1)
            img_emb_2 = img_emb_1.reshape(-1, emb_size) @ Dim_learned_mask_2.t()
            img_emb_2 = l2norm(img_emb_2.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_2.size(-1)
            img_emb_3 = img_emb_2.reshape(-1, emb_size) @ Dim_learned_mask_3.t()
            img_emb_3 = l2norm(img_emb_3.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_3.size(-1)
            img_emb_4 = img_emb_3.reshape(-1, emb_size) @ Dim_learned_mask_4.t()
            img_emb_4 = l2norm(img_emb_4.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_4.size(-1)
            img_emb_5 = img_emb_4.reshape(-1, emb_size) @ Dim_learned_mask_5.t()
            img_emb_5 = l2norm(img_emb_5.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_5.size(-1)
            img_emb_6 = img_emb_5.reshape(-1, emb_size) @ Dim_learned_mask_6.t()
            img_emb_6 = l2norm(img_emb_6.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_6.size(-1)
            img_emb_7 = img_emb_6.reshape(-1, emb_size) @ Dim_learned_mask_7.t()
            img_emb_7 = l2norm(img_emb_7.reshape(batch_size, n_region, -1), -1)

            emb_size = img_emb_7.size(-1)
            img_emb_8 = img_emb_7.reshape(-1, emb_size) @ Dim_learned_mask_8.t()
            img_emb_8 = l2norm(img_emb_8.reshape(batch_size, n_region, -1), -1)

            return torch.cat(
                [img_emb, img_emb_1, img_emb_2, img_emb_3, img_emb_4, img_emb_5, img_emb_6, img_emb_7, img_emb_8], -1)

        def forward(self, img_emb):

            batch_size, n_region, embed_size = img_emb.size(0), img_emb.size(1), img_emb.size(2)

            return self.get_image_levels(img_emb, batch_size, n_region)


    class Text_levels(nn.Module):

        def __init__(self, opt):
            super(Text_levels, self).__init__()
            self.sub_space = opt.embed_size
            self.kernel_size = int(opt.kernel_size)
            self.out_channels = 2

            self.masks_1 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 1)),
                                              int(opt.embed_size / math.pow(self.kernel_size,
                                                                            0)))  # num_embedding, dims_input
            self.masks_2 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 2)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 1)))
            self.masks_3 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 3)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 2)))
            self.masks_4 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 4)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 3)))
            self.masks_5 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 5)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 4)))
            self.masks_6 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 6)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 5)))
            self.masks_7 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 7)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 6)))
            self.masks_8 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 8)),
                                              int(opt.embed_size / math.pow(self.kernel_size, 7)))

        def get_text_levels(self, cap_i, batch_size, n_word):

            sub_space_index = torch.tensor(torch.linspace(0, 1024, steps=1024, dtype=torch.int)).cuda()
            Dim_learned_mask_1 = l2norm(
                self.masks_1(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 1))]), dim=-1)
            Dim_learned_mask_2 = l2norm(
                self.masks_2(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 2))]), dim=-1)
            Dim_learned_mask_3 = l2norm(
                self.masks_3(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 3))]), dim=-1)
            Dim_learned_mask_4 = l2norm(
                self.masks_4(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 4))]), dim=-1)
            Dim_learned_mask_5 = l2norm(
                self.masks_5(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 5))]), dim=-1)
            Dim_learned_mask_6 = l2norm(
                self.masks_6(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 6))]), dim=-1)
            Dim_learned_mask_7 = l2norm(
                self.masks_7(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 7))]), dim=-1)
            Dim_learned_mask_8 = l2norm(
                self.masks_8(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 8))]), dim=-1)

            if Dim_learned_mask_1.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_1.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_1.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_1 = (Dim_learned_mask_1 >= Dim_learned_range).float() * Dim_learned_mask_1

            if Dim_learned_mask_2.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_2.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_2.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_2 = (Dim_learned_mask_2 >= Dim_learned_range).float() * Dim_learned_mask_2

            if Dim_learned_mask_3.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_3.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_3.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_3 = (Dim_learned_mask_3 >= Dim_learned_range).float() * Dim_learned_mask_3

            if Dim_learned_mask_4.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_4.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_4.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_4 = (Dim_learned_mask_4 >= Dim_learned_range).float() * Dim_learned_mask_4

            if Dim_learned_mask_5.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_5.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_5.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_5 = (Dim_learned_mask_5 >= Dim_learned_range).float() * Dim_learned_mask_5

            if Dim_learned_mask_6.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_6.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_6.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_6 = (Dim_learned_mask_6 >= Dim_learned_range).float() * Dim_learned_mask_6

            if Dim_learned_mask_7.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_7.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_7.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_7 = (Dim_learned_mask_7 >= Dim_learned_range).float() * Dim_learned_mask_7

            if Dim_learned_mask_8.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_8.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_8.sort(1, descending=True)[0][:, select_nums - 1].unsqueeze(-1)
            Dim_learned_mask_8 = (Dim_learned_mask_8 >= Dim_learned_range).float() * Dim_learned_mask_8

            cap_i_1 = cap_i.reshape(-1, self.sub_space) @ Dim_learned_mask_1.t()
            cap_i_expand_1 = l2norm(cap_i_1.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_1.size(-1)
            cap_i_2 = cap_i_1.reshape(-1, emb_size) @ Dim_learned_mask_2.t()
            cap_i_expand_2 = l2norm(cap_i_2.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_2.size(-1)
            cap_i_3 = cap_i_2.reshape(-1, emb_size) @ Dim_learned_mask_3.t()
            cap_i_expand_3 = l2norm(cap_i_3.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_3.size(-1)
            cap_i_4 = cap_i_3.reshape(-1, emb_size) @ Dim_learned_mask_4.t()
            cap_i_expand_4 = l2norm(cap_i_4.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_4.size(-1)
            cap_i_5 = cap_i_4.reshape(-1, emb_size) @ Dim_learned_mask_5.t()
            cap_i_expand_5 = l2norm(cap_i_5.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_5.size(-1)
            cap_i_6 = cap_i_5.reshape(-1, emb_size) @ Dim_learned_mask_6.t()
            cap_i_expand_6 = l2norm(cap_i_6.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_6.size(-1)
            cap_i_7 = cap_i_6.reshape(-1, emb_size) @ Dim_learned_mask_7.t()
            cap_i_expand_7 = l2norm(cap_i_7.reshape(batch_size, n_word, -1), -1)

            emb_size = cap_i_expand_7.size(-1)
            cap_i_8 = cap_i_7.reshape(-1, emb_size) @ Dim_learned_mask_8.t()
            cap_i_expand_8 = l2norm(cap_i_8.reshape(batch_size, n_word, -1), -1)

            return torch.cat(
                [cap_i, cap_i_expand_1, cap_i_expand_2, cap_i_expand_3, cap_i_expand_4, cap_i_expand_5, cap_i_expand_6,
                 cap_i_expand_7, cap_i_expand_8], -1)

        def forward(self, cap_i):

            batch_size, n_word, embed_size = cap_i.size(0), cap_i.size(1), cap_i.size(2)

            return self.get_text_levels(cap_i, batch_size, n_word)


class Image_Text_Encoders(nn.Module):
    def __init__(self, opt):

        super(Image_Text_Encoders, self).__init__()
        self.text_levels = Text_levels(opt)
        self.image_levels = Image_levels(opt)

    def forward(self, images, captions, return_type):

        if return_type == 'image':
            img_embs = self.image_levels(images)
            return img_embs
        else:
            cap_embs = self.text_levels(captions)
            return cap_embs


class Image_Text_Processing(nn.Module):

    def __init__(self, opt):
        super(Image_Text_Processing, self).__init__()
        self.encoders_1 = Image_Text_Encoders(opt)

    def forward(self, images, captions):
        image_processed = self.encoders_1(images, captions, 'image')
        text_processed = self.encoders_1(images, captions, 'text')

        return image_processed, text_processed


class sims_claculator(nn.Module):
    def __init__(self, opt):
        super(sims_claculator, self).__init__()
        self.sub_space = opt.embed_size
        self.kernel_size = int(opt.kernel_size)

        self.opt = opt
        self.sim_eval = nn.Linear(9, 1, bias=False)
        self.temp_scale = nn.Linear(1, 1, bias=False)
        self.temp_scale_1 = nn.Linear(1, 1, bias=False)
        self.temp_scale_2 = nn.Linear(1, 1, bias=False)

        self.masks_0 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 0)),
                                          int(opt.embed_size / math.pow(self.kernel_size, 0)))
        self.masks_1 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 1)),
                                          int(opt.embed_size / math.pow(self.kernel_size, 1)))
        self.masks_2 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 2)),
                                          int(opt.embed_size / math.pow(self.kernel_size, 2)))
        self.masks_3 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 3)),
                                          int(opt.embed_size / math.pow(self.kernel_size, 3)))
        self.masks_4 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 4)),
                                          int(opt.embed_size / math.pow(self.kernel_size, 4)))
        self.masks_5 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 5)),
                                          int(opt.embed_size / math.pow(self.kernel_size, 5)))
        self.masks_6 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 6)),
                                          int(opt.embed_size / math.pow(self.kernel_size, 6)))
        self.masks_7 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 7)),
                                          int(opt.embed_size / math.pow(self.kernel_size, 7)))
        self.masks_8 = torch.nn.Embedding(int(opt.embed_size / math.pow(self.kernel_size, 8)),
                                          int(opt.embed_size / math.pow(self.kernel_size, 8)))

        self.lynorm_0 = nn.LayerNorm(
            [int(opt.embed_size / math.pow(self.kernel_size, 0)), int(opt.embed_size / math.pow(self.kernel_size, 0))],
            eps=1e-08, elementwise_affine=True)
        self.lynorm_1 = nn.LayerNorm(
            [int(opt.embed_size / math.pow(self.kernel_size, 1)), int(opt.embed_size / math.pow(self.kernel_size, 1))],
            eps=1e-08, elementwise_affine=True)
        self.lynorm_2 = nn.LayerNorm(
            [int(opt.embed_size / math.pow(self.kernel_size, 2)), int(opt.embed_size / math.pow(self.kernel_size, 2))],
            eps=1e-08, elementwise_affine=True)
        self.lynorm_3 = nn.LayerNorm(
            [int(opt.embed_size / math.pow(self.kernel_size, 3)), int(opt.embed_size / math.pow(self.kernel_size, 3))],
            eps=1e-08, elementwise_affine=True)
        self.lynorm_4 = nn.LayerNorm(
            [int(opt.embed_size / math.pow(self.kernel_size, 4)), int(opt.embed_size / math.pow(self.kernel_size, 4))],
            eps=1e-08, elementwise_affine=True)
        self.lynorm_5 = nn.LayerNorm(
            [int(opt.embed_size / math.pow(self.kernel_size, 5)), int(opt.embed_size / math.pow(self.kernel_size, 5))],
            eps=1e-08, elementwise_affine=True)
        self.lynorm_6 = nn.LayerNorm(
            [int(opt.embed_size / math.pow(self.kernel_size, 6)), int(opt.embed_size / math.pow(self.kernel_size, 6))],
            eps=1e-08, elementwise_affine=True)
        self.lynorm_7 = nn.LayerNorm(
            [int(opt.embed_size / math.pow(self.kernel_size, 7)), int(opt.embed_size / math.pow(self.kernel_size, 7))],
            eps=1e-08, elementwise_affine=True)
        self.lynorm_8 = nn.LayerNorm(
            [int(opt.embed_size / math.pow(self.kernel_size, 8)), int(opt.embed_size / math.pow(self.kernel_size, 8))],
            eps=1e-08, elementwise_affine=True)

        self.list_length = [int(opt.embed_size / math.pow(self.kernel_size, 0)),
                            int(opt.embed_size / math.pow(self.kernel_size, 1)),
                            int(opt.embed_size / math.pow(self.kernel_size, 2)),
                            int(opt.embed_size / math.pow(self.kernel_size, 3)),
                            int(opt.embed_size / math.pow(self.kernel_size, 4)),
                            int(opt.embed_size / math.pow(self.kernel_size, 5)),
                            int(opt.embed_size / math.pow(self.kernel_size, 6)),
                            int(opt.embed_size / math.pow(self.kernel_size, 7)),
                            int(opt.embed_size / math.pow(self.kernel_size, 8))]

        self.init_weights()

    def init_weights(self):
        self.temp_scale.weight.data.fill_(np.log(1 / 0.07))
        self.sim_eval.weight.data.fill_(0.1)
        self.temp_scale_1.weight.data.fill_(0)
        self.temp_scale_2.weight.data.fill_(3)

    def get_weighted_features(self, attn, smooth):
        # --> (batch, sourceL, queryL)
        attnT = torch.transpose(attn, 1, 2).contiguous()
        attn = nn.LeakyReLU(0.1)(attnT)
        attn = l2norm(attn, 2)
        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch, queryL, sourceL
        attn = F.softmax(attn * smooth, dim=2)

        return attn

    def get_sims_levels(self, X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8,
                        Y_0, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8,
                        D_0, D_1, D_2, D_3, D_4, D_5, D_6, D_7, D_8):
        attn_0 = (X_0 @ D_0 @ Y_0.transpose(1, 2))
        attn_1 = (X_1 @ D_1 @ Y_1.transpose(1, 2))
        attn_2 = (X_2 @ D_2 @ Y_2.transpose(1, 2))
        attn_3 = (X_3 @ D_3 @ Y_3.transpose(1, 2))
        attn_4 = (X_4 @ D_4 @ Y_4.transpose(1, 2))
        attn_5 = (X_5 @ D_5 @ Y_5.transpose(1, 2))
        attn_6 = (X_6 @ D_6 @ Y_6.transpose(1, 2))
        attn_7 = (X_7 @ D_7 @ Y_7.transpose(1, 2))
        attn_8 = (X_8 @ D_8 @ Y_8.transpose(1, 2))

        attn = attn_0 + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6 + attn_7 + attn_8

        return attn

    def forward(self, img_emb, cap_emb, cap_lens):

        n_caption = cap_emb.size(0)
        sim_all = []
        batch_size, n_region, embed_size = img_emb.size(0), img_emb.size(1), img_emb.size(2)
        sub_space_index = torch.tensor(torch.linspace(0, self.sub_space, steps=self.sub_space, dtype=torch.int)).cuda()
        smooth = torch.exp(self.temp_scale.weight)

        sigma_ = self.temp_scale_1.weight
        lambda_ = torch.exp(self.temp_scale_2.weight)
        threshold = (torch.abs(self.sim_eval.weight).max() - torch.abs(
            self.sim_eval.weight).min()) * sigma_ + torch.abs(self.sim_eval.weight).min()
        if not heuristic_strategy:
            lambda_ = 0

        weight_0 = (torch.exp((torch.abs(self.sim_eval.weight[0, 0]) - threshold) * lambda_) * self.sim_eval.weight[
            0, 0])
        weight_1 = (torch.exp((torch.abs(self.sim_eval.weight[0, 1]) - threshold) * lambda_) * self.sim_eval.weight[
            0, 1])
        weight_2 = (torch.exp((torch.abs(self.sim_eval.weight[0, 2]) - threshold) * lambda_) * self.sim_eval.weight[
            0, 2])
        weight_3 = (torch.exp((torch.abs(self.sim_eval.weight[0, 3]) - threshold) * lambda_) * self.sim_eval.weight[
            0, 3])
        weight_4 = (torch.exp((torch.abs(self.sim_eval.weight[0, 4]) - threshold) * lambda_) * self.sim_eval.weight[
            0, 4])
        weight_5 = (torch.exp((torch.abs(self.sim_eval.weight[0, 5]) - threshold) * lambda_) * self.sim_eval.weight[
            0, 5])
        weight_6 = (torch.exp((torch.abs(self.sim_eval.weight[0, 6]) - threshold) * lambda_) * self.sim_eval.weight[
            0, 6])
        weight_7 = (torch.exp((torch.abs(self.sim_eval.weight[0, 7]) - threshold) * lambda_) * self.sim_eval.weight[
            0, 7])
        weight_8 = (torch.exp((torch.abs(self.sim_eval.weight[0, 8]) - threshold) * lambda_) * self.sim_eval.weight[
            0, 8])

        Dim_learned_mask_0 = self.lynorm_0(
            self.masks_0(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 0))])) * weight_0
        Dim_learned_mask_1 = self.lynorm_1(
            self.masks_1(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 1))])) * weight_1
        Dim_learned_mask_2 = self.lynorm_2(
            self.masks_2(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 2))])) * weight_2
        Dim_learned_mask_3 = self.lynorm_3(
            self.masks_3(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 3))])) * weight_3
        Dim_learned_mask_4 = self.lynorm_4(
            self.masks_4(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 4))])) * weight_4
        Dim_learned_mask_5 = self.lynorm_5(
            self.masks_5(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 5))])) * weight_5
        Dim_learned_mask_6 = self.lynorm_6(
            self.masks_6(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 6))])) * weight_6
        Dim_learned_mask_7 = self.lynorm_7(
            self.masks_7(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 7))])) * weight_7
        Dim_learned_mask_8 = self.lynorm_8(
            self.masks_8(sub_space_index[:int(self.sub_space / math.pow(self.kernel_size, 8))])) * weight_8

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]

            ## ------------------------------------------------------------------------------------------------------------------------
            # attention

            attn = self.get_sims_levels(
                cap_emb[i, :n_word, :sum(self.list_length[:1])].unsqueeze(0).repeat(batch_size, 1, 1),
                cap_emb[i, :n_word, sum(self.list_length[:1]):sum(self.list_length[:2])].unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1),
                cap_emb[i, :n_word, sum(self.list_length[:2]):sum(self.list_length[:3])].unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1),
                cap_emb[i, :n_word, sum(self.list_length[:3]):sum(self.list_length[:4])].unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1),
                cap_emb[i, :n_word, sum(self.list_length[:4]):sum(self.list_length[:5])].unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1),
                cap_emb[i, :n_word, sum(self.list_length[:5]):sum(self.list_length[:6])].unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1),
                cap_emb[i, :n_word, sum(self.list_length[:6]):sum(self.list_length[:7])].unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1),
                cap_emb[i, :n_word, sum(self.list_length[:7]):sum(self.list_length[:8])].unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1),
                cap_emb[i, :n_word, sum(self.list_length[:8]):sum(self.list_length[:9])].unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1),
                img_emb[:, :, :sum(self.list_length[:1])],
                img_emb[:, :, sum(self.list_length[:1]):sum(self.list_length[:2])],
                img_emb[:, :, sum(self.list_length[:2]):sum(self.list_length[:3])],
                img_emb[:, :, sum(self.list_length[:3]):sum(self.list_length[:4])],
                img_emb[:, :, sum(self.list_length[:4]):sum(self.list_length[:5])],
                img_emb[:, :, sum(self.list_length[:5]):sum(self.list_length[:6])],
                img_emb[:, :, sum(self.list_length[:6]):sum(self.list_length[:7])],
                img_emb[:, :, sum(self.list_length[:7]):sum(self.list_length[:8])],
                img_emb[:, :, sum(self.list_length[:8]):sum(self.list_length[:9])],
                Dim_learned_mask_0, Dim_learned_mask_1, Dim_learned_mask_2, Dim_learned_mask_3, Dim_learned_mask_4,
                Dim_learned_mask_5, Dim_learned_mask_6, Dim_learned_mask_7, Dim_learned_mask_8)

            ##################################################################################################
            # --> (batch, sourceL, queryL)
            attnT = torch.transpose(attn, 1, 2).contiguous()
            attn_t2i_weight = self.get_weighted_features(torch.tanh(attn), smooth)
            sims_t2i = attn.mul(attn_t2i_weight).sum(-1).mean(dim=1, keepdim=True)
            ##################################################################################################
            attn_i2t_weight = self.get_weighted_features(torch.tanh(attnT), smooth)
            sims_i2t = attnT.mul(attn_i2t_weight).sum(-1).mean(dim=1, keepdim=True)
            ##################################################################################################

            sims = sims_t2i + sims_i2t
            sim_all.append(sims)

        sim_all = torch.cat(sim_all, 1)
        return sim_all


class Sims_Measuring(nn.Module):
    def __init__(self, opt):
        super(Sims_Measuring, self).__init__()

        self.calculator_1 = sims_claculator(opt)

    def forward(self, img_embs, cap_embs, lengths):
        sims = self.calculator_1(img_embs, cap_embs, lengths)

        return sims


class Sim_vec(nn.Module):

    def __init__(self, embed_size, opt):
        super(Sim_vec, self).__init__()
        self.plus_encoder = Image_Text_Processing(opt)
        self.sims = Sims_Measuring(opt)

    def forward(self, img_emb, cap_emb, cap_lens, is_Train):
        region_features, word_features = self.plus_encoder(img_emb, cap_emb)
        sims = self.sims(region_features, word_features, cap_lens)

        return sims, sims


class VSE(nn.Module):
    def __init__(self, args):
        super(VSE, self).__init__()
        self.args = args
        self.img_enc_type = args.img_enc_type
        self.txt_enc_type = args.txt_enc_type
        self.margin = args.margin
        self.sim_vec = Sim_vec(args.embed_size, args)
        # visiual encoder
        if self.img_enc_type == 'project':
            self.img_enc = EncoderImage_ProjFeature(args.img_dim, args.emb_size,
                                                    precomp_enc_type=args.precomp_enc_type,
                                                    no_imgnorm=args.no_imgnorm)
            self.proj_params = list(self.img_enc.parameters()) + \
                               list(self.sim_vec.parameters())
        elif self.img_enc_type == 'butd':

            backbone_cnn = ResNetFeatureExtractor(self.args.backbone_weights_path,
                                                  fixed_blocks=2)
            backbone_cnn.load_state_dict(
                torch.load(f'./data/weights/{args.data_name}_{args.txt_enc_type}_backbone.pth'))
            self.img_enc = EncoderImage_BUTD(backbone_cnn, self.args.img_dim,
                                             self.args.emb_size,
                                             precomp_enc_type='backbone')
            self.backbone_params = list(self.img_enc.backbone.base.parameters()) + \
                                   list(self.img_enc.backbone.top.parameters())
            self.proj_params = list(self.img_enc.img_proj.parameters()) + \
                               list(self.sim_vec.parameters())
        else:
            self.img_enc = EncoderImage_VIT(args.vit_type, args.emb_size,
                                            args.precomp_enc_type, args.no_imgnorm)
            self.vit_params = list(self.img_enc.vit_model.embeddings.parameters()) + \
                              list(self.img_enc.vit_model.encoder.parameters())
            self.proj_params = list(self.img_enc.vit_model.pooler.parameters()) + \
                               list(self.img_enc.img_proj.parameters()) + \
                               list(self.sim_vec.parameters())

        # textual encoder
        if self.txt_enc_type == 'rnn':
            self.txt_enc = EncoderText_BiGRU(args.vocab_size, args.word_dim,
                                             args.emb_size, args.num_layers,
                                             args.use_bigru, args.no_txtnorm)
            self.rnn_params = list(self.txt_enc.parameters())

        elif self.txt_enc_type == 'bert':
            self.txt_enc = EncoderText_BERT(args.bert_type, args.emb_size,
                                            args, args.no_txtnorm)
            # self.txt_enc_bert_params = list(self.txt_enc.bert.parameters())
            self.txt_enc_bert_params = list(self.txt_enc.bert_1.parameters()) + list(self.txt_enc.bert_2.parameters())

            bert_params_ptr = [p.data_ptr() for p in self.txt_enc_bert_params]
            self.txt_enc_nobert_params = list()
            for p in list(self.txt_enc.parameters()):
                if p.data_ptr() not in bert_params_ptr:
                    self.txt_enc_nobert_params.append(p)
        self.txt_enc_params = list(self.txt_enc.parameters())
        self.img_enc_params = list(self.img_enc.parameters())
        self.sim_vec_params = list(self.sim_vec.parameters())
        self.enc_params = self.img_enc_params + self.txt_enc_params + self.sim_vec_params

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.sim_vec = nn.DataParallel(self.sim_vec)
        self.data_parallel = True

    def freeze_backbone(self):
        if self.img_enc_type == 'butd':
            self.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if self.img_enc_type == 'butd':
            self.img_enc.unfreeze_backbone(fixed_blocks)

    def forward_sim(self, img_emb, cap_emb, cap_lens):
        is_Train = True
        sim_all, L1 = self.sim_vec(img_emb, cap_emb, cap_lens, is_Train)

        return sim_all, L1

    def forward_sim_test(self, img_emb, cap_emb, cap_lens):
        is_Train = False
        sim_all, L1 = self.sim_vec(img_emb, cap_emb, cap_lens, is_Train)

        return sim_all, L1
    def forward(self, img, img_lengths, txt, txt_lengths, et_txts, et_txt_lengths, sub_txts, rel_txts, obj_txts):
        # img = [batch_size, n_region, img_dim]
        # txt = [batch_size, seq_len]

        # img_emb = [batch_size, n_region, emb_size]
        # txt_emb  = [batch_size, seq_len, emb_size]
        img_emb, fg_img_emb = self.img_enc(img, img_lengths)
        txt_emb, fg_txt_emb, txt_lens = self.txt_enc(txt, txt_lengths, et_txts, et_txt_lengths, sub_txts, rel_txts, obj_txts)

        return img_emb, txt_emb, fg_img_emb, fg_txt_emb, txt_lens
