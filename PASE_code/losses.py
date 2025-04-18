import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable




class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0.2, max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, scores):
        # compute image-sentence score matrix
        # scores = get_sim(im, s)
        # diagonal = scores.diag().view(im.size(0), 1)

        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities



# class ContrastiveLoss(nn.Module):
#     def __init__(self, args, device, lamda=0.01, acc_mode='acc'):
#         super(ContrastiveLoss, self).__init__()
#         self.args = args
#         self.device = device
#         self.lamda = lamda
#         self.acc_mode = acc_mode
#         self.scale = args.scale
#         self.const_num = args.const_num
#         self.loss_func = F.cross_entropy
#
#     def align(self, d):
#         return d.diag().mean()
#
#     def uniform(self, d):
#         d = d - torch.eye(d.shape[0]).to(d.device) * d
#         return d.mul(-1).exp().mean()
#
#     def ADCTO(self, d, query, keys, acc_mode, scale, num_negs=None):
#         """
#             d: [batch_size, batch_size] <--> query x keys
#             querys: [batch_size, emb_size]
#             keys: [batch_size, emb_size]
#         """
#         # positive_logit = [batch_size, 1]
#         positive_logit = torch.diag(d).unsqueeze(-1)
#         if acc_mode == 'acc':
#             align = self.align(d).detach().data.cpu().numpy()
#             uniform = self.uniform(d).detach().data.cpu().numpy()
#             tmp = align + uniform
#             num_negs = max(int(np.cos(tmp ** scale) * d.shape[0]) + 1, 1)
#             num_negs = min(num_negs, d.shape[0] - 1)
#         elif acc_mode == 'constant':
#             num_negs = 1 if not num_negs else num_negs
#         elif acc_mode == 'random':
#             num_negs = np.random.randint(1, d.shape[0])
#
#         # mask = [batch_size, batch_size]
#         mask = (torch.eye(d.size(0)) > .5).to(d.device)
#         d = d.masked_fill(mask, 0)
#
#         # sorted_idx = [batch_size, num_negs, emb_size]
#         _, sorted_idx = torch.sort(d, dim=-1, descending=True)
#         sorted_idx = sorted_idx[:, :num_negs]
#         sorted_idx = sorted_idx.unsqueeze(-1).repeat(1, 1, keys.shape[-1])
#
#         # negative_keys = [batch_size, num_negs, emb_size]
#         negative_keys = torch.gather(keys.repeat(d.shape[0], 1).view(d.shape[0], d.shape[0], -1),
#                                      1, sorted_idx)
#
#         # negative_logits = [batch_size, num_negs]
#         negative_logits = torch.matmul(query.unsqueeze(1),
#                                        negative_keys.transpose(-2, -1)).squeeze(1) + self.args.margin
#
#         logits = torch.cat([positive_logit, negative_logits], dim=1)
#         labels = torch.zeros(len(logits), dtype=torch.long, device=d.device)
#
#         return self.loss_func(logits / self.lamda, labels), num_negs
#
#     def forward(self, img, txt, txt_lens):
#         # cos_sim = [batch_size, batch_size]
#         cos_sim = txt.mm(img.t())
#         t2i_loss, t2i_num_negs = self.ADCTO(cos_sim, txt, img, self.acc_mode,
#                                             self.scale, self.const_num)
#         i2t_loss, i2t_num_negs = self.ADCTO(cos_sim.t(), img, txt, self.acc_mode,
#                                             self.scale, self.const_num)
#         return (t2i_loss + i2t_loss) / 2, [t2i_num_negs, i2t_num_negs]


# class TripletLoss(nn.Module):
#     def __init__(self, args, margin=0, max_violation=False):
#         super(TripletLoss, self).__init__()
#         self.args = args
#         self.margin = margin
#         self.max_violation = max_violation
#
#     def forward(self, img, txt, txt_lens):
#         scores = img.mm(txt.t())
#
#         # scores = [batch_size, batch_size]
#         # diagonal = [batch_size, 1]
#         diagonal = scores.diag().view(img.size(0), 1)
#         d1 = diagonal.expand_as(scores)
#         d2 = diagonal.t().expand_as(scores)
#
#         cost_s = (self.margin + scores - d1).clamp(min=0)
#         cost_im = (self.margin + scores - d2).clamp(min=0)
#         mask = torch.eye(scores.size(0)) > .5
#         I = Variable(mask)
#         if torch.cuda.is_available():
#             I = I.cuda()
#         cost_s = cost_s.masked_fill_(I, 0)
#         cost_im = cost_im.masked_fill_(I, 0)
#
#         # keep the maximum violating negative for each query
#         if self.max_violation:
#             cost_s = cost_s.max(1)[0]
#             cost_im = cost_im.max(0)[0]
#         return cost_s.sum() + cost_im.sum(), [1, 1]
def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities
def pos_neg_mask(labels):
    # 其中第i行和第j列的元素为True表示标签labels[i]和labels[j]相等
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    return pos_mask, neg_mask


def pos_neg_mask_xy(labels_col, labels_row):

    pos_mask = (labels_row.unsqueeze(0) == labels_col.unsqueeze(1))
    neg_mask = (labels_row.unsqueeze(0) != labels_col.unsqueeze(1))

    return pos_mask, neg_mask

class TripletLoss(nn.Module):

    def __init__(self, opt=None, margin=0.2, ):
        super().__init__()

        self.opt = opt
        self.margin = margin

        self.cut_off = 0.5
        self.d = 512

        if opt.data_name == 'coco':
            self.nonzero_loss_cutoff = 1.9
        else:
            self.nonzero_loss_cutoff = 1.7

    def forward(self, im, s, img_ids, sim_mat=None):
        if sim_mat is None:
            sim_mat = get_sim(im, s)
        img_ids = img_ids.cuda()

        if im.size(0) == s.size(0):
            pos_mask, neg_mask = pos_neg_mask(img_ids)
        else:
            pos_mask, neg_mask = pos_neg_mask_xy(torch.unique(img_ids), img_ids)

        loss_im = self.loss_forward(sim_mat, pos_mask, neg_mask)
        loss_s = self.loss_forward(sim_mat.t(), pos_mask.t(), neg_mask.t())

        loss = loss_im + loss_s

        return loss, [1, 1]

    def loss_forward(self, sim_mat, pos_mask, neg_mask):

        pos_pair_idx = pos_mask.nonzero(as_tuple=False)
        anchor_idx = pos_pair_idx[:, 0]
        pos_idx = pos_pair_idx[:, 1]

        dist = (2 - 2 * sim_mat).sqrt()
        dist = dist.clamp(min=self.cut_off)

        log_weight = (2.0 - self.d) * dist.log() - ((self.d - 3.0) / 2.0) * (1.0 - 0.25 * (dist * dist)).log()
        inf_or_nan = torch.isinf(log_weight) | torch.isnan(log_weight)

        log_weight = log_weight * neg_mask
        log_weight[inf_or_nan] = 0.

        weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
        weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float()

        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
        weight = weight[anchor_idx]

        # maybe not exist
        try:
            neg_idx = torch.multinomial(weight, 1).squeeze(1)
        except Exception:
            return torch.zeros([], requires_grad=True, device=sim_mat.device)

        s_ap = sim_mat[anchor_idx, pos_idx]
        s_an = sim_mat[anchor_idx, neg_idx]

        loss = F.relu(self.margin + s_an - s_ap)
        loss = loss.sum()

        return loss

class InfoNCELoss(nn.Module):
    """
    Compute InfoNCELoss loss
    """

    def __init__(self, temperature=0.01, margin=0.2):
        super(InfoNCELoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, img, txt, txt_lens):
        sims = txt.mm(img.t())

        ## cost of image retrieval
        img_ret = sims - sims.diag().expand_as(sims).t() + self.margin
        img_ret[torch.eye(sims.size(0)) > .5] = 0
        cost_im = torch.log(torch.sum(torch.exp(img_ret / self.temperature), dim=1))

        ## cost of text retrieval
        txt_ret = sims - sims.diag().expand_as(sims) + self.margin
        txt_ret[torch.eye(sims.size(0)) > .5] = 0
        cost_s = torch.log(torch.sum(torch.exp(txt_ret / self.temperature), dim=0))

        return cost_s.mean() + cost_im.mean()

    def max_violation_on(self):
        return

    def max_violation_off(self):
        return
