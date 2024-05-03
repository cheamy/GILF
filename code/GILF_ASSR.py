import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

class Net(nn.Module):
    def __init__(self, angular_in, angular_out, factor):
        super(Net, self).__init__()
        channels = 64
        self.angRes = angular_in
        self.angRes_out = angular_out
        self.factor = factor
        self.channel = 64
        self.FeaExtract = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.altblock = nn.Sequential(
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
            AltFilter(self.angRes, channels),
        )
        self.epiAngRebuild = EpiAngRebuild(self.angRes, self.angRes_out, channels, feat_unfold=False)
        self.epiFeatureRebuild = EpiFeatureRebuild(self.angRes_out,self.factor, channels, feat_unfold=False)
        self.DownSample = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels // 4, channels // 4 // 4, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels // 4 // 4, 1, kernel_size=1, stride=1,
                      padding=0, bias=False),
        )


    def forward(self, x):

        x_mv = LFsplit(x, self.angRes)
        buffer = self.FeaExtract(x_mv)
        buffer = self.conv_init(buffer) + buffer

        buffer = self.altblock(buffer) + buffer

        buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=self.angRes, v=self.angRes)
        buffer = self.epiAngRebuild(buffer)
        buffer = self.epiFeatureRebuild(buffer)
        buffer = buffer[:, :, ::2, ::2]


        buffer = rearrange(buffer, 'b c u v h w -> b c (u v) h w')
        b, c, n, h, w = buffer.shape
        buffer = self.DownSample(buffer).view(b, 1, self.angRes * self.angRes, h, w)  # n == angRes * angRes
        out = FormOutput(buffer)
        return out



class AltFilter(nn.Module):
    def __init__(self, angRes, channels):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.epi_trans = BasicTrans(channels, channels*2)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, buffer):
        shortcut = buffer
        [_, _, _, h, w] = buffer.size()
        self.epi_trans.mask_field = [self.angRes * 2, 11]

        # Horizontal
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        # Vertical
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        return buffer

class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int):
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()
        attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)

        epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st.permute(0, 2, 1, 3, 4)

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class EpiFeatureRebuild(nn.Module):
    def __init__(self, angRes, factor, channels, feat_unfold=True, local_ensemble=False, cell_decode=False):
        super().__init__()
        self.angRes = angRes
        self.factor = factor
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        imnet_in_dim = channels
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=channels, hidden_list=[256, 256, 256, 256])

    def query_feature(self, Feature, coord, cell=None):
        feat = Feature

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device)\
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t

            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret



    def query_Epi(self, epi):
        buh, c, v, w = epi.shape

        # V x W --> outV x facor*W
        coord = make_coord([self.angRes, self.factor*w]).to(epi.device)
        coord = coord.unsqueeze(0)
        coord = coord.expand(epi.shape[0], w * self.factor * self.angRes, 2)
        output_epi = self.query_feature(epi, coord, cell=None).permute(0, 2, 1) \
            .view(epi.shape[0], -1, self.angRes, self.factor*w)

        # buh, c, angRes_out, w
        return output_epi

    def forward(self, x):
        batch_size, channle, u, v, h, w = x.shape

        # u x v x H x W --> u x outv x H x facor*W
        horizontal_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b u h) c v w'))
        x = rearrange(horizontal_x, '(b u h) c v w -> b c u v h w', b=batch_size, u=u)
        # u x v x H x W --> outu x outv x facor*H x factor*W
        vertical_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b v w) c u h'))
        output = rearrange(vertical_x, '(b v w) c u h -> b c u v h w', b=batch_size, v=self.angRes)

        return output

class EpiAngRebuild(nn.Module):
    def __init__(self, angRes_in, angRes_out, channels, feat_unfold=True, local_ensemble=False, cell_decode=False):
        super().__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        imnet_in_dim = channels
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=channels, hidden_list=[256, 256, 256, 256])

    def query_feature(self, Feature, coord, cell=None):
        feat = Feature

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t

            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def query_Epi(self, epi):
        buh, c, v, w = epi.shape

        # 2 x W --> 7 x W
        coord = make_coord([self.angRes_out, w]).cuda() \
            .unsqueeze(0).expand(epi.shape[0], w * self.angRes_out, 2)
        output_epi = self.query_feature(epi, coord, cell=None).permute(0, 2, 1) \
            .view(epi.shape[0], -1, self.angRes_out, w)

        # buh, c, angRes_out, w
        return output_epi

    def forward(self, x):
        batch_size, channle, u, v, h, w = x.shape

        # 2 x 2 x H x W --> 2 x 7 x H x W
        horizontal_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b u h) c v w'))
        x = rearrange(horizontal_x, '(b u h) c v w -> b c u v h w', b=batch_size, h=h)
        # 2 x 7 x H x W --> 7 x 7 x H x W
        vertical_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b v w) c u h'))
        output = rearrange(vertical_x, '(b v w) c u h -> b c u v h w', b=batch_size, w=w)

        return output


def FormOutput(x_sv):
    x_sv = x_sv.permute(0, 2, 1, 3, 4)
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n + 1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk + 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'


    net = Net(5, 9, 2).cuda()

    input = torch.randn(1, 1, 60, 60).cuda()
    out = net(input)
    print(out.shape)