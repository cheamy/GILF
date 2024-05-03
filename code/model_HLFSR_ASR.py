import torch
import torch.nn as nn
from common import * 
import torch.nn.functional as F
import matplotlib as plt
from einops import rearrange

class HLFSR_ASR(nn.Module):
	def __init__(self, angRes_in, angRes_out, n_blocks, channels):
		super(HLFSR_ASR, self).__init__()
		
		self.angRes_in = angRes_in
		self.angRes_out = angRes_out
		self.channels = channels

		self.n_blocks = n_blocks
		
	   
		
		self.HFEM_1 = HFEM(angRes_in, n_blocks, channels,first=True)
		self.HFEM_2 = HFEM(angRes_in, n_blocks, channels,first=False)
		self.HFEM_3 = HFEM(angRes_in, n_blocks, channels,first=False)
		self.HFEM_4 = HFEM(angRes_in, n_blocks, channels,first=False)
		self.HFEM_5 = HFEM(angRes_in, n_blocks, channels,first=False)



		self.epiFeatureRebuild = EpiFeatureRebuild(self.angRes_in, self.angRes_out, self.channels, feat_unfold=False)

		self.DownSample = nn.Sequential(
			nn.Conv3d(self.channels, self.channels // 4, kernel_size=1,
					  stride=1, padding=0, bias=False),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv3d(self.channels // 4, self.channels // 4 // 4, kernel_size=1,
					  stride=1, padding=0, bias=False),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv3d(self.channels // 4 // 4, 1, kernel_size=1, stride=1,
					  padding=0, bias=False),
		)


	def forward(self, x):
		
		#Reshaping 
		x = SAI2MacPI(x,self.angRes_in)
		
		HFEM_1 = self.HFEM_1(x)
		HFEM_2 = self.HFEM_2(HFEM_1)
		HFEM_3 = self.HFEM_3(HFEM_2)
		HFEM_4 = self.HFEM_4(HFEM_3)
		HFEM_5 = self.HFEM_5(HFEM_4)

		x_out = MI2StackView(HFEM_5,self.angRes_in) #(b, c*u*v, h, w)


		x_out = rearrange(x_out, 'b (c u v) h w -> b c u v h w', c=self.channels, u=self.angRes_in, v=self.angRes_in)
		x_out = self.epiFeatureRebuild(x_out)
		x_out = rearrange(x_out, 'b c u v h w -> b c (u v) h w')
		b, c, n, h, w = x_out.shape

		buffer = self.DownSample(x_out).view(b, 1, self.angRes_out * self.angRes_out, h, w)  # n == angRes * angRes
		out = FormOutput(buffer)
		return x_out


class EpiFeatureRebuild(nn.Module):
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


class HFEM(nn.Module):
	def __init__(self, angRes, n_blocks, channels,first=False):
		super(HFEM, self).__init__()
		
		self.first = first 
		self.n_blocks = n_blocks
		self.angRes = angRes
		self.channel = channels

		# define head module epi feature
		head_epi = []
		if first:  
			head_epi.append(nn.Conv2d(angRes, channels, kernel_size=3, stride=1, padding=1, bias=False))
		else:
			head_epi.append(nn.Conv2d(angRes*channels, channels, kernel_size=3, stride=1, padding=1, bias=False))

		self.head_epi = nn.Sequential(*head_epi)

		self.epi2spa = nn.Sequential(
			nn.Conv2d(4*channels, int(angRes * angRes * channels // 2), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(angRes),
		)

		self.spafuse = nn.Sequential(
			nn.Conv2d(4*channels, int(channels // 2), kernel_size=1,stride =1,dilation=1,padding=0, bias=False)
		)

		# define head module intra spatial feature
		head_spa_intra = []
		if first:  
			head_spa_intra.append(nn.Conv2d(1 ,channels, kernel_size=3, stride=1,dilation=int(angRes), padding=int(angRes), bias=False))
		else:
			head_spa_intra.append(nn.Conv2d(channels ,channels, kernel_size=3, stride=1,dilation=int(angRes), padding=int(angRes), bias=False))
			

		self.head_spa_intra = nn.Sequential(*head_spa_intra)


		# define head module inter spatial feature
		head_spa_inter = []
		if first:  
			head_spa_inter.append(nn.Conv2d(1 ,channels, kernel_size=3, stride=1,dilation=1, padding=1, bias=False))
		else:
			head_spa_inter.append(nn.Conv2d(channels ,channels, kernel_size=3, stride=1,dilation=1, padding=1, bias=False))
			

		self.head_spa_inter = nn.Sequential(*head_spa_inter)

		

		# define head module intra angular feature
		head_ang_intra = []
		if first: 
			head_ang_intra.append(nn.Conv2d(1 ,channels, kernel_size=int(angRes), stride = int(angRes), dilation=1, padding=0, bias=False))

		else:
			head_ang_intra.append(nn.Conv2d(channels ,channels, kernel_size=int(angRes), stride = int(angRes), dilation=1, padding=0, bias=False))
			

		self.head_ang_intra = nn.Sequential(*head_ang_intra)

		self.ang2spa_intra = nn.Sequential(
			nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(angRes), 
		)


		# define head module inter angular feature
		head_ang_inter = []
		if first:  
			head_ang_inter.append(nn.Conv2d(1 ,channels, kernel_size=int(angRes*2), stride = int(angRes*2), dilation=1, padding=0, bias=False))

		else:
			head_ang_inter.append(nn.Conv2d(channels ,channels, kernel_size=int(angRes*2), stride = int(angRes*2), dilation=1, padding=0, bias=False))
			

		self.head_ang_inter = nn.Sequential(*head_ang_inter)

			
		self.ang2spa_inter = nn.Sequential(
			nn.Conv2d(channels, int(4*angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(2*angRes),
		)

		# define  module attention fusion feature
		# self.attention_fusion =  AttentionFusion(channels)
		self.MHSA_params = {}
		self.MHSA_params['num_heads'] = 8
		self.MHSA_params['dropout'] = 0.
		self.selfAttention = EPIX_Trans(self.angRes, self.channel, self.MHSA_params)
		self.crossAttention = SA_Epi_Trans(self.angRes, self.channel, self.MHSA_params)
											
		# define  module spatial residual group
		self.SRG = nn.Sequential(ResidualGroup(self.n_blocks, channels,kernel_size=3,stride =1,dilation=int(angRes), padding=int(angRes), bias=False))



	def forward(self, x):

		# MO-EPI feature extractor
		data_0, data_90, data_45, data_135 = MacPI2EPI(x,self.angRes)

		data_0 = self.head_epi(data_0)
		data_90 = self.head_epi(data_90)
		data_45 = self.head_epi(data_45)
		data_135 = self.head_epi(data_135)

		mid_merged = torch.cat((data_0, data_90, data_45, data_135), 1)
		x_epi = self.epi2spa(mid_merged)

		# intra/inter spatial feature extractor
		x_s_intra = self.head_spa_intra(x)

		x_s_inter = self.head_spa_inter(x)

		# intra/inter angular feature extractor
		x_a_intra = self.head_ang_intra(x)
		x_a_intra = self.ang2spa_intra(x_a_intra)

		x_a_inter = self.head_ang_inter(x)
		x_a_inter = self.ang2spa_inter(x_a_inter)

		# fusion feature and refinement
		# out = x_epi.unsqueeze(1)
		x_spa = torch.cat([x_s_intra, x_s_inter, x_a_intra, x_a_inter], 1)
		x_spa = self.spafuse(x_spa)

		out = torch.cat([x_epi, x_spa], 1)

		out = rearrange(out, 'b c (u h) (v w) -> b c (u v) h w', u=self.angRes, v=self.angRes)
		out = self.selfAttention(out)
		out = self.crossAttention(out)
		out = rearrange(out, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)
		out = self.SRG(out)

		return out

class SA_Epi_CrossAttention_Trans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(SA_Epi_CrossAttention_Trans, self).__init__()
        self.emb_dim = emb_dim
        self.sa_linear_in = nn.Linear(channels//2, emb_dim, bias=False)
        self.epi_linear_in = nn.Linear(channels//2, emb_dim, bias=False)
        self.sa_norm = nn.LayerNorm(emb_dim)
        self.epi_norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim * 2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels//2, bias=False)

    def forward(self, buffer):
        b, c, u, v, h, w = buffer.shape
        token = buffer.permute(3, 5, 0, 2, 4, 1).reshape(v * w, b * u * h, c)
        sa_token = token[:, :, :c//2]
        epi_token = token[:, :, c//2:]

        epi_token_short_cut = epi_token

        sa_token = self.sa_linear_in(sa_token)
        epi_token = self.epi_linear_in(epi_token)

        sa_token_norm = self.sa_norm(sa_token)
        epi_token_norm = self.epi_norm(epi_token)
        sa_token = self.attention(query=sa_token_norm,
                                   key=epi_token_norm,
                                   value=sa_token,
                                   need_weights=False)[0] + sa_token

        sa_token = self.feed_forward(sa_token) + sa_token
        sa_token = self.linear_out(sa_token)

        buffer = torch.cat((sa_token, epi_token_short_cut), 2)
        buffer = buffer.reshape(v, w, b, u, h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer


class SA_Epi_Trans(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(SA_Epi_Trans, self).__init__()
        self.angRes = angRes

        self.epi_trans = SA_Epi_CrossAttention_Trans(channels, channels * 2, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )

    def forward(self, x):
        b, c, n, h, w = x.size()

        u, v = self.angRes, self.angRes

        shortcut = x

        # EPI uh
        buffer = x.reshape(b, c, u, v, h, w).permute(0, 1, 3, 2, 5, 4)  # (b,c,v,u,w,h)
        buffer = self.conv_1(self.epi_trans(buffer).permute(0, 1, 3, 2, 5, 4).reshape(b, c, n, h, w)) + shortcut


        # EPI vw
        buffer = buffer.reshape(b, c, u, v, h, w)
        buffer = self.conv_1(self.epi_trans(buffer).reshape(b, c, n, h, w)) + shortcut

        return buffer


class EpiXTrans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(EpiXTrans, self).__init__()
        self.emb_dim = emb_dim
        self.linear_in = nn.Linear(channels, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim * 2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels, bias=False)

    ######### very important!!!
    def gen_mask(self, h: int, w: int, maxdisp: int = 18):  # when 30 Scenes Reflective Occlusion
        # def gen_mask(self, h: int, w: int, maxdisp: int=18):  # when HCI data
        attn_mask = torch.zeros([h, w, h, w])
        [ii, jj] = torch.meshgrid(torch.arange(h), torch.arange(w))

        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[(ii - i).abs() * maxdisp >= (jj - j).abs()] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = attn_mask.reshape(h * w, h * w)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        b, c, u, v, h, w = buffer.shape
        attn_mask = self.gen_mask(v, w, ).to(buffer.device)

        epi_token = buffer.permute(3, 5, 0, 2, 4, 1).reshape(v * w, b * u * h, c)
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = epi_token.reshape(v, w, b, u, h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer


class EPIX_Trans(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(EPIX_Trans, self).__init__()
        self.angRes = angRes

        self.epi_trans = EpiXTrans(channels, channels * 2, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )

    def forward(self, x):
        b, c, n, h, w = x.size()

        u, v = self.angRes, self.angRes

        shortcut = x

        # EPI uh
        buffer = x.reshape(b, c, u, v, h, w).permute(0, 1, 3, 2, 5, 4)  # (b,c,v,u,w,h)
        buffer = self.conv_1(self.epi_trans(buffer).permute(0, 1, 3, 2, 5, 4).reshape(b, c, n, h, w)) + shortcut

        # EPI vw
        buffer = buffer.reshape(b, c, u, v, h, w)
        buffer = self.conv_1(self.epi_trans(buffer).reshape(b, c, n, h, w)) + shortcut

        return buffer



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
	net = HLFSR_ASR(2,7,15, 64).cuda()
	input = torch.randn(1, 1, 128, 128).cuda()
	output = net(input)
	print(output.shape)