#from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from skimage import metrics
from scipy.interpolate import interp2d
from math import ceil, floor
from einops import rearrange


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        # file_list = file_list[0:12] # to test if the code can run
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/%06d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('Lr_SAI_y'))
            label = np.array(hf.get('Hr_SAI_y'))
            data, label = augmentation(data, label)
            data = ToTensor()(data.copy())
            label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.item_num
    

class TrainSetLoader_cache_inmemory(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader_cache_inmemory, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        # file_list = file_list[0:12] # to test if the code can run
        item_num = len(file_list)
        self.item_num = item_num

        self.data =[]
        self.label = []

        for ii in range(item_num):
            file_name = [dataset_dir + '/%06d' % (ii+1) + '.h5']
            with h5py.File(file_name[0], 'r') as hf:
                self.data.append(np.array(hf.get('data'))) 
                self.label.append(np.array(hf.get('label'))) 

        # file_name = [dataset_dir + '/%06d' % index + '.h5']

    def __getitem__(self, index):
        data = self.data[index % self.item_num]
        label = self.label[index % self.item_num]

        # dataset_dir = self.dataset_dir
        # index = index + 1
        # file_name = [dataset_dir + '/%06d' % index + '.h5']
        # with h5py.File(file_name[0], 'r') as hf:
        #     data = np.array(hf.get('data'))
        #     label = np.array(hf.get('label'))
        data, label = augmentation(data, label)
        data = ToTensor()(data.copy())
        label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.item_num




class TrainSetLoader_k(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader_k, self).__init__()
        self.dataset_dir = dataset_dir
        # file_list = os.listdir(dataset_dir)
        self.file_list = []
        self.inp_size = 32
        self.scale = 2
        # self.data_list = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']
        self.data_list = ['HCI_new', 'HCI_old']
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        item_num = len(self.file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [self.dataset_dir + self.file_list[index % self.item_num]]



        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('Lr_SAI_y'))
            label = np.array(hf.get('Hr_SAI_y'))
            data, label = augmentation(data, label)
            data = ToTensor()(data.copy())
            label = ToTensor()(label.copy())
            data = LFsplit(data, 5)
            label = LFsplit(label, 5)
            # data = ToTensor()(data_mv.copy())
            # label = ToTensor()(label_mv.copy())
            r_ang = np.random.randint(1, 5)
            label = np.rot90(label, r_ang, (2, 3))
            label = np.rot90(label, r_ang, (0, 1))
            data = np.rot90(data, r_ang, (2, 3))
            data = np.rot90(data, r_ang, (0, 1))
            u1, v1, H1, W1 = data.shape
            u2, v2, H2, W2 = label.shape
            data = data.reshape(-1, H1, W1)
            label = label.reshape(-1, H2, W2)
            label_mix = label.mean(axis=0)
            data_mix = data.mean(axis=0)
            if np.random.rand(1) > 0.9:
                label, data = cutmib(label, data, label_mix, data_mix, self.scale, prob=0.9, alpha=0.1)
            label = torch.from_numpy(label.astype(np.float32))
            data = torch.from_numpy(data.astype(np.float32))
            data = data.view(1, u1, v1, H1, W1)
            label = label.view(1, u2, v2, H2, W2)
            label = rearrange(label, 'b u v h w -> b (u h) (v w)')
            data = rearrange(data, 'b u v h w -> b (u h) (v w)')

        return data, label

    @staticmethod
    def augmentation(label):
        if random.random() < 0.5:  # flip along W-V direction
            label = torch.flip(label, dims=[2, 4])
        if random.random() < 0.5:  # flip along W-V direction
            label = torch.flip(label, dims=[1, 3])
        if random.random() < 0.5:  # transpose between U-V and H-W
            label = label.permute(0, 2, 1, 4, 3)
        return label

    def __len__(self):
        return self.item_num

def LFsplit(data, angRes):
    n, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            if u % 2 == 0 and v % 2 == 0:
                data_sv.append(data[:, u * h:(u + 1) * h, v * w:(v + 1) * w])
    data_st = torch.stack(data_sv)
    shape = (angRes, angRes, h, w)
    data_reshape = data_st.view(shape)
    return data_reshape

def cutmib(im1, im2, im1_mix, im2_mix, scale, prob=1.0, alpha=0.7):
    cut_ratio = np.random.randn() * 0.01 + alpha
    an, h_lr, w_lr = im2.shape
    ch_lr, cw_lr = np.int(h_lr*cut_ratio), np.int(w_lr*cut_ratio)
    ch_hr, cw_hr = ch_lr*scale, cw_lr*scale
    cy_lr = np.random.randint(0, h_lr-ch_lr+1)
    cx_lr = np.random.randint(0, w_lr-cw_lr+1)
    cy_hr, cx_hr = cy_lr*scale, cx_lr*scale

    if np.random.random() < prob:
        if np.random.random() > 0.5:
            for i in range(an):
                im2[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = imresize(im1_mix[..., cy_hr:cy_hr+ch_hr, cx_hr:cx_hr+cw_hr], scalar_scale=1/scale)
                # im2[..., cy:cy+ch_lr, cx:cx+cw_lr] = im1[..., cy:cy+ch_lr, cx:cx+cw_lr]
            # print(cy:cy+ch_lr, cx:cx+cw_lr)
        else:
            im2_aug = im2
            for i in range(an):
                im2_aug[i] = imresize(im1[i], scalar_scale=1/scale)
                im2_aug[i, cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr] = im2_mix[..., cy_lr:cy_lr+ch_lr, cx_lr:cx_lr+cw_lr]
                im2 = im2_aug

        return im1, im2
    else:
        return im1, im2

def MultiTestSetDataLoader(args, test_name):
    dataset_dir = args.testset_dir
    data_list = sorted(os.listdir(dataset_dir))
    data_list1 = []
    test_Loaders = []
    length_of_tests = 0

    for data_name in data_list:
        if(data_name in test_name):
            test_Dataset = TestSetDataLoader(args, data_name)
            length_of_tests += len(test_Dataset)
            test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))
            data_list1.append(data_name)
    return data_list1, test_Loaders, length_of_tests

def MultiTestSetDataLoader_ori(args):
    dataset_dir = args.testset_dir
    data_list = sorted(os.listdir(dataset_dir))
    test_Loaders = []
    length_of_tests = 0

    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL'):
        super(TestSetDataLoader, self).__init__()
        self.angRes = args.angRes
        self.dataset_dir = args.testset_dir + data_name
        # self.dataset_dir = args.testset_dir
        self.file_list = []
        tmp_list = os.listdir(self.dataset_dir)
        for index, _ in enumerate(tmp_list):
            tmp_list[index] = tmp_list[index]
        self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = self.dataset_dir + '/' + self.file_list[index]
        with h5py.File(file_name, 'r') as hf:
            data = np.array(hf.get('Lr_SAI_y'))
            label = np.array(hf.get('Hr_SAI_y'))
            data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
            data, label = ToTensor()(data.copy()), ToTensor()(label.copy())

        return data, label

    def __len__(self):
        return self.item_num

def interplot2D(original_array, factor):
    # 定义新的x和y坐标轴
    x_old = np.arange(original_array.shape[1])
    y_old = np.arange(original_array.shape[0])
    x_new = np.linspace(0, original_array.shape[1] - 1, original_array.shape[1] * factor)
    y_new = np.linspace(0, original_array.shape[0] - 1, original_array.shape[0] * factor)

    # 创建一个双线性插值函数
    interp_func = interp2d(x_old, y_old, original_array, kind='linear')

    # 使用插值函数对新的坐标轴进行插值
    new_array = interp_func(x_new, y_new)

    # 打印结果
    return new_array

def resize_fuc(data, label, angRes_in, angRes_out):
    factor = 2
    patchsize = data.shape[0] // angRes_in
    new_data = np.zeros_like(data)
    new_label = np.zeros_like(label)
    start = patchsize // 2 - patchsize // 2 // 2
    end = patchsize // 2 + patchsize // 2 // 2
    for u in range(angRes_in):
        for v in range(angRes_in):
            temp = data[u*patchsize:(u+1)*patchsize, v*patchsize:(v+1)*patchsize]
            new_data[u*patchsize:(u+1)*patchsize, v*patchsize:(v+1)*patchsize] = interplot2D(temp[start:end, start:end], factor)
    for u in range(angRes_out):
        for v in range(angRes_out):
            temp = label[u*patchsize:(u+1)*patchsize, v*patchsize:(v+1)*patchsize]
            new_label[u*patchsize:(u+1)*patchsize, v*patchsize:(v+1)*patchsize] = interplot2D(temp[start:end, start:end], factor)

    return new_data, new_label

def new_augmentation(data, label, angRes_in, angRes_out):
    if random.random() < 0.5:  # resize data
        data, label = resize_fuc(data, label, angRes_in, angRes_out)
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)

    return data, label

def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5: # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label


def LFdivide(data, angRes, patch_size, stride):
    uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU-1) + patch_size
    wE = stride * (numV-1) + patch_size

    dataE = torch.zeros(hE*angRes, wE*angRes)
    for u in range(angRes):
        for v in range(angRes):
            Im = data[u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[u*hE : u*hE+h, v*wE : v*wE+w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, patch_size*angRes, patch_size*angRes)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, u*patch_size:(u+1)*patch_size, v*patch_size:(v+1)*patch_size] = dataE[uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def ImageExtend(Im, bdr):
    h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def LFintegrate(subLF, angRes, pz, stride, h0, w0):
    numU, numV, pH, pW = subLF.shape
    ph, pw = pH //angRes, pW //angRes
    bdr = (pz - stride) //2
    temp = torch.zeros(stride*numU, stride*numV)
    outLF = torch.zeros(angRes, angRes, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]

            outLF[u, v, :, :] = temp[0:h0, 0:w0]

    return outLF


def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np)#, data_range=1.0

def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy().clip(0, 1)
    img2_np = img2.data.cpu().numpy().clip(0, 1)

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True)#, data_range=1.0)

def cal_ssim_1(img1, img2):
    img1_np = img1.data.cpu().numpy().clip(0, 1)
    img2_np = img2.data.cpu().numpy().clip(0, 1)

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, data_range=1.0)

def cal_metrics(img1, img2, angRes):
    if len(img1.size())==2:
        [H, W] = img1.size()
        img1 = img1.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)
    if len(img2.size())==2:
        [H, W] = img2.size()
        img2 = img2.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)

    [U, V, h, w] = img1.size()
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    # SSIM = np.zeros(shape=(U, V), dtype='float32')
    SSIM_1 = np.zeros(shape=(U, V), dtype='float32')

    if angRes==5:
        for u in range(U):
            for v in range(V):
                PSNR[u, v] = cal_psnr(img1[u, v, :, :], img2[u, v, :, :])
                # SSIM[u, v] = cal_ssim(img1[u, v, :, :], img2[u, v, :, :])
                SSIM_1[u, v] = cal_ssim_1(img1[u, v, :, :], img2[u, v, :, :])
                pass
            pass
    else:
        bd = 22
        for u in range(U):
            for v in range(V):
                PSNR[u, v] = cal_psnr(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
                # SSIM[u, v] = cal_ssim(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
                SSIM_1[u, v] = cal_ssim_1(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
                pass
            pass


    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    # ssim_mean = SSIM.sum() / np.sum(SSIM > 0)
    ssim_mean_1 = SSIM_1.sum() / np.sum(SSIM_1 > 0)

    return psnr_mean, ssim_mean_1#, ssim_mean_1




# implementation of matlab bicubic interpolation in pytorch
class Bicubic(object):
    def __init__(self):
        super(Bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0), torch.FloatTensor([in_size[0]])).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1), torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def __call__(self, input, scale=1/4):
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0].to(input.device)
        weight1 = weight1[0].to(input.device)

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = torch.sum(out, dim=3)
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out



def LF_rgb2ycbcr(x):
    y = torch.zeros_like(x)
    y[:,0,:,:,:,:] =  65.481 * x[:,0,:,:,:,:] + 128.553 * x[:,1,:,:,:,:] +  24.966 * x[:,2,:,:,:,:] +  16.0
    y[:,1,:,:,:,:] = -37.797 * x[:,0,:,:,:,:] -  74.203 * x[:,1,:,:,:,:] + 112.000 * x[:,2,:,:,:,:] + 128.0
    y[:,2,:,:,:,:] = 112.000 * x[:,0,:,:,:,:] -  93.786 * x[:,1,:,:,:,:] -  18.214 * x[:,2,:,:,:,:] + 128.0

    y = y / 255.0
    return y


def LF_ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = torch.zeros_like(x)
    y[:,0,:,:,:,:] = mat_inv[0,0] * x[:,0,:,:,:,:] + mat_inv[0,1] * x[:,1,:,:,:,:] + mat_inv[0,2] * x[:,2,:,:,:,:] - offset[0]
    y[:,1,:,:,:,:] = mat_inv[1,0] * x[:,0,:,:,:,:] + mat_inv[1,1] * x[:,1,:,:,:,:] + mat_inv[1,2] * x[:,2,:,:,:,:] - offset[1]
    y[:,2,:,:,:,:] = mat_inv[2,0] * x[:,0,:,:,:,:] + mat_inv[2,1] * x[:,1,:,:,:,:] + mat_inv[2,2] * x[:,2,:,:,:,:] - offset[2]
    return y


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2,
                                                                            (1 < absx) & (absx <= 2))
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method is 'bicubic':
        kernel = cubic
    elif method is 'bilinear':
        kernel = triangle
    else:
        print('Error: Unidentified method supplied')

    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255 * B
    return np.around(B).astype(np.uint8)

