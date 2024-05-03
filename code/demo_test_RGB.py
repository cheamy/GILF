import argparse
from utils_append import *
from imresize import *
# from model import Net
# from new_model_v6 import newNet_v6
from model_crossAttention_anyAng_HCI import Net_CrossAttention
import numpy as np
import imageio
from einops import rearrange
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes_in", type=int, default=2, help="input angular resolution")
    parser.add_argument("--angRes_out", type=int, default=7, help="output angular resolution")
    parser.add_argument("--scene_type", type=str, default='stanford')
    parser.add_argument("--model_name", type=str, default='ASRNet_CrossAttention_anyAng_2x2xSR_7x7_epoch_125')
    # parser.add_argument("--model_name", type=str, default='DistgASR_HCI_2x2-7x7_model_v6_4x3_h256_singlecard_ep40')
    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument('--input_dir', type=str, default='../demo_test/')
    parser.add_argument('--save_path', type=str, default='../demo_test_stanford_7x7/')

    return parser.parse_args()

def demo_test(cfg):
    net = Net_CrossAttention(2, cfg.angRes_out)
    # net = newNet_v6(2, 7)
    net.to(cfg.device)
    model = torch.load('../checkpoint/' + cfg.model_name + '.pth.tar', map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])
    scene_list = os.listdir(cfg.input_dir + cfg.scene_type)
    PSNR = []
    SSIM = []
    for scenes in scene_list:
        print('Working on scene: ' + scenes + '...')
        name = scenes.split("_")[1]

        temp = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/out_05_05.png')
        lf_rgb_in = np.zeros(shape=(cfg.angRes_in, cfg.angRes_in, temp.shape[0], temp.shape[1], 3))
        lf_rgb_out = np.zeros(shape=(cfg.angRes_out, cfg.angRes_out, temp.shape[0], temp.shape[1], 3)).astype('float32')

        lf_rgb_in[0, 0, :, :, :] = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/out_05_05.png')
        lf_rgb_in[0, 1, :, :, :] = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/out_05_11.png')
        lf_rgb_in[1, 0, :, :, :] = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/out_11_05.png')
        lf_rgb_in[1, 1, :, :, :] = imageio.imread(cfg.input_dir + cfg.scene_type + '/' + scenes + '/out_11_11.png')

        lf_r_in = np.zeros(shape=(cfg.angRes_in, cfg.angRes_in, temp.shape[0], temp.shape[1], 3))
        lf_g_in = np.zeros(shape=(cfg.angRes_in, cfg.angRes_in, temp.shape[0], temp.shape[1], 3))
        lf_b_in = np.zeros(shape=(cfg.angRes_in, cfg.angRes_in, temp.shape[0], temp.shape[1], 3))

        lf_r_in[:, :, :, :, :] = lf_rgb_in[:, :, :, :, 0]
        lf_g_in[:, :, :, :, :] = lf_rgb_in[:, :, :, :, 1]
        lf_b_in[:, :, :, :, :] = lf_rgb_in[:, :, :, :, 2]

        # lf_y_in = (0.256789 * lf_rgb_in[:,:,:,:,0] + 0.504129 * lf_rgb_in[:,:,:,:,1] + 0.097906 * lf_rgb_in[:,:,:,:,2] + 16).astype('float32')
        # lf_cb_in = (-0.148223 * lf_rgb_in[:,:,:,:,0] - 0.290992 * lf_rgb_in[:,:,:,:,1] + 0.439215 * lf_rgb_in[:,:,:,:,2] + 128).astype('float32')
        # lf_cr_in = (0.439215 * lf_rgb_in[:,:,:,:,0] - 0.367789 * lf_rgb_in[:,:,:,:,1] - 0.071426 * lf_rgb_in[:,:,:,:,2] + 128).astype('float32')

        lf_r_out = np.zeros(shape=(cfg.angRes_out, cfg.angRes_out, temp.shape[0], temp.shape[1])).astype('float32')
        lf_g_out = np.zeros(shape=(cfg.angRes_out, cfg.angRes_out, temp.shape[0], temp.shape[1])).astype('float32')
        lf_b_out = np.zeros(shape=(cfg.angRes_out, cfg.angRes_out, temp.shape[0], temp.shape[1])).astype('float32')

        # lf_cb_out = np.zeros(shape=(cfg.angRes_out, cfg.angRes_out, temp.shape[0], temp.shape[1])).astype('float32')
        # lf_cr_out = np.zeros(shape=(cfg.angRes_out, cfg.angRes_out, temp.shape[0], temp.shape[1])).astype('float32')
        for h in range(temp.shape[0]):
            for w in range(temp.shape[1]):
                lf_r_out[:, :, h, w] = imresize(lf_r_in[:, :, h, w], cfg.angRes_out/cfg.angRes_in)
                lf_g_out[:, :, h, w] = imresize(lf_g_in[:, :, h, w], cfg.angRes_out/cfg.angRes_in)
                lf_b_out[:, :, h, w] = imresize(lf_b_in[:, :, h, w], cfg.angRes_out / cfg.angRes_in)

        datar = rearrange(lf_r_in, 'u v h w -> (u h) (v w)')
        datar = torch.from_numpy(datar) / 255.0
        datag = rearrange(lf_g_in, 'u v h w -> (u h) (v w)')
        datag = torch.from_numpy(datag) / 255.0
        datab = rearrange(lf_b_in, 'u v h w -> (u h) (v w)')
        datab = torch.from_numpy(datab) / 255.0

        if cfg.crop == False:
            with torch.no_grad():
                outLF = net(data.unsqueeze(0).unsqueeze(0).to(cfg.device))
                # lf_y_out =  rearrange(outLF.squeeze(), '(u h) (v w) -> u v h w', u=cfg.angRes_out, v=cfg.angRes_out)
        else:
            psnr = 0
            ssim = 0
            for i in range(3):
                file_name = cfg.input_dir + cfg.scene_type + '/' + scenes + '/' + name + '_' + str(i) + '.h5'
                with h5py.File(file_name, 'r') as hf:
                    # data = np.array(hf.get('data'))
                    label = np.array(hf.get('label'))
                    # data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
                    label = np.transpose(label, (1, 0))
                    # data, label = ToTensor()(data.copy()), ToTensor()(label.copy())
                    label = ToTensor()(label.copy())
                    label = label.squeeze().to(cfg.device)

                patchsize = cfg.patchsize
                stride = patchsize // 2
                uh, vw = data.shape
                h0, w0 = uh // cfg.angRes_in, vw // cfg.angRes_in
                if i == 0:
                    subLFin = LFdivide(datar, cfg.angRes_in, patchsize, stride)  # numU, numV, h*angRes, w*angRes
                if i == 1:
                    subLFin = LFdivide(datag, cfg.angRes_in, patchsize, stride)
                if i == 2:
                    subLFin = LFdivide(datab, cfg.angRes_in, patchsize, stride)
                numU, numV, H, W = subLFin.shape
                subLFout = torch.zeros(numU, numV, cfg.angRes_out * patchsize, cfg.angRes_out * patchsize)

                for u in range(numU):
                    for v in range(numV):
                        tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                        with torch.no_grad():
                            torch.cuda.empty_cache()
                            out = net(tmp.to(cfg.device))
                            subLFout[u, v, :, :] = out.squeeze()
                lf_y_out = LFintegrate(subLFout, cfg.angRes_out, patchsize, stride, h0, w0)
                psnr_s, ssim_s = cal_metrics_RE(label, lf_y_out, cfg.angRes_in, cfg.angRes_out)
                psnr += psnr_s
                ssim += ssim_s
            psnr /= 3
            ssim /= 3

        print('Dataset----%10s, PSNR---%f, SSIM---%f' % (scenes, psnr, ssim))

        PSNR.append(psnr)
        SSIM.append(ssim)



        # lf_y_out = 255 * lf_y_out.data.cpu().numpy()
        # lf_rgb_out[:, :, :, :, 0] = 1.164383 * (lf_y_out - 16) + 1.596027 * (lf_cr_out - 128)
        # lf_rgb_out[:, :, :, :, 1] = 1.164383 * (lf_y_out - 16) - 0.391762 * (lf_cb_out - 128) - 0.812969 * (lf_cr_out - 128)
        # lf_rgb_out[:, :, :, :, 2] = 1.164383 * (lf_y_out - 16) + 2.017230 * (lf_cb_out - 128)
        #
        #
        # lf_rgb_out = np.clip(lf_rgb_out, 0, 255)
        # output_path = cfg.save_path + scenes
        # if not (os.path.exists(output_path)):
        #     os.makedirs(output_path)
        # for u in range(cfg.angRes_out):
        #     for v in range(cfg.angRes_out):
        #         imageio.imwrite(output_path + '/view_%.2d_%.2d.png' % (u, v), np.uint8(lf_rgb_out[u, v, :, :]))

        print('Finished! \n')
    print('angular_out: %d Total: PSNR---%f, SSIM---%f' % (cfg.angRes_out,sum(PSNR)/len(PSNR), sum(SSIM)/len(SSIM)))

if __name__ == '__main__':
    cfg = parse_args()
    demo_test(cfg)