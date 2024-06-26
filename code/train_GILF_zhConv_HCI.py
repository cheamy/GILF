import time
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils_anyang import *
import math
# from model_C42ConvXformer import Net
# from model_crossAttention_anyAng_HCI import Net_CrossAttention
from GILF_zhConv import Net_CrossAttention
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# torch.set_num_threads(16)
# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=2, help="angular resolution")
    parser.add_argument("--angout", type=int, default=7, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='ASRNet_CrossAttention_anyAng')
    parser.add_argument('--trainset_dir', type=str, default='/home/vision/work/zhonglin/Data/TrainData_RE_HCI_2x2-7x7/')#TrainingData_SIG_2x2_ASR_7x7  TrainingData_HCIori_2x2_ASR_7x7
    parser.add_argument('--testset_dir', type=str, default='/home/vision/work/zhonglin/Data/TestData_RE_HCI_2x2-7x7/')#    TestData_SIG_2x2_ASR_7x7   TestData_HCI_2x2_ASR_7x7

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')##1e-4,25 for LFEASR
    parser.add_argument('--n_epochs', type=int, default=120, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument("--smooth", type=float, default=0.001, help="smooth loss")
    parser.add_argument("--epi", type=float, default=1.0, help="epi loss")

    parser.add_argument("--patchsize", type=int, default=64, help="crop into patches for validation")
    parser.add_argument("--stride", type=int, default=32, help="stride for patch cropping")

    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='../checkpoint_crossAttention_anyAng_HCI/ASRNet_CrossAttention_anyAng_2x2xSR_7x7_epoch_35.pth.tar')

    return parser.parse_args()

if not os.path.exists('../checkpoint_zhConv_HCI'):
        os.mkdir('../checkpoint_zhConv_HCI')

def train(cfg, train_loader, test_Names, test_loaders):

    # net = Net(cfg.angin, cfg.angout)#, cfg.upscale_factor)
    net = Net_CrossAttention(cfg.angin, cfg.angout)#, cfg.upscale_factor)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0
    ##### get input index ######         
    ind_all = np.arange(cfg.angout*cfg.angout).reshape(cfg.angout, cfg.angout)        
    delt = (cfg.angout-1) // (cfg.angin-1)
    ind_source = ind_all[0:cfg.angout:delt, 0:cfg.angout:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))
    #ind_source = torch.tensor([9,14,49,54])
    #ind_source = torch.tensor([18,21,42,45])

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            optimizer.load_state_dict(model['optimazer'])
            epoch_state = model["epoch"]
            print("load pre-train at epoch {}".format(epoch_state))
        else:
            print("=> no model found at '{}'".format(cfg.load_model))
    scheduler._step_count = epoch_state
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    loss_epoch = []
    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            out  = net(data)
            loss = criterion_Loss(out, label)
            #loss += cfg.epi * epi_loss(out, label, cfg.angout)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            txtfile = open('../checkpoint_zhConv_HCI/' + cfg.model_name + '_training.txt', 'a')
            txtfile.write(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())) + '\n')
            txtfile.close()
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'optimazer': optimizer.state_dict(),
                # 'state_dict': net.module.state_dict(),  # for torch.nn.DataParallel
                'loss': loss_list},
                save_path='../checkpoint_zhConv_HCI/', filename=cfg.model_name + '_' + str(cfg.angin) + 'x' + str(cfg.angin)+ 'xSR_' + str(cfg.angout) +
                            'x' + str(cfg.angout) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []

        ''' evaluation '''
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []
            for index, test_name in enumerate(test_Names):
                test_loader = test_loaders[index]
                psnr_epoch_test, ssim_epoch_test = valid(test_loader, net, ind_source)
                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
                txtfile = open('../checkpoint_zhConv_HCI/' + cfg.model_name + '_training.txt', 'a')
                txtfile.write('Dataset----%10s,\t PSNR---%f,\t SSIM---%f\n' % (test_name, psnr_epoch_test, ssim_epoch_test))
                txtfile.close()
                pass
            pass

        scheduler.step()
        pass


def valid(test_loader, net, ind_source):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angin, w*angin
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angin, vw // cfg.angin
        subLFin = LFdivide(data, cfg.angin, cfg.patchsize, cfg.stride)  # numU, numV, h*angin, w*angin
        numU, numV, H, W = subLFin.shape
        minibatch = 4
        num_inference = numU*numV//minibatch
        tmp_in = subLFin.contiguous().view(numU*numV, subLFin.shape[2], subLFin.shape[3])
        
        with torch.no_grad():
            out_lf = []
            for idx_inference in range(num_inference):
                tmp = tmp_in[idx_inference*minibatch:(idx_inference+1)*minibatch,:,:].unsqueeze(1)
                out_lf.append(net(tmp.to(cfg.device)))#
            if (numU*numV)%minibatch:
                tmp = tmp_in[(idx_inference+1)*minibatch:,:,:].unsqueeze(1)
                out_lf.append(net(tmp.to(cfg.device)))#
        out_lf = torch.cat(out_lf, 0)
        subLFout = out_lf.view(numU, numV, cfg.angout * cfg.patchsize, cfg.angout * cfg.patchsize)

        outLF = LFintegrate(subLFout, cfg.angout, cfg.patchsize, cfg.stride, h0, w0)
        # print(outLF.shape)
        # print(label.shape)

        psnr, ssim = cal_metrics(label, outLF, cfg.angout, ind_source)

        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test

def reconstruction_loss(X,Y):
# L1 Charbonnier loss
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt( diff * diff + eps )
    loss = torch.sum(error) / torch.numel(error)
    return loss

def gradient(pred):
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def epi_loss(pred, label, angout):
    pred = LFsplit(pred, angout)
    label = LFsplit(label, angout)
    b, n, c, h, w = pred.shape
    pred = pred.contiguous().view(b,-1,h,w)
    label = label.contiguous().view(b,-1,h,w)
    #print(pred.shape)

    def lf2epi(lf):
        N,an2,h,w = lf.shape
        an = int(math.sqrt(an2))
        # [N,an2,h,w] -> [N*ah*h,aw,w]  &  [N*aw*w,ah,h]
        # print(an)
        # print(lf.view(N,an,an,h,w).permute(0,1,3,2,4).view(-1,an,w).shape)
        epi_h = lf.view(N,an,an,h,w).permute(0,1,3,2,4).contiguous().view(-1,1,an,w)
        epi_v = lf.view(N,an,an,h,w).permute(0,2,4,1,3).contiguous().view(-1,1,an,h)
        return epi_h, epi_v
    
    epi_h_pred, epi_v_pred = lf2epi(pred)
    dx_h_pred, dy_h_pred = gradient(epi_h_pred)
    dx_v_pred, dy_v_pred = gradient(epi_v_pred)
    
    epi_h_label, epi_v_label = lf2epi(label)
    dx_h_label, dy_h_label = gradient(epi_h_label)
    dx_v_label, dy_v_label = gradient(epi_v_label)
    
    return reconstruction_loss(dx_h_pred, dx_h_label) + reconstruction_loss(dy_h_pred, dy_h_label) + reconstruction_loss(dx_v_pred, dx_v_label) + reconstruction_loss(dy_v_pred, dy_v_label)

def save_ckpt(state, save_path='../checkpoint_zhConv_HCI', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename), _use_new_zipfile_serialization=False)


def main(cfg):
    setup_seed(10)
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=cfg.batch_size, shuffle=True)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, train_loader, test_Names, test_Loaders)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)