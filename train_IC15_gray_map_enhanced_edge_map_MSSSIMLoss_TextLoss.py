# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io, scipy.misc
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import glob

import utils
from unet import GrayEdgeAttentionUNet
from dataset.ICDAR15 import ICDAR15Dataset

from CRAFTpytorch.craft import CRAFT
from collections import OrderedDict

input_dir = './dataset/IC15_004/train/low/'
gt_dir = './dataset/IC15_004/train/high/'
edge_dir = './dataset/IC15_004/train/edge_en/'
list_file= './dataset/IC15_004/train_list.txt'
checkpoint_dir = './IC15_004_results_supp/result_IC15_gray_map_enhanced_edge_map_MSSSIMLoss_TextLoss_085_015_08506/'
result_dir = checkpoint_dir
w1, w2, w3 = 0.85, 0.15, 0.85*0.6

writer = SummaryWriter(log_dir=checkpoint_dir + 'logs')

bs = 1
ps = 512  # patch size for training
save_freq = 500

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = GrayEdgeAttentionUNet()
unet.to(device)
if lastepoch > 0:
    unet.load_state_dict(torch.load(checkpoint_dir + 'model.pth'))
unet.train()

learning_rate = 1e-4
G_opt = optim.Adam(unet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(G_opt, milestones=[2000], gamma=0.1)

dataset = ICDAR15Dataset(list_file = list_file, root_dir = './dataset/', ps=ps)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
iteration = 0

# text detection model
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
# CRAFT net
craft_net = CRAFT()
craft_pretrained_model = './CRAFTpytorch/craft_ic15_20k.pth'
craft_net.load_state_dict(copyStateDict(torch.load(craft_pretrained_model)))
craft_net.to(device)
craft_net.eval()

for epoch in tqdm(range(lastepoch, 4001)):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    g_loss = []
    cnt = 0

    for sample in iter(dataloader):
        st = time.time()
        cnt += bs

        in_imgs = sample['in_img'].to(device)
        gt_imgs = sample['gt_img'].to(device)
        in_gray_imgs = sample['in_gray_img'].to(device)
        in_edge_imgs = sample['in_edge_img'].to(device)
        
        G_opt.zero_grad()
        out_imgs = unet(in_imgs, in_gray_imgs, in_edge_imgs)

        mae_loss = utils.MAELoss(out_imgs, gt_imgs)
        ms_ssim_loss = utils.MS_SSIMLoss(out_imgs, gt_imgs)
        text_loss = utils.TextDetectionLoss(out_imgs, gt_imgs, craft_net)
        loss = w1*mae_loss + w2*ms_ssim_loss + w3*text_loss
        loss.backward()
        G_opt.step()

        g_loss.append(loss.item())

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss), time.time() - st))

        outputs = out_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        outputs = np.minimum(np.maximum(outputs,0),1)
        gt_imgs = gt_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        gt_imgs = np.minimum(np.maximum(gt_imgs,0),1)
        in_gray_imgs = in_gray_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        in_gray_imgs = np.minimum(np.maximum(in_gray_imgs,0),1)
        in_edge_imgs = in_edge_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        in_edge_imgs = np.minimum(np.maximum(in_edge_imgs,0),1)

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_imgs[0, :, :, :], outputs[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%05d_train.jpg' % (epoch, sample['ind'][0]))

        iteration += 1
        writer.add_scalar('Train/MAE_Loss', mae_loss, iteration)
        writer.add_scalar('Train/MS_SSIM_Loss', ms_ssim_loss, iteration)
        writer.add_scalar('Train/Text_Loss', text_loss, iteration)
        writer.add_scalar('Train/G_Loss', loss, iteration)
        writer.add_scalar('Train/G_Loss_Mean', np.mean(g_loss), iteration)
    
    scheduler.step()
    torch.save(unet.state_dict(), checkpoint_dir + 'model.pth')
    writer.close()
