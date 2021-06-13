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
from PIL import Image
import glob


import utils
from unet import UNet
from dataset.ICDAR15 import ICDAR15Dataset

input_dir = '/content/drive/MyDrive/thesis data/IC15_004/train/low/'#'./dataset/IC15_004/train/low/'
gt_dir = '/content/drive/MyDrive/thesis data/IC15_004/train/high/'#'./dataset/IC15_004/train/high/'
list_file= '/content/drive/MyDrive/thesis data/IC15_004/train_list.txt'#'./dataset/IC15_004/train_list.txt'
checkpoint_dir = './IC15_004_results/result_IC15_no_ratio/'
result_dir = checkpoint_dir

#local='/content/drive/MyDrive/local/'

writer = SummaryWriter(log_dir=checkpoint_dir + 'logs')

bs = 1
ps = 32 #512  # patch size for training
save_freq = 500

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet()
unet.to(device)

learning_rate = 1e-4
G_opt = optim.Adam(unet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(G_opt, milestones=[2000], gamma=0.1)

dataset = ICDAR15Dataset(list_file = list_file, root_dir = '/content/drive/MyDrive/thesis data/', ps=ps)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
iteration = 0

for epoch in tqdm(range(lastepoch, 4001)):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    g_loss = []
    cnt = 0

    # Training Module 
    unet.train()
    for sample in iter(dataloader):
        st = time.time()
        cnt += bs

        in_imgs = sample['in_img'].to(device)
        gt_imgs = sample['gt_img'].to(device)
        
        G_opt.zero_grad()
        out_imgs = unet(in_imgs)

        mae_loss = utils.MAELoss(out_imgs, gt_imgs)
        loss = mae_loss
        loss.backward()
        G_opt.step()

        g_loss.append(loss.item())

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss), time.time() - st))

        outputs = out_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        outputs = np.minimum(np.maximum(outputs,0),1)
        in_imgs = in_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        in_imgs = np.minimum(np.maximum(in_imgs,0),1)
        gt_imgs = gt_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        gt_imgs = np.minimum(np.maximum(gt_imgs,0),1)

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((in_imgs[0, :, :, :], gt_imgs[0, :, :, :], outputs[0, :, :, :]), axis=1)
            
            Image.fromarray((temp * 255).astype('uint8')).convert('RGB').save(
                result_dir + '%04d/%05d_train.jpg' % (epoch, sample['ind'][0]))            
            
            #scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
             #   result_dir + '%04d/%05d_train.jpg' % (epoch, sample['ind'][0]))

        iteration += 1
        writer.add_scalar('Train/MAE_Loss', mae_loss, iteration)
        writer.add_scalar('Train/G_Loss', loss, iteration)
        writer.add_scalar('Train/G_Loss_Mean', np.mean(g_loss), iteration)
    
    scheduler.step()
    torch.save(unet.state_dict(), checkpoint_dir + 'model.pth')
    writer.close()
    
