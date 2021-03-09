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
# import rawpy
import glob

import utils
from unet import UNet
from dataset.SID import SIDSonyTrainDataset, SIDSonyValDataset

train_input_dir = './dataset/SID/Sony/train/short/'
train_gt_dir = './dataset/SID/Sony/train/long/'
train_list_file= './dataset/SID/Sony_train_list.txt'
# val_input_dir = './dataset/SID/Sony/val/short/'
# val_gt_dir = './dataset/SID/Sony/val/long/'
# val_list_file= './dataset/SID/Sony_val_list.txt'
checkpoint_dir = './Sony_results/result_Sony_no_ratio/'
result_dir = checkpoint_dir

writer = SummaryWriter(log_dir=checkpoint_dir + 'logs')

bs = 1
ps = 512  # patch size for training
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

train_dataset = SIDSonyTrainDataset(list_file = train_list_file, root_dir = './dataset/SID/', ps=ps)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
iteration = 0

# val_dataset = SIDSonyValDataset(list_file=val_list_file, root_dir='./dataset/SID/')
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
# min_v_loss = np.float(999.0)

for epoch in tqdm(range(lastepoch, 4001)):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    g_loss, v_loss = [], []
    cnt = 0

    # Training
    unet.train()
    for sample in iter(train_dataloader):
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
        gt_imgs = gt_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        gt_imgs = np.minimum(np.maximum(gt_imgs,0),1)

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_imgs[0, :, :, :], outputs[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, sample['ind'][0], sample['ratio'][0]))

        iteration += 1
        writer.add_scalar('Train/MAE_Loss', mae_loss, iteration)
        writer.add_scalar('Train/G_Loss', loss, iteration)
        writer.add_scalar('Train/G_Loss_Mean', np.mean(g_loss), iteration)
    
    # Validation
    # unet.eval()
    # with torch.no_grad():
    #     for sample in iter(val_dataloader):
    #         in_img = sample['in_img'].to(device)
    #         gt_img = sample['gt_img'].to(device)

    #         out_img = unet(in_img)

    #         mae_loss = utils.MAELoss(out_img, gt_img)
    #         loss = mae_loss
    #         v_loss.append(loss.item())
    #     print("%d Validation_Loss=%.3f" % (epoch, np.mean(v_loss)))
    #     writer.add_scalar('Validation/Valid_Loss_Mean', np.mean(v_loss), epoch)

    #     if np.mean(v_loss) < min_v_loss:
    #         min_v_loss = np.mean(v_loss)
    #         torch.save(unet.state_dict(), checkpoint_dir + 'model_best.pth')

    scheduler.step()
    torch.save(unet.state_dict(), checkpoint_dir + 'model.pth')
    writer.close()
    