# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io, scipy.misc, cv2
import torch
import numpy as np
import glob
import utils

from unet import UNet
from torch.utils.data import DataLoader
from dataset.SID import SIDFujiTestDataset

test_input_dir = './dataset/SID/Fuji/test/short/'
test_gt_dir = './dataset/SID/Fuji/test/long/'
test_list_file= './dataset/SID/Fuji_test_list.txt'
checkpoint_dir = './Fuji_results/result_Fuji_MSSSIMLoss025/'
result_dir = checkpoint_dir
ckpt = checkpoint_dir + 'model.pth'

# get test IDs
test_fns = glob.glob(test_gt_dir + '*.png')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet()
unet.load_state_dict(torch.load(ckpt))
unet.to(device)

test_dataset = SIDFujiTestDataset(list_file=test_list_file, root_dir='./dataset/SID/')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
iteration = 0

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

with torch.no_grad():
    unet.eval()

    for sample in iter(test_dataloader):
        in_fn = sample['in_fn'][0]
        print(in_fn)
        in_img = sample['in_img'].to(device)
        gt_img = sample['gt_img'].to(device)
        out_img = unet(in_img)

        output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
        output = np.minimum(np.maximum(output, 0), 1)
        gt_img = gt_img.permute(0, 2, 3, 1).cpu().data.numpy()
        gt_img = np.minimum(np.maximum(gt_img, 0), 1)

        output = output[0, :, :, :]
        gt_img = gt_img[0, :, :, :]

        output = cv2.resize(output, (sample['width'], sample['hight']))
        # if '_00_' in in_fn:
        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/' + in_fn)

    