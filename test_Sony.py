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
from dataset.SID import SIDSonyTestDataset

test_input_dir = './dataset/SID/Sony/test/short/'
test_gt_dir = './dataset/SID/Sony/test/long/'
test_list_file= './dataset/SID/Sony_test_list.txt'
checkpoint_dir = './Sony_results/result_Sony_no_ratio/'
result_dir = checkpoint_dir
ckpt = checkpoint_dir + 'model.pth'

# get test IDs
test_fns = glob.glob(test_gt_dir + '*.png')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNet()
unet.load_state_dict(torch.load(ckpt))
unet.to(device)

test_dataset = SIDSonyTestDataset(list_file=test_list_file, root_dir='./dataset/SID/')
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

        # if '_00_' in in_fn:
        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/' + in_fn)

    # for test_id in test_ids:
    #     # test the first image in each sequence
    #     in_files = glob.glob(input_dir + '%05d_00*.png' % test_id)
    #     for k in range(len(in_files)):
    #         in_path = in_files[k]
    #         in_fn = os.path.basename(in_path)
    #         print(in_fn)
    #         gt_files = glob.glob(gt_dir + '%05d_00*.png' % test_id)
    #         gt_path = gt_files[0]
    #         gt_fn = os.path.basename(gt_path)
    #         in_exposure = float(in_fn[9:-5])
    #         gt_exposure = float(gt_fn[9:-5])
    #         ratio = min(gt_exposure / in_exposure, 300)

    #         ratio = int(ratio / 2)
            
    #         input_img = cv2.imread(in_path)
    #         input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    #         input_full = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio
            
    #         # scale_full = np.expand_dims(np.float32(input_img / 255.0), axis=0)

    #         gt_full = cv2.imread(gt_path)
    #         gt_full = cv2.cvtColor(gt_full, cv2.COLOR_BGR2RGB)
    #         gt_full = np.expand_dims(np.float32(gt_full / 255.0), axis=0)

    #         input_full = np.minimum(input_full, 1.0)
    #         in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
    #         gt_img = torch.from_numpy(gt_full).permute(0,3,1,2).to(device)
    #         out_img = unet(in_img)

    #         psnr.append(utils.PSNR(out_img, gt_img))
    #         ssim.append(utils.SSIM(out_img, gt_img))

    #         output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
    #         output = np.minimum(np.maximum(output, 0), 1)
    #         gt_img = gt_img.permute(0, 2, 3, 1).cpu().data.numpy()
    #         gt_img = np.minimum(np.maximum(gt_img, 0), 1)

    #         output = output[0, :, :, :]
    #         gt_full = gt_full[0, :, :, :]
    #         # scale_full = scale_full[0, :, :, :]
    #         # scale_full = scale_full * np.mean(gt_full) / np.mean(
    #         #     scale_full)  # scale the low-light image to the same mean of the groundtruth

    #         scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
    #             result_dir + 'final/' + in_fn)
    #         # scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
    #         #     result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
    #         # scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
    #         #     result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
    #         # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
    #         #     result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))

    