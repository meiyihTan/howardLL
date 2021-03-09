# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io, scipy.misc, cv2, time
import torch
import numpy as np
import glob
import utils

from unet import GrayEdgeAttentionUNet
from torch.utils.data import DataLoader
from dataset.SID import SIDSonyTestDataset

input_dir = './dataset/SID/Sony/test/short/'
gt_dir = './dataset/SID/Sony/test/long/'
edge_dir = './dataset/SID/Sony/test/edge_en/'
list_file= './dataset/SID/Sony_test_list.txt'
checkpoint_dir = './Sony_results/result_Sony_gray_map_enhanced_edge_map_no_ratio/'
result_dir = checkpoint_dir
ckpt = checkpoint_dir + 'model.pth'

# get test IDs
test_fns = glob.glob(gt_dir + '*.png')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = GrayEdgeAttentionUNet()
unet.load_state_dict(torch.load(ckpt))
unet.to(device)

total_params = sum(p.numel() for p in unet.parameters())
print(total_params)

test_dataset = SIDSonyTestDataset(list_file=list_file, root_dir='./dataset/SID/')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
iteration = 0

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

total_time = 0
count = 0

with torch.no_grad():
    unet.eval()
    psnr, ssim = [], []

    for sample in iter(test_dataloader):
        
        in_fn = sample['in_fn'][0]
        print(in_fn)
        in_img = sample['in_img'].to(device)
        gt_img = sample['gt_img'].to(device)
        in_gray_img = sample['in_gray_img'].to(device)
        in_edge_img = sample['in_edge_img'].to(device)
        
        # st = time.time()
        out_img = unet(in_img, in_gray_img, in_edge_img)
        # print(time.time() - st)
        # total_time += time.time() - st
        # count += 1.0

        # psnr.append(utils.PSNR(out_img, gt_img).item())
        # ssim.append(utils.SSIM(out_img, gt_img).item())

        output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
        output = np.minimum(np.maximum(output, 0), 1)
        gt_img = gt_img.permute(0, 2, 3, 1).cpu().data.numpy()
        gt_img = np.minimum(np.maximum(gt_img, 0), 1)

        output = output[0, :, :, :]
        gt_img = gt_img[0, :, :, :]

        # if '_00_' in in_fn:
        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/' + in_fn)
    # print(total_time / count)
    # print('PSNR=%.2f SSIM=%.3f' % (np.mean(psnr), np.mean(ssim)))
    
    # txt_f = open(result_dir + 'psnr_ssim.txt', 'a')
    # psnr_ssim = '%.2f/%.3f' % (np.mean(psnr), np.mean(ssim))
    # txt_f.write(psnr_ssim)
    # txt_f.close()
    
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
    #         gray_path = os.path.join(gray_dir + '%05d_00_%0d.png' % (test_id, ratio))
    #         edge_path = os.path.join(edge_dir + '%05d_00_%0d.png' % (test_id, ratio))
    #         # edge_path = os.path.join(edge_dir + '%05d_00_%0ds.png' % (test_id, int(gt_exposure)))
            
    #         input_img = cv2.imread(in_path)
    #         input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    #         input_full = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio
            
    #         # scale_full = np.expand_dims(np.float32(input_img / 255.0), axis=0)

    #         gt_full = cv2.imread(gt_path)
    #         gt_full = cv2.cvtColor(gt_full, cv2.COLOR_BGR2RGB)
    #         gt_full = np.expand_dims(np.float32(gt_full / 255.0), axis=0)

    #         gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    #         gray_img = np.expand_dims(gray_img, axis=2)
    #         gray_full = np.expand_dims(np.float32(gray_img / 255.0), axis=0)

    #         edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    #         edge_img = np.expand_dims(edge_img, axis=2)
    #         edge_full = np.expand_dims(np.float32(edge_img / 255.0), axis=0)

    #         input_full = np.minimum(input_full, 1.0)
    #         in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
    #         gray_img = torch.from_numpy(gray_full).permute(0,3,1,2).to(device)
    #         edge_img = torch.from_numpy(edge_full).permute(0,3,1,2).to(device)
    #         out_img = unet(in_img, gray_img, edge_img)
    #         output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
    #         output = np.minimum(np.maximum(output, 0), 1)

    #         output = output[0, :, :, :]
    #         gt_full = gt_full[0, :, :, :]
    #         # scale_full = scale_full[0, :, :, :]
    #         # scale_full = scale_full * np.mean(gt_full) / np.mean(
    #         #     scale_full)  # scale the low-light image to the same mean of the groundtruth
            
    #         scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
    #             result_dir + 'final/' + in_fn)
    #         # scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
    #         #     result_dir + 'final/%05d_00_%d_out.png' % (test_id, ratio))
    #         # scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
    #         #     result_dir + 'final/%05d_00_%d_scale.png' % (test_id, ratio))
    #         # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
    #         #     result_dir + 'final/%05d_00_%d_gt.png' % (test_id, ratio))