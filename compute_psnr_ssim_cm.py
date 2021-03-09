from __future__ import division
import os, scipy.io, scipy.misc, cv2
import numpy as np
import glob
import math
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

input_dirs = [None] * 1
input_dirs[0] = './compare_methods/pix2pix/Sony/'
# input_dirs[1] = './compare_methods/SRIE/Sony_exposed/'
# input_dirs[2] = './compare_methods/BIMEF/Sony_exposed/'
# input_dirs[3] = './compare_methods/CycleGAN/Sony_exposed/'
# input_dirs[4] = './compare_methods/EnlightenGAN/Sony_exposed/'
gt_dir = './dataset/SID/Sony/test/long/'

for input_dir in input_dirs:
     # get filenames
     fns = sorted(glob.glob(input_dir + '*.png'))

     input_dir_len = len(input_dir)
     psnrs, ssims = [], []
     gt_images = [None] * 20000

     for fn in fns:
          idx = int(fn[input_dir_len:input_dir_len+5])
          in_fn = fn
          # gt_fn = in_fn.replace('out', 'gt')
          # fn_idx = in_fn.split('final/')[-1]
          if idx > 10070 and idx < 10177:
               gt_fn = gt_dir + str(idx) + '_00_30s.png'
          else:
               gt_fn = gt_dir + str(idx) + '_00_10s.png'
          
          in_img = cv2.imread(in_fn)
          if gt_images[idx] is None:
               gt_images[idx] = cv2.imread(gt_fn)
          
          psnrs.append(compare_psnr(gt_images[idx], in_img))
          ssims.append(compare_ssim(gt_images[idx], in_img, multichannel=True))

     print(np.mean(psnrs))
     print(np.mean(ssims))

#      txt_f = open(input_dir + '../psnr_ssim.txt', 'a')
#      psnr_ssim = '%.2f/%.3f' % (np.mean(psnrs), np.mean(ssims))
#      txt_f.write(psnr_ssim)
#      txt_f.close()

# # ICDAR15     
# input_dir = './compare_methods/pix2pix/IC15/'
# gt_dir = './dataset/ICDAR15/test/high/'

# # get filenames
# fns = sorted(glob.glob(input_dir + '*.png'))

# input_dir_len = len(input_dir)
# psnrs, ssims = [], []

# cnt = 0
# for fn in fns:
#      in_fn = fn
#      fn_idx = in_fn.split('/')[-1]
#      fn_idx = fn_idx.split('.')[0]
#      # fn_idx = fn_idx.split('_fake')[0]
#      gt_fn = gt_dir + str(fn_idx) + '.jpg'
     
#      in_img = cv2.imread(in_fn)
#      gt_img = cv2.imread(gt_fn)
     
#      psnrs.append(compare_psnr(gt_img, in_img))
#      ssims.append(compare_ssim(gt_img, in_img, multichannel=True))

# print(np.mean(psnrs))
# print(np.mean(ssims))

# txt_f = open(input_dir + '../psnr_ssim.txt', 'a')
# psnr_ssim = '%.2f/%.3f' % (np.mean(psnrs), np.mean(ssims))
# txt_f.write(psnr_ssim)
# txt_f.close()