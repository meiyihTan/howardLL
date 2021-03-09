from __future__ import division
import os, scipy.io, scipy.misc, cv2
import numpy as np
import glob
import math
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

# input_dirs = [None] * 1
# input_dirs[0] = './Fuji_results/result_Fuji/final/'
# # input_dirs[1] = './Fuji_results/result_Fuji_gray_map/final/'
# # input_dirs[0] = './Fuji_results/result_Fuji_gray_map_enhanced_edge_map/final/'
# gt_dir = './dataset/SID/Fuji/test/long/'

# for input_dir in input_dirs:
#      # get filenames
#      fns = sorted(glob.glob(input_dir + '*.png'))

#      input_dir_len = len(input_dir)
#      psnrs, ssims = [], []
#      gt_images = [None] * 20000

#      for fn in fns:
#           idx = int(fn[input_dir_len:input_dir_len+5])
#           in_fn = fn
#           gt_fn = gt_dir + str(idx) + '_00_10s.png'
          
#           in_img = cv2.imread(in_fn)
#           if gt_images[idx] is None:
#                gt_images[idx] = cv2.imread(gt_fn)
          
#           psnrs.append(compare_psnr(gt_images[idx], in_img))
#           ssims.append(compare_ssim(gt_images[idx], in_img, multichannel=True))

#      print(np.mean(psnrs))
#      print(np.mean(ssims))

#      txt_f = open(input_dir + '../psnr_ssim.txt', 'a')
#      psnr_ssim = '%.2f/%.3f' % (np.mean(psnrs), np.mean(ssims))
#      txt_f.write(psnr_ssim)
#      txt_f.close()

# Sony
input_dirs = [None] * 1
input_dirs[0] = './Sony_results/result_Sony_gray_map_enhanced_edge_map_no_ratio/final/'
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
# input_dirs = [None] * 1
# # input_dirs[0] = './IC15_004_results_supp/result_IC15_gray_map_enhanced_edge_map_MSSSIMLoss_TextLoss_1_1_1/final/'
# input_dirs[0] = './IC15_004_results/result_IC15_gray_map_enhanced_edge_map_no_ratio/final/'
# gt_dir = './dataset/ICDAR15/test/high/'

# for input_dir in input_dirs:
#      # get filenames
#      fns = sorted(glob.glob(input_dir + '*.jpg'))

#      input_dir_len = len(input_dir)
#      psnrs, ssims = [], []

#      cnt = 0
#      for fn in fns:
#           in_fn = fn
#           fn_idx = in_fn.split('/')[-1]
#           gt_fn = gt_dir + str(fn_idx)
          
#           in_img = cv2.imread(in_fn)
#           gt_img = cv2.imread(gt_fn)
          
#           psnrs.append(compare_psnr(gt_img, in_img))
#           ssims.append(compare_ssim(gt_img, in_img, multichannel=True))
          
#      print(np.mean(psnrs))
#      print(np.mean(ssims))

     # txt_f = open(input_dir + '../psnr_ssim.txt', 'a')
     # psnr_ssim = '%.2f/%.3f' % (np.mean(psnrs), np.mean(ssims))
     # txt_f.write(psnr_ssim)
     # txt_f.close()