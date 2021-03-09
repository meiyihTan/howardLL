# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io, scipy.misc, cv2
import torch
import numpy as np
import glob

from unet import GrayEdgeAttentionUNet

input_dir = './dataset/IC15_004/test/low/'
gt_dir = './dataset/IC15_004/test/high/'
edge_dir = './dataset/IC15_004/test/edge_en/'
list_file= './dataset/IC15_004/test_list.txt'
checkpoint_dir = './IC15_004_results/result_IC15_gray_map_enhanced_edge_map_no_ratio/'
result_dir = checkpoint_dir
ckpt = checkpoint_dir + 'model.pth'

# get test IDs
test_fns = glob.glob(gt_dir + '*.jpg')
test_ids = [int(((os.path.basename(test_fn).split('/')[-1]).split('.')[0])) for test_fn in test_fns]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = GrayEdgeAttentionUNet()
unet.load_state_dict(torch.load(ckpt))
unet.to(device)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

with torch.no_grad():
    unet.eval()
    for test_id in test_ids:
        in_file = input_dir + '%d.jpg' % test_id
        print(in_file)
        in_path = in_file
        gt_file = gt_dir + '%d.jpg' % test_id
        gt_path = gt_file
        ratio = 1

        edge_path = edge_dir + '%d.png' % test_id
        
        input_img = cv2.imread(in_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_full = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio
        
        # scale_full = np.expand_dims(np.float32(input_img / 255.0), axis=0)

        gt_full = cv2.imread(gt_path)
        gt_full = cv2.cvtColor(gt_full, cv2.COLOR_BGR2RGB)
        gt_full = np.expand_dims(np.float32(gt_full / 255.0), axis=0)

        input_edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        input_edge_full = np.expand_dims(np.expand_dims(np.float32(input_edge_img / 255.0), axis=2), axis=0)

        input_full = np.minimum(input_full, 1.0)
        in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)

        r,g,b = in_img[0,0,:,:]+1, in_img[0,1,:,:]+1, in_img[0,2,:,:]+1
        in_gray_img = (1.0 - (0.299*r+0.587*g+0.114*b)/2.0).unsqueeze(0).unsqueeze(0).to(device)

        in_edge_img = torch.from_numpy(input_edge_full).permute(0,3,1,2).to(device)
        
        out_img = unet(in_img, in_gray_img, in_edge_img)
        output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
        output = np.minimum(np.maximum(output, 0), 1)

        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]
        # scale_full = scale_full[0, :, :, :]
        # scale_full = scale_full * np.mean(gt_full) / np.mean(
        #     scale_full)  # scale the low-light image to the same mean of the groundtruth

        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%d.jpg' % (test_id))
        # scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     result_dir + 'final/%d_%d_out.png' % (test_id, ratio))
        # scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     result_dir + 'final/%d_%d_scale.png' % (test_id, ratio))
        # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     result_dir + 'final/%d_%d_gt.png' % (test_id, ratio))