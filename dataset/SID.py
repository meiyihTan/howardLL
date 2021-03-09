import os
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
from torch.nn.functional import interpolate

import matplotlib.pyplot as plt

class SIDSonyTrainDataset(Dataset):
    """SID Sony Train dataset."""
    def __init__(self, list_file ,root_dir, ps, transform=None):
        self.ps = ps
        self.list_file = open(list_file, "r")
        self.list_file_lines = self.list_file.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.gt_images = [None] * 6000
        self.input_images = {}
        self.input_images['150'] = [None] * len(self.list_file_lines)
        self.input_images['125'] = [None] * len(self.list_file_lines)
        self.input_images['50'] = [None] * len(self.list_file_lines)
        self.input_gray_images = {}
        self.input_gray_images['150'] = [None] * len(self.list_file_lines)
        self.input_gray_images['125'] = [None] * len(self.list_file_lines)
        self.input_gray_images['50'] = [None] * len(self.list_file_lines)
        self.input_edge_images = {}
        self.input_edge_images['150'] = [None] * len(self.list_file_lines)
        self.input_edge_images['125'] = [None] * len(self.list_file_lines)
        self.input_edge_images['50'] = [None] * len(self.list_file_lines)

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]
        in_exposure = float(input_img_name[28:-5])
        gt_exposure = float(gt_img_name[27:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        ind = int(input_img_name[19:24])
        ratio = int(ratio / 2)

        in_fn = input_img_name.split('/')[-1]
        
        if self.input_images[str(ratio)[0:3]][ind] is None:
            input_img_path = os.path.join(self.root_dir, input_img_name)
            input_img = cv2.imread(input_img_path)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            self.input_images[str(ratio)[0:3]][ind] = np.expand_dims(np.float32(input_img / 255.0), axis=0) #* ratio

            gt_img_path = os.path.join(self.root_dir, gt_img_name)
            im = cv2.imread(gt_img_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.gt_images[ind] = np.expand_dims(np.float32(im / 255.0), axis=0)

            # gray_path = os.path.join(self.root_dir, 'Sony/train/gray', in_fn)
            # input_gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            # self.input_gray_images[str(ratio)[0:3]][ind] = np.expand_dims(np.expand_dims(np.float32(input_gray / 255.0), axis=2), axis=0)

            # edge_path = os.path.join(self.root_dir, 'Sony/train/edge_GT/%05d_00_%0ds.png' % (ind, int(gt_exposure)))
            edge_path = os.path.join(self.root_dir, 'Sony/train/edge', in_fn)
            input_edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            self.input_edge_images[str(ratio)[0:3]][ind] = np.expand_dims(np.expand_dims(np.float32(input_edge / 255.0), axis=2), axis=0)

        # crop
        H = self.input_images[str(ratio)[0:3]][ind].shape[1]
        W = self.input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        input_patch = self.input_images[str(ratio)[0:3]][ind][:, yy:yy + self.ps, xx:xx + self.ps, :]
        gt_patch = self.gt_images[ind][:, yy:yy + self.ps, xx:xx + self.ps, :]
        # input_gray_patch = self.input_gray_images[str(ratio)[0:3]][ind][:, yy:yy + self.ps, xx:xx + self.ps, :]
        input_edge_patch = self.input_edge_images[str(ratio)[0:3]][ind][:, yy:yy + self.ps, xx:xx + self.ps, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
            # input_gray_patch = np.flip(input_gray_patch, axis=1)
            input_edge_patch = np.flip(input_edge_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
            # input_gray_patch = np.flip(input_gray_patch, axis=2)
            input_edge_patch = np.flip(input_edge_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))
            # input_gray_patch = np.transpose(input_gray_patch, (0, 2, 1, 3))
            input_edge_patch = np.transpose(input_edge_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # input_gray_patch = np.maximum(input_gray_patch, 0.0)
        input_edge_patch = np.maximum(input_edge_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2)
        # in_gray_img = torch.from_numpy(input_gray_patch).permute(0,3,1,2)
        in_edge_img = torch.from_numpy(input_edge_patch).permute(0,3,1,2)

        r,g,b = in_img[0,0,:,:]+1, in_img[0,1,:,:]+1, in_img[0,2,:,:]+1
        in_gray_img = (1.0 - (0.299*r+0.587*g+0.114*b)/2.0).unsqueeze(0).unsqueeze(0)

        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'ind': ind, 'ratio': ratio}
        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'ind': ind, 'ratio': ratio}
        sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'in_edge_img': in_edge_img.squeeze(0), 'ind': ind, 'ratio': ratio}

        return sample

class SIDSonyTestDataset(Dataset):
    """SID Sony Test dataset."""
    def __init__(self, list_file ,root_dir):
        self.list_file = open(list_file, "r")
        self.list_file_lines = self.list_file.readlines()
        self.root_dir = root_dir
        self.gt_images = [None] * 20230

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]
        in_exposure = float(input_img_name[27:-5])
        gt_exposure = float(gt_img_name[26:-5])
        # in_exposure = float(input_img_name[26:-5])
        # gt_exposure = float(gt_img_name[25:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        ind = int(input_img_name[18:23])
        # ind = int(input_img_name[17:22])
        ratio = int(ratio / 2)
        
        if self.gt_images[ind] is None:
            gt_img_path = os.path.join(self.root_dir, gt_img_name)
            im = cv2.imread(gt_img_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.gt_images[ind] = np.expand_dims(np.float32(im / 255.0), axis=0)

        input_img_path = os.path.join(self.root_dir, input_img_name)
        input_img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_full = np.expand_dims(np.float32(input_img / 255.0), axis=0) # * ratio

        in_fn = input_img_name.split('/')[-1]
        # gray_path = os.path.join(self.root_dir, 'Sony/test/gray', in_fn)
        # input_gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        # input_gray_full = np.expand_dims(np.expand_dims(np.float32(input_gray / 255.0), axis=2), axis=0)

        # # edge_path = os.path.join(self.root_dir, 'Sony/test/edge_GT/%05d_00_%0ds.png' % (ind, int(gt_exposure)))
        edge_path = os.path.join(self.root_dir, 'Sony/test/edge_en', in_fn)
        input_edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        input_edge_full = np.expand_dims(np.expand_dims(np.float32(input_edge / 255.0), axis=2), axis=0)

        gt_full = self.gt_images[ind]

        input_full = np.minimum(input_full, 1.0)
        gt_full = np.maximum(gt_full, 0.0)
        # input_gray_full = np.maximum(input_gray_full, 0.0)
        input_edge_full = np.maximum(input_edge_full, 0.0)
        
        in_img = torch.from_numpy(input_full).permute(0,3,1,2)
        gt_img = torch.from_numpy(gt_full).permute(0,3,1,2)
        # in_gray_img = torch.from_numpy(input_gray_full).permute(0,3,1,2)
        in_edge_img = torch.from_numpy(input_edge_full).permute(0,3,1,2)

        r,g,b = in_img[0,0,:,:]+1, in_img[0,1,:,:]+1, in_img[0,2,:,:]+1
        in_gray_img = (1.0 - (0.299*r+0.587*g+0.114*b)/2.0).unsqueeze(0).unsqueeze(0)

        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'ind': ind, 'ratio': ratio, 'in_fn': in_fn}
        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'ind': ind, 'ratio': ratio, 'in_fn': in_fn}
        sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'in_edge_img': in_edge_img.squeeze(0), 'ind': ind, 'ratio': ratio, 'in_fn': in_fn}

        return sample

class SIDSonyValDataset(Dataset):
    """SID Sony Val dataset."""
    def __init__(self, list_file ,root_dir):
        self.list_file = open(list_file, "r")
        self.list_file_lines = self.list_file.readlines()
        self.root_dir = root_dir
        self.gt_images = [None] * 20250
        self.input_images = {}
        self.input_images['150'] = [None] * 20250
        self.input_images['125'] = [None] * 20250
        self.input_images['50'] = [None] * 20250
        self.input_original_images = [None] * 20250
        # self.input_gray_images = {}
        # self.input_gray_images['150'] = [None] * 20250
        # self.input_gray_images['125'] = [None] * 20250
        # self.input_gray_images['50'] = [None] * 20250
        # self.input_edge_images = {}
        # self.input_edge_images['150'] = [None] * 20250
        # self.input_edge_images['125'] = [None] * 20250
        # self.input_edge_images['50'] = [None] * 20250

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]
        in_exposure = float(input_img_name[26:-5])
        gt_exposure = float(gt_img_name[25:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        ind = int(input_img_name[17:22])
        ratio = int(ratio / 2)

        
        if self.input_images[str(ratio)[0:3]][ind] is None:
            input_img_path = os.path.join(self.root_dir, input_img_name)
            # print(input_img_path)
            input_img = cv2.imread(input_img_path)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            self.input_images[str(ratio)[0:3]][ind] = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio
            self.input_original_images[ind] = np.expand_dims(np.float32(input_img / 255.0), axis=0)

            gt_img_path = os.path.join(self.root_dir, gt_img_name)
            im = cv2.imread(gt_img_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.gt_images[ind] = np.expand_dims(np.float32(im / 255.0), axis=0)

            # gray_path = os.path.join(self.root_dir, 'Sony/train/gray/%05d_00_%0d.png' % (ind, ratio))
            # input_gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            # self.input_gray_images[str(ratio)[0:3]][ind] = np.expand_dims(np.expand_dims(np.float32(input_gray / 255.0), axis=2), axis=0)

            # edge_path = os.path.join(self.root_dir, 'Sony/train/edge_en/%05d_00_%0d.png' % (ind, ratio))
            # # edge_path = os.path.join(self.root_dir, 'Sony/train/edge_GT/%05d_00_%0ds.png' % (ind, int(gt_exposure)))
            # input_edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            # self.input_edge_images[str(ratio)[0:3]][ind] = np.expand_dims(np.expand_dims(np.float32(input_edge / 255.0), axis=2), axis=0)

        input_patch = self.input_images[str(ratio)[0:3]][ind]
        gt_patch = self.gt_images[ind]
        input_original_patch = self.input_original_images[ind]
        # input_gray_patch = self.input_gray_images[str(ratio)[0:3]][ind]
        # input_edge_patch = self.input_edge_images[str(ratio)[0:3]][ind]

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        input_original_patch = np.minimum(input_original_patch, 0.0)
        # input_gray_patch = np.maximum(input_gray_patch, 0.0)
        # input_edge_patch = np.maximum(input_edge_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2)
        in_original_img = torch.from_numpy(input_original_patch).permute(0,3,1,2)
        # in_gray_img = torch.from_numpy(input_gray_patch).permute(0,3,1,2)
        # in_edge_img = torch.from_numpy(input_edge_patch).permute(0,3,1,2)

        # r,g,b = in_img[0,0,:,:]+1, in_img[0,1,:,:]+1, in_img[0,2,:,:]+1
        # in_gray_img = (1.0 - (0.299*r+0.587*g+0.114*b)/2.0).unsqueeze(0).unsqueeze(0)

        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'ind': ind, 'ratio': ratio}
        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'ind': ind, 'ratio': ratio}
        sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_original_img': in_original_img.squeeze(0), 'ind': ind, 'ratio': ratio}
        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'in_edge_img': in_edge_img.squeeze(0), 'ind': ind, 'ratio': ratio}
        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'in_edge_img': in_edge_img.squeeze(0), 'in_original_img': in_original_img.squeeze(0), 'ind': ind, 'ratio': ratio}

        return sample

class SIDFujiDataset(Dataset):
    """SID Fuji dataset."""
    def __init__(self, list_file ,root_dir, ps,transform=None):
        self.ps = ps
        self.list_file = open(list_file, "r")
        self.list_file_lines = self.list_file.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.gt_images = [None] * 6000
        self.input_images = {}
        self.input_images['150'] = [None] * len(self.list_file_lines)
        self.input_images['125'] = [None] * len(self.list_file_lines)
        self.input_images['50'] = [None] * len(self.list_file_lines)
        self.input_edge_images = {}
        self.input_edge_images['150'] = [None] * len(self.list_file_lines)
        self.input_edge_images['125'] = [None] * len(self.list_file_lines)
        self.input_edge_images['50'] = [None] * len(self.list_file_lines)

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]
        in_exposure = float(input_img_name[28:-5])
        gt_exposure = float(gt_img_name[27:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        ind = int(input_img_name[19:24])
        ratio = int(ratio / 2)

        in_fn = input_img_name.split('/')[-1]
        
        if self.input_images[str(ratio)[0:3]][ind] is None:
            input_img_path = os.path.join(self.root_dir, input_img_name)
            input_img = cv2.imread(input_img_path)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            self.input_images[str(ratio)[0:3]][ind] = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio
            _, h, w, c = self.input_images[str(ratio)[0:3]][ind].shape

            gt_img_path = os.path.join(self.root_dir, gt_img_name)
            im = cv2.imread(gt_img_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.gt_images[ind] = np.expand_dims(np.float32(im / 255.0), axis=0)

            # edge_path = os.path.join(self.root_dir, 'Fuji/train/edge_en/', in_fn)
            # input_edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            # input_edge = cv2.resize(input_edge, (w, h))
            # self.input_edge_images[str(ratio)[0:3]][ind] = np.expand_dims(np.expand_dims(np.float32(input_edge / 255.0), axis=2), axis=0)

        # crop
        H = self.input_images[str(ratio)[0:3]][ind].shape[1]
        W = self.input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        input_patch = self.input_images[str(ratio)[0:3]][ind][:, yy:yy + self.ps, xx:xx + self.ps, :]
        gt_patch = self.gt_images[ind][:, yy:yy + self.ps, xx:xx + self.ps, :]
        # input_edge_patch = self.input_edge_images[str(ratio)[0:3]][ind][:, yy:yy + self.ps, xx:xx + self.ps, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
            # input_edge_patch = np.flip(input_edge_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
            # input_edge_patch = np.flip(input_edge_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))
            # input_edge_patch = np.transpose(input_edge_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # input_edge_patch = np.maximum(input_edge_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2)
        # in_edge_img = torch.from_numpy(input_edge_patch).permute(0,3,1,2)

        # r,g,b = in_img[0,0,:,:]+1, in_img[0,1,:,:]+1, in_img[0,2,:,:]+1
        # in_gray_img = (1.0 - (0.299*r+0.587*g+0.114*b)/2.0).unsqueeze(0).unsqueeze(0)

        sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'ind': ind, 'ratio': ratio}
        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'ind': ind, 'ratio': ratio}
        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'in_edge_img': in_edge_img.squeeze(0), 'ind': ind, 'ratio': ratio}

        return sample

class SIDFujiTestDataset(Dataset):
    """SID Fuji Test dataset."""
    def __init__(self, list_file ,root_dir):
        self.list_file = open(list_file, "r")
        self.list_file_lines = self.list_file.readlines()
        self.root_dir = root_dir
        self.gt_images = [None] * 10200

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]
        in_exposure = float(input_img_name[27:-5])
        gt_exposure = float(gt_img_name[26:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        ind = int(input_img_name[18:23])
        ratio = int(ratio / 2)
        
        if self.gt_images[ind] is None:
            gt_img_path = os.path.join(self.root_dir, gt_img_name)
            im = cv2.imread(gt_img_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.gt_images[ind] = np.expand_dims(np.float32(im / 255.0), axis=0)

        input_img_path = os.path.join(self.root_dir, input_img_name)
        input_img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        h, w, c = input_img.shape
        input_img = cv2.resize(input_img, (5120, 3072), interpolation = cv2.INTER_AREA)

        input_full = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio

        in_fn = input_img_name.split('/')[-1]
        # edge_path = os.path.join(self.root_dir, 'Fuji/test/edge_en', in_fn)
        # input_edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        # input_edge = cv2.resize(input_edge, (5120, 3072))
        # input_edge_full = np.expand_dims(np.expand_dims(np.float32(input_edge / 255.0), axis=2), axis=0)

        gt_full = self.gt_images[ind]

        input_full = np.minimum(input_full, 1.0)
        gt_full = np.maximum(gt_full, 0.0)
        # input_edge_full = np.maximum(input_edge_full, 0.0)
        
        in_img = torch.from_numpy(input_full).permute(0,3,1,2)
        gt_img = torch.from_numpy(gt_full).permute(0,3,1,2)
        # in_edge_img = torch.from_numpy(input_edge_full).permute(0,3,1,2)

        # r,g,b = in_img[0,0,:,:]+1, in_img[0,1,:,:]+1, in_img[0,2,:,:]+1
        # in_gray_img = (1.0 - (0.299*r+0.587*g+0.114*b)/2.0).unsqueeze(0).unsqueeze(0)

        sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'ind': ind, 'ratio': ratio, 'in_fn': in_fn, 'width': w, 'hight': h}
        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'ind': ind, 'ratio': ratio, 'in_fn': in_fn, 'width': w, 'hight': h}
        # sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'in_edge_img': in_edge_img.squeeze(0), 'ind': ind, 'ratio': ratio, 'in_fn': in_fn, 'width': w, 'hight': h}

        return sample