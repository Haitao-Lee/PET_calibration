import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class DeeplabDataset(Dataset):
    def __init__(self, flood_lines, peak_lines, input_shape, num_classes, dataset_path, set):
        super(DeeplabDataset, self).__init__()
        self.flood_lines   = flood_lines
        self.peak_lines = peak_lines
        self.length             = len(flood_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.dataset_path       = dataset_path
        self.set = set

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        flood_name = self.flood_lines[index].split()[0]
        peak_name = self.peak_lines[index].split()[0]

        flood = np.load(os.path.join(self.dataset_path, self.set, 'flood_GY_npy', flood_name + ".npy")).astype(np.float32)
        peak = np.load(os.path.join(self.dataset_path, self.set, 'peak_npy', peak_name+'.npy')).astype(np.uint8)
        floods = np.stack((flood,) * 3, axis=0)

        seg_labels  = np.eye(self.num_classes + 1)[peak.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return floods, peak, seg_labels



# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    floods      = []
    peaks        = []
    seg_labels  = []
    for flood, peak, labels in batch:
        floods.append(flood)
        peaks.append(peak)
        seg_labels.append(labels)
    floods      = torch.from_numpy(np.array(floods)).type(torch.FloatTensor)
    peaks        = torch.from_numpy(np.array(peaks)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return floods, peaks, seg_labels
