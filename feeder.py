import numpy as np
import torch
import h5py


class Feeder(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()

    def load_data(self):
        with open(self.data_path, 'rb') as reader:
            hf = h5py.File(reader)
            self.feature = np.array(hf["all_data"])
            self.adjacency = np.array(hf["all_adjacency"])
            self.mean_xy = np.array(hf["all_mean_xy"])
            self.feature[:, [5], :, :] = self.feature[:, [5], :, :] / 23.68296
            self.feature[:, [6], :, :] = self.feature[:, [6], :, :] / 2.5908
            self.feature[:, [7], :, :] = self.feature[:, [7], :, :] / 95.3

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        feature = self.feature[idx]
        adjacency = self.adjacency[idx]
        mean_xy = self.mean_xy[idx]

        return feature, adjacency, mean_xy
