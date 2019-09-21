import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'LMC':
            feature = np.vstack((self.dataset[index]['logmelspec'],
                                 self.dataset[index]['chroma'],
                                 self.dataset[index]['tonnetz'],
                                 self.dataset[index]['spectral_contrast']))
            feature = torch.from_numpy(feature)
        elif self.mode == 'MC':
            feature = np.vstack((self.dataset[index]['mfcc'],
                                 self.dataset[index]['chroma'],
                                 self.dataset[index]['tonnetz'],
                                 self.dataset[index]['spectral_contrast']))
            feature = torch.from_numpy(feature)
        elif self.mode == 'MLMC':
            feature = np.vstack((self.dataset[index]['logmelspec'],
                                 self.dataset[index]['mfcc'],
                                 self.dataset[index]['chroma'],
                                 self.dataset[index]['tonnetz'],
                                 self.dataset[index]['spectral_contrast']))
            feature = torch.from_numpy(feature)
        elif self.mode == 'LMC+MC':
            lmc = np.vstack((self.dataset[index]['logmelspec'],
                             self.dataset[index]['chroma'],
                             self.dataset[index]['tonnetz'],
                             self.dataset[index]['spectral_contrast']))

            mc = np.vstack((self.dataset[index]['mfcc'],
                            self.dataset[index]['chroma'],
                            self.dataset[index]['tonnetz'],
                            self.dataset[index]['spectral_contrast']))
            feature = (torch.from_numpy(lmc), torch.from_numpy(mc))
        label = self.dataset[index]['classID']
        return feature, label

    def __len__(self):
        return len(self.dataset)
