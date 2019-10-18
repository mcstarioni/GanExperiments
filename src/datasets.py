import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from src.utils import to_tensors
import numpy as np


class NumpyDataset(Dataset):
    def __init__(self, *data):
        super().__init__()
        self.arraylist = list(data)
        self.tensorlist = to_tensors(self.arraylist)
        self.sample_size = data[0][0].shape

    def __len__(self):
        return len(self.tensorlist[0])

    def __getitem__(self, item):
        return [data[item] for data in self.tensorlist]

    def visualize(self, ax=None, **kwargs):
        if ax is None:
            ax = plt
        for data in self.arraylist:
            ax.scatter(data[:, 0], data[:, 1], **kwargs)


class GaussianGrid(NumpyDataset):
    def __init__(self, size=4000, rows=5, cols=5, scale=20, variance=0.1):
        M = int(size/(rows*cols))
        y_step = 2*scale/(rows-1)
        x_step = 2*scale/(cols-1)
        print(y_step, x_step)
        data = np.array([[np.random.multivariate_normal(mean=(x_step*i - scale, y_step*j - scale),
                                                        cov=[[variance, 0], [0, variance]],
                                                        size=M)
                          for j in range(cols)]
                         for i in range(rows)])
        c1, c2, M, N = data.shape
        data = data.reshape((c1 * c2 * M, N))
        super().__init__(data)


class GaussianCircle(NumpyDataset):
    def __init__(self, size=4000, clusters=9, scale=20, variance=0.1):
        M = int(size/clusters)
        step = (2 * np.pi) / clusters
        angles = iter([i*step for i in range(clusters)])
        data = np.array([np.random.multivariate_normal(mean=(0, 0),
                                                       cov=[[variance, 0], [0, variance]],
                                                       size=M) + scale * np.array([[np.cos(a),
                                                                                 np.sin(a)]])
                          for a in angles])
        c1, M, N = data.shape
        data = data.reshape((c1 * M, N))
        super().__init__(data)


class GaussianSpiral(NumpyDataset):
    def __init__(self, size=4000, scale=20, rotations=2):
        noise = np.random.normal(size=(size, 2))
        data = []
        angles = np.linspace(0, rotations*np.pi, size)
        scale = scale/(rotations*np.pi)
        for theta in angles:
            x = theta*np.cos(theta)*scale
            y = theta*np.sin(theta)*scale
            data.append([x, y])
        data = np.array(data) + 3*noise/rotations
        super().__init__(data)

