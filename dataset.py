from torchvision.datasets import MNIST
import numpy as np


class FastMNIST(MNIST):
    """
    Improves performance of training on MNIST by removing the PIL interface and pre-loading on the GPU (2-3x speedup)

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', 'cpu')
        train_class = kwargs.pop('train_class', 'all')
        split = kwargs.pop('split', 'train')
        super().__init__(*args, **kwargs)

        self.split = split

        if self.train:
            if not isinstance(train_class, str):
                print(train_class)
                self.targets = np.array(self.targets)
                index_class = np.isin(self.targets, train_class)
                self.data = self.data[index_class]
                self.targets = self.targets[index_class]
                self.len = self.data.shape[0]

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.to(device)

        self.targets = self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
