# coding=utf-8
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from config import args  # Import training configuration arguments
from tqdm import tqdm  # Progress bar library
import data_preprocess  # Custom data preprocessing module


class DatasetCustom(Dataset):
    """
    Custom dataset class for handling training, validation, and test datasets.

    Attributes:
    - xs (list of str): Paths to input data files.
    - ys (list of str): Paths to corresponding label files.
    - flag (str): Indicates the mode (training, validation, or test).
    """
    def __init__(self, x_paths, y_paths):
        """
        Initialize the dataset instance.

        Parameters:
        - x_paths (list of str): Paths to input feature files.
        - y_paths (list of str): Paths to target label files.
        """
        self.xs = [np.load(x_path) for x_path in x_paths]
        self.ys = [np.load(y_path) for y_path in y_paths]

    def __getitem__(self, index):
        """
        Retrieve a single data sample and its label.

        Parameters:
        - index (int): Index of the data sample to retrieve.

        Returns:
        - tuple: Normalized input data and corresponding label.
        """
        assert index in range(0, len(self.xs)), "Index out of range"
        _x = self.normalize_01(self.xs[index])[None, :]  # Normalize input data to range [0, 1]
        _y = self.ys[index].reshape(1, -1) # Retrieve corresponding label
        return _x, _y
    
    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
        - int: Number of samples.
        """
        return len(self.xs)
    
    @staticmethod
    def normalize_01(x):
        """
        Normalize data to the range [0, 1].

        Parameters:
        - x (array-like): Input data.

        Returns:
        - array-like: Normalized data.
        """
        _min = x.min()
        _max = x.max()
        if _max != _min:
            return (x - _min) / (_max - _min)
        else:
            return x / _min if _min != 0 else x  # Avoid division by zero
    
    @staticmethod
    def normalize_255(x):
        """
        Normalize data to the range [0, 255].

        Parameters:
        - x (array-like): Input data.

        Returns:
        - array-like: Normalized data.
        """
        return x / 255
