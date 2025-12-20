from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>4I', f.read(16))
            images = np.frombuffer(f.read(), dtype = np.uint8)
            images = images.reshape((num, rows, cols, 1))
            images = images.astype(np.float32) / 255.0

        with gzip.open(label_filename, 'rb') as f:
            magic, num = struct.unpack('>2I', f.read(8))
            labels = np.frombuffer(f.read(), dtype = np.uint8)
            labels = labels.reshape((num, ))
        
        self.images = images
        self.labels = labels
        self.transforms = transforms

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        label = self.labels[index]
        
        img = self.apply_transforms(img)

        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION