import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

# Order of data list [BB, BK, BKi, BP, BQ, BR, Empty, WB, WK, WKi, WP, WQ, WR]

dataTransformations = transforms.Compose([
    transforms.ColorJitter(brightness=(0.5,1.0)),
])

class ChessboardDataset(Dataset):
    def __init__(self, path_to_data, transform=None):
        self.data = []
        
        for piece_directory in os.listdir(path_to_data):
            full_path = os.path.join(path_to_data, piece_directory)

            if os.path.isdir(full_path):
                self.data.append([])

                for image in os.listdir(full_path):
                    self.data[len(self.data) - 1].append(os.path.join(full_path, image))

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

data = ChessboardDataset("data/")