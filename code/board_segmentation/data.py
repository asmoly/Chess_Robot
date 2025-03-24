import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

PATH_TO_TRAIN_DATA = "data/train/_annotations.coco.json"
PATH_TO_IMAGES = "data/train"
MAX_ID = 1447

class ChessboardDataset(Dataset):
    def __init__(self, path_to_data_annotations, path_to_images, max_id, transform=None):
        self.max_id = max_id
        self.path_to_data_annotations = path_to_data_annotations
        self.path_to_images = path_to_images
        self.transform = transform

        self.image_annotations = np.empty((MAX_ID + 1, 4, 2))
        self.image_names = np.empty((MAX_ID + 1), dtype=object)

        f = open(PATH_TO_TRAIN_DATA)
        data = json.load(f)

        print("Loading Dataset ...")

        for image in data["images"]:
            self.image_names[image["id"]] = str(image["file_name"])

        for image in data["annotations"]:
            segmentation = np.array(image["segmentation"][0]).reshape((int(len(image["segmentation"][0])/2), 2)).astype(np.int64)
            segmentation = ChessboardDataset.approximate_segmentation_to_four_points(segmentation)

            self.image_annotations[image["id"]] = segmentation.reshape(4, 2)

        print("Loaded Dataset")

    def __len__(self):
        return self.max_id + 1

    @staticmethod
    def approximate_segmentation_to_four_points(segmentation, max_iterations=100):
        segmentation = segmentation.reshape((-1, 1, 2))
        
        epsilon = 1.0
        iterations = 0

        while iterations < max_iterations:
            approximated_polygon = cv2.approxPolyDP(segmentation, epsilon, True)

            if len(approximated_polygon) == 4:
                return approximated_polygon

            if len(approximated_polygon) > 4:
                epsilon += 1.0
            else:
                epsilon -= 0.1

            iterations += 1

        return approximated_polygon

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_segmentations = self.image_annotations[idx]

        image = cv2.imread(os.path.join(self.path_to_images, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, image_segmentations
    
data = ChessboardDataset(path_to_data_annotations=PATH_TO_TRAIN_DATA, path_to_images=PATH_TO_IMAGES, max_id=MAX_ID)

for i in range (0, 100):
    image, image_segmentations = data.__getitem__(i)
    image_segmentations = image_segmentations.reshape((-1, 1, 2)).astype(np.int64)
    
    cv2.polylines(image, [image_segmentations], isClosed=False, color=(0, 255, 0), thickness=3)

    cv2.imshow("main", image)
    cv2.waitKey(0)