import os
import datetime

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import AdamW
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import cv2
from math import cos, sin, pi

from torch.utils.tensorboard import SummaryWriter

from data import *
from models import *

d_model = 16
img_size = (640, 640)
patch_size = (16, 16)
n_channels = 3
n_heads = 4
n_sa_blocks = 8
start_epoch = 0
n_epochs = 100
learning_rate = 0.001 # 0.005 -- works
val_every_n_epoch = 3
log_every_n_batches = 10

train_batch_size = 20
test_batch_size = 1

MAX_TRAIN_ID = 1447
MAX_TEST_ID = 19

PATH_TO_TRAIN_ANNOTATIONS = "data/train/_annotations.coco.json"
PATH_TO_TEST_ANNOTATIONS = "data/valid/_annotations.coco.json"
PATH_TO_TRAIN_IMAGES = "data/train"
PATH_TO_TEST_IMAGES = "data/valid"

PATH_TO_LOGS = "logs/run"
PATH_TO_MODEL = None

def load_model(path):
    model = BoardSegmentTransformer(d_model, img_size, patch_size, n_channels, n_heads, n_sa_blocks)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

# Main function
def main():
    # Create new log directory
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_path = f"{PATH_TO_LOGS}_{timestamp}"
    writer = SummaryWriter(log_dir=log_path)

    # Create datasets
    train_dataset = ChessboardDataset(PATH_TO_TRAIN_ANNOTATIONS, PATH_TO_TRAIN_IMAGES, MAX_TRAIN_ID, transform=dataTransformations)
    test_dataset = ChessboardDataset(PATH_TO_TEST_ANNOTATIONS, PATH_TO_TEST_IMAGES, MAX_TEST_ID, transform=dataTransformations)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    # Create arm model
    board_transformer_model = BoardSegmentTransformer(d_model, img_size, patch_size, n_channels, n_heads, n_sa_blocks).to(device)
    if PATH_TO_MODEL != None :
        board_transformer_model = load_model(PATH_TO_MODEL).to(device)

    optimizer = AdamW(board_transformer_model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss(reduction='mean')

    # Train
    print("-- Starting training --")
    
    total_samples = 0
    iter_to_log = total_samples
    for epoch in range(start_epoch, n_epochs):
        board_transformer_model.train()
        training_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):
            images, board_coords = data
            images, board_coords = images.to(device), board_coords.to(device)

            optimizer.zero_grad()

            outputs = board_transformer_model(images)
            loss = criterion(outputs, board_coords)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if i % log_every_n_batches == 0:
                iter_to_log = total_samples

                # Log losses
                loss_to_log = training_loss/log_every_n_batches
                training_loss = 0.0
                writer.add_scalar("Loss", loss_to_log, total_samples)
                print(f"Training Loss: {loss_to_log}")

                # Log data
                image = images[0].detach().cpu().numpy()
                target_coords = board_coords[0].detach().cpu().numpy()
                predicted_coords = outputs[0].detach().cpu().numpy()

                image = np.transpose(image, (1, 2, 0))*255
                target_coords = (target_coords*img_size[0]).reshape((4, 2))
                predicted_coords = (predicted_coords*img_size[0]).reshape((4, 2))

                image = image.astype(np.uint8)
                image = image.copy()
                target_coords = target_coords
                predicted_coords = predicted_coords

                target_image = draw_annotations_on_image(image, target_coords, normalized=True)
                predicted_image = draw_annotations_on_image(image, predicted_coords, normalized=True)

                writer.add_image("target", transforms.ToTensor()(target_image), iter_to_log)
                writer.add_image("predicted", transforms.ToTensor()(predicted_image), iter_to_log)
                writer.flush()

            total_samples += train_batch_size

        # Save the model checkpoint for this epoch
        print(f'Epoch {epoch}/{n_epochs}')
        torch.save(board_transformer_model.state_dict(), os.path.join(log_path, f"boardsegmentnet_{epoch}.pt")) 

        # Validate 
        if epoch % val_every_n_epoch == 0:
            board_transformer_model.eval()  # Set the model to evaluation mode
            val_total_loss = 0.0

            with torch.no_grad():   # Disable gradient calculation
                
                for data in test_dataloader:
                    images, board_coords = data
                    images, board_coords = images.to(device), board_coords.to(device)

                    outputs = board_transformer_model(images)
                    loss = criterion(outputs, board_coords)
                    val_total_loss += loss.item()

                validation_loss = val_total_loss / len(test_dataloader)
                print(f"Validation Loss: {validation_loss}")
                writer.add_scalar("Validation Loss", validation_loss, iter_to_log)


# __name__
if __name__=="__main__":
    main()


