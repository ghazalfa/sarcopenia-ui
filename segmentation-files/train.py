
import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model import TResUnet
from metrics import DiceLoss, DiceBCELoss

def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"train", "frontal", name) + ".png" for name in data]
    masks = [os.path.join(path,"train", "frontal-mask", name) + ".png" for name in data]
    return images, masks

def load_names_val(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"test", "frontal", name) + ".png" for name in data]
    masks = [os.path.join(path,"test", "frontal-mask", name) + ".png" for name in data]
    return images, masks

def load_names_test(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"test", "raw", name) + ".png" for name in data]
    masks = [os.path.join(path,"test", "label", name) + ".png" for name in data]
    return images, masks

def load_data(path):
    train_names_path = f"{path}/frontal_train.txt"
    valid_names_path = f"{path}/test.txt"  

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names_val(path, valid_names_path) 

    return (train_x, train_y), (valid_x, valid_y)

def load_data_test(path):
    test_names_path = f"{path}/test.txt"
    test_x, test_y = load_names_test(path, test_names_path)

    return (test_x, test_y)

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        """ Image """
        # Load image and mask using OpenCV
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        
        # If using PIL, read image and mask for any transformations
        image_pil = Image.open(self.images_path[index]).convert("RGB")
        mask_pil = Image.open(self.masks_path[index]).convert("L")

        # Apply transformations if any
        if self.transform is not None:
            augmentations = self.transform(image=image_pil, mask=mask_pil)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Convert PIL image and mask to NumPy arrays before resizing
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        # Resize image and mask
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)

        # Normalize the image and mask
        image = image / 255.0
        mask = mask / 255.0

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # Change the order of dimensions to match PyTorch (C, H, W)
        image = image.permute(2, 0, 1)  # from (H, W, C) to (C, H, W)
        mask = mask.unsqueeze(0)  # Add channel dimension for the mask (1, H, W)

        return image, mask

    def __len__(self):
        return self.n_samples


def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y, y_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def evaluate(model, loader, loss_fn, device):
    model.eval()    

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

if __name__ == "__main__":
    
    """ Seeding """
    seeding(42)

    """ Training Directories """
    scratch_base = "/scratch/st-ilker-1/capstone_2024/sep"
    train_dir = os.path.join(scratch_base, "train")
    create_dir(train_dir)
    checkpoint_path = os.path.join(train_dir, "checkpoint_localization_m20.pth")
    train_log_path = os.path.join(train_dir, "train_log_localization_m20.txt")
    # train_log_path = "files/train_log_mar19_segment.txt"

    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("/scratch/st-ilker-1/capstone_2024/sep/train/train_log_localization_m20.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 300
    patch_size = 4
    lr = 1e-4
    early_stopping_patience = 20
    # checkpoint_path = "files/checkpoint_0319_train_seg.pth"
    path = "/arc/project/st-ilker-1/2024_capstone/data/kavanati-resized-correct"


    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nPatch Size: {patch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)
    # train_x = train_x[:100]
    # train_y = train_y[:100]
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=None)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    create_dir("data")
    for i, (x, y) in enumerate(train_dataset):
        x = np.transpose(x, (1, 2, 0)) * 255
        y = np.transpose(y, (1, 2, 0)) * 255
        y = np.concatenate([y, y, y], axis=-1)
        cv2.imwrite(f"data/{i}.png", np.concatenate([x, y], axis=1))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu')
        )
    model = TResUnet(patch_size=patch_size)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_and_save(train_log_path, f"Trainable parameters: {param_count}")
    

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
