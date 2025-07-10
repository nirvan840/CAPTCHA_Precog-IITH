# general
from tqdm import tqdm
# time 
from datetime import datetime
import pytz

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights

# module
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# data tf
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
data_tf = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# character dataset
# train_data_root = "task2_generation/data_chars/train"
# test_data_root = "task2_generation/data_chars/test"
# bonus dataset
train_data_root = "task0_dataset/data_bonus/train"
test_data_root = "task0_dataset/data_bonus/test"
# make dataset
tr_dataset = datasets.ImageFolder(root=train_data_root, transform=data_tf)
tst_dataset = datasets.ImageFolder(root=test_data_root, transform=data_tf)

# dataloader
train_loader = DataLoader(tr_dataset, batch_size=512, shuffle=True, num_workers=8)
test_loader  = DataLoader(tst_dataset, batch_size=512, shuffle=False, num_workers=8)
print(f"\nClass to Index mapping: {tr_dataset.class_to_idx}\n")