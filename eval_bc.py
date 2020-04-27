import numpy as np
import torch
import torchvision
import os
import argparse

from config import Config
from dataset import KOTestDataset
from bc_dataset import BCTrainDataset

parser = argparse.ArgumentParser()
parser.add_argument('mixture_type', choices=['linear', 'double', 'prop', 'VH', 'Gauss'], help='method of mixing for BC learning')
parser.add_argument('save_path', help='path to model save')
args = parser.parse_args()

configs = Config()
configs.bc_mixing_method = args.mixture_type

test_set = KOTestDataset(configs)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=8,
    num_workers=8,
    pin_memory=False,
    shuffle=False,
    drop_last=True
)

model = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=configs.num_classes)

print("Loading Model from: {}".format(args.save_path))
model.load_state_dict(torch.load(args.save_path))

model.eval()

correct = np.zeros(configs.num_classes)
total = np.zeros(configs.num_classes)
with torch.no_grad():
    for batch in test_loader:
        data, label = batch

        data.to(configs.device)
        label.to(configs.device)

        preds = model(data)

        preds = torch.argmax(preds, dim=1)

        for idx in range(configs.num_classes):

            correct[idx] += torch.sum((preds == label) * (label == (torch.ones(label.shape[0]) * idx)))
            total[idx] += torch.sum(label == (torch.ones(label.shape[0]) * idx))

for idx in range(configs.num_classes):
    print("Validation Accuracy for Class {}: {}".format(idx, correct[idx] / total[idx]))

print("Mean Validation Accuracy: {}".format(np.sum(correct) / np.sum(total)))
