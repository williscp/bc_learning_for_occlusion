import numpy as np
import torch
import torchvision

from config import Config
from dataset import KOTrainDataset, KOTestDataset
from model.baseline import LinearModel

configs = Config()

train_set = KOTrainDataset(configs)
test_set = KOTestDataset(configs)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=8,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=8,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

#model = LinearModel(configs).to(configs.device)

model = torchvision.models.resnet18(pretrained=False, num_classes=configs.num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
losses = []

for epoch in range(configs.epochs):
    for batch in train_loader:
        data, label = batch

        data.to(configs.device)
        label.to(configs.device)

        preds = model(data)

        #print(preds.shape)

        loss = torch.nn.functional.cross_entropy(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        print(losses[-1])

model.eval()

correct = 0.0
total = 0.0
with torch.no_grad():
    for batch in test_loader:
        data, label = batch

        data.to(configs.device)
        label.to(configs.device)

        preds = model(data)

        preds = torch.argmax(preds, dim=1)

        correct += torch.sum(preds == label)
        total += label.shape[0]

print("Validation Accuracy: {}".format(correct / total) )
