import numpy as np
import torch
import torchvision
import os

from config import Config
from dataset import KOTestDataset
from bc_dataset import BCTrainDataset

configs = Config()

train_set = BCTrainDataset(configs)
test_set = KOTestDataset(configs)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=configs.batch_size,
    num_workers=8,
    pin_memory=False,
    shuffle=True,
    drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=8,
    num_workers=8,
    pin_memory=False,
    shuffle=False,
    drop_last=True
)

print("Downloading Model")

model = torchvision.models.mobilenet_v2(pretrained=False, progress=True, num_classes=configs.num_classes)

print("Starting Training")
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
losses = []

correct = np.zeros(configs.num_classes)
total = np.zeros(configs.num_classes)

best_acc = 0
best_epoch = 0

for epoch in range(configs.epochs):
    for batch in train_loader:
        data, label = batch

        data.to(configs.device)
        label.to(configs.device)

        preds = model(data)
        
        #print(label.type())
        #print(preds.type())
        preds = torch.nn.functional.log_softmax(preds, dim=1)
        loss = torch.nn.functional.kl_div(preds, label, reduction='batchmean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(np.mean(losses[-5]))

    if epoch % 1 == 0:
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

        mean_acc = np.sum(correct) / np.sum(total)
        print("Mean Validation Accuracy: {}".format(mean_acc))

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(configs.model_save_path, 'bc_prop_model_{}.pth'.format(epoch)))

        model.train()

np.save(os.path.join(configs.output_dir, 'bc_prop_train_loss.npy'), losses)

# evaluate on best model:

print("Best Epoch: {}".format(best_epoch))

model.load_state_dict(torch.load(os.path.join(configs.model_save_path, 'bc_prop_model_{}.pth'.format(best_epoch))))

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
