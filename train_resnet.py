import numpy as np
import torch
import torchvision
import os

from config import Config
from dataset import KOTrainDataset, KOTestDataset
from model.baseline import LinearModel

configs = Config()

train_set = KOTrainDataset(configs)
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

#model = LinearModel(configs).to(configs.device)

print("Downloading Model")
model = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=configs.num_classes)
#model = torchvision.models.mobilenet_v2(pretrained=False, progress=True, num_classes=configs.num_classes)

print("Starting Training")
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
if configs.schedule:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=configs.schedule, gamma=configs.decay)
train_losses = []
val_losses = []

correct = np.zeros(configs.num_classes)
total = np.zeros(configs.num_classes)

best_acc = 0 # best accuracy so far
best_epoch = 0

for epoch in range(configs.epochs):
    for batch in train_loader:
        data, label = batch
        
        data = data - configs.data_mean

        data.to(configs.device)
        label.to(configs.device)

        preds = model(data)

        #print(preds.shape)

        loss = torch.nn.functional.cross_entropy(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    print(np.mean(train_losses[-5]))


    if epoch % 1 == 0:
        model.eval()

        correct = np.zeros(configs.num_classes)
        total = np.zeros(configs.num_classes)
        with torch.no_grad():
            for batch in test_loader:
                data, label = batch
                
                data = data - configs.data_mean

                data.to(configs.device)
                label.to(configs.device)

                preds = model(data)
                log_preds = model(data)
                loss = torch.nn.functional.cross_entropy(log_preds, label)
                
                val_losses.append(loss.item())
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
            torch.save(model.state_dict(), os.path.join(configs.model_save_path, 'model_resnet_{}.pth'.format(epoch)))
        
        np.save(os.path.join(configs.output_dir, 'resnet_train_loss.npy'), train_losses)
        np.save(os.path.join(configs.output_dir, 'resnet_val_loss.npy'), val_losses)
        np.save(os.path.join(configs.output_dir, 'resnet_val_acc.npy'), val_acc)
        model.train()
        
    scheduler.step()


# evaluate on best model:

print("Best Epoch: {}".format(best_epoch))

model.load_state_dict(torch.load(os.path.join(configs.model_save_path, 'model_resnet_{}.pth'.format(best_epoch))))

model.eval()

correct = np.zeros(configs.num_classes)
total = np.zeros(configs.num_classes)
with torch.no_grad():
    for batch in test_loader:
        data, label = batch
        
        data = data - configs.data_mean
            
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
