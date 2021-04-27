import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from tqdm import tqdm

from models import HMT
from plots import *
from FRDEEP import FRDEEPN, FRDEEPF


valid_size    = 110    # number of samples for validation
batch_size    = 16     # number of samples per mini-batch
num_classes   = 2      # The number of output classes. FRI/FRII
lr0           = torch.tensor(1e-2)  # The speed of convergence
momentum      = torch.tensor(9e-1)  # momentum for optimizer
num_batches   = 55     # multiplies up the total samples to ~30k like in paper
class_weights = torch.FloatTensor([0.6,0.4]) # for training
random_seed   = 42

# -----------------------------------------------------------------------------

transform = transforms.Compose([
#    transforms.CenterCrop(28),
    transforms.RandomRotation(0.,360.),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

train_data = FRDEEPF('first', train=True, download=True, transform=transform)

num_train = len(train_data)
indices = list(range(num_train))
split = valid_size

np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

# -----------------------------------------------------------------------------

model = HMT()
learning_rate = lr0
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# -----------------------------------------------------------------------------

summary(model, (1, 150, 150))

# -----------------------------------------------------------------------------

epochs = 2
epoch_trainaccs, epoch_validaccs = [], []
for epoch in range(epochs):

    model.train()
    train_accs=[]; acc = 0
    for iter in range(num_batches):
        for batch, (x_train, y_train) in enumerate(train_loader):
            model.zero_grad()
            pred = model(x_train)
            loss = criterion(pred,y_train)
            loss.backward()
            optimizer.step()
            acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
            train_accs.append(acc.mean().item())

    print('Epoch: {}, Loss: {}, Train Accuracy: {}'.format(epoch, loss, np.mean(train_accs)))


    with torch.no_grad():
        model.eval()
        valid_losses, valid_accs = [], []; acc = 0
        for iter in range(num_batches):
            for i, (x_val, y_val) in enumerate(valid_loader):
                valid_pred = model(x_val)
                loss = criterion(valid_pred,y_val)
                acc = (valid_pred.argmax(dim=-1) == y_val).to(torch.float32).mean()
                valid_losses.append(loss.item())
                valid_accs.append(acc.mean().item())

    print('Epoch: {}, Loss: {}, Validation Accuracy: {}'.format(epoch, np.mean(valid_losses), np.mean(valid_accs)))
    epoch_trainaccs.append(np.mean(train_accs))
    epoch_validaccs.append(np.mean(valid_accs))

print("Final validation error: ",100.*(1 - epoch_validaccs[-1]))

plot_error(epoch_trainaccs, epoch_validaccs)

# -----------------------------------------------------------------------------

classes = ('FRI', 'FRII')

test_data = FRDEEPF('first', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
