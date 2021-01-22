import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import pandas as pd
import collections


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler


from FRDEEP import FRDEEPF

from models import HMT



def classfication_procedure(max_num):
    valid_size    = 110    # number of samples for validation
    batch_size    = 16     # number of samples per mini-batch
    num_classes   = 2      # The number of output classes. FRI/FRII
    lr0           = torch.tensor(1e-3)  # The speed of convergence # initially 1e-2
    momentum      = torch.tensor(9e-1)  # momentum for optimizer
    num_batches   = 55     # multiplies up the total samples to ~30k like in paper
    class_weights = torch.FloatTensor([0.6,0.4]).cuda() # for training
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
    
    model= HMT().cuda()
    learning_rate = lr0
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-6)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.0001)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    data = np.zeros((6,30)).T

    columns = ['Epoch','Train_Loss', 'Validation_Loss', 'Train_Accuracy', 'Test_Accuracy' ,'Validation_Accuracy']
    df = pd.DataFrame(data, columns=columns)


    # -----------------------------------------------------------------------------
    print('stattng training')
 
    epochs = max_num
    epoch_trainaccs, epoch_validaccs = [], []
    for epoch in range(epochs):
    
        model.train()
        train_accs=[]; acc = 0
        for iter in range(num_batches):
            for batch, (x_train, y_train) in enumerate(train_loader):
                model.zero_grad()
                pred = model(x_train.cuda())
                loss = criterion(pred.cuda(),y_train.cuda())
                loss.backward()
                optimizer.step()
                acc = (pred.argmax(dim=-1) == y_train.cuda()).to(torch.float32).mean()
                train_accs.append(acc.mean().item())

        print('Epoch: {}, Loss: {}, Train Accuracy: {}'.format(epoch, loss, np.mean(train_accs)))
    
    
        df['Epoch'][epoch]=epoch
        df['Train_Loss'][epoch]=loss
        df['Train_Accuracy'][epoch]=np.mean(train_accs)
    


        with torch.no_grad():
            model.eval()
            valid_losses, valid_accs = [], []; acc = 0
            for iter in range(num_batches):
                for i, (x_val, y_val) in enumerate(valid_loader):
                    valid_pred = model(x_val.cuda())
                    loss = criterion(valid_pred,y_val.cuda())
                    acc = (valid_pred.argmax(dim=-1) == y_val.cuda()).to(torch.float32).mean()
                    valid_losses.append(loss.item())
                    valid_accs.append(acc.mean().item())

        print('Epoch: {}, Loss: {}, Validation Accuracy: {}'.format(epoch, np.mean(valid_losses), np.mean(valid_accs)))
    
        epoch_trainaccs.append(np.mean(train_accs))
        epoch_validaccs.append(np.mean(valid_accs))
    
    

        df['Validation_Loss'][epoch]=np.mean(valid_losses)
        df['Validation_Accuracy'][epoch]=np.mean(valid_accs)
    
    
        print("Final validation error: ",100.*(1 - epoch_validaccs[-1]))



        # -----------------------------------------------------------------------------

        classes = ('FRI', 'FRII')
 
        test_data = FRDEEPF('first', train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
        df['Test_Accuracy'][epoch] = (100 * correct / total)

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images.cuda())
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels.cuda()).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        
        print('---------------------------------------------------------------------')
    
    return model.eval()
