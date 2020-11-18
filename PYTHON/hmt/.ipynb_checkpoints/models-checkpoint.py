import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers import Linear_BBB

import numpy as np

# -----------------------------------------------------------------------------

class Classifier_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.h1  = nn.Linear(in_dim, hidden_dim)
        self.h2  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = m.in_features
                nn.init.normal_(m.weight, 0, 1/np.sqrt(y))
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.log_softmax(self.out(x),dim=1)
        return x

# -----------------------------------------------------------------------------

class Classifier_MLPDropout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.h1  = nn.Linear(in_dim, hidden_dim)
        self.h2  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.dr1 = nn.Dropout(p=0.5)
        self.dr2 = nn.Dropout(p=0.2)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.dr2(x)
        x = F.relu(self.h1(x))
        x = self.dr1(x)
        x = F.relu(self.h2(x))
        x = self.dr1(x)
        x = F.log_softmax(self.out(x),dim=1)
        return x

# -----------------------------------------------------------------------------

class LeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5):
        super(LeNet, self).__init__()

        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        self.drop  = nn.Dropout(p=0.5)

        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                y = m.in_features
                nn.init.uniform_(m.weight, -np.sqrt(3./y), np.sqrt(3./y))
                nn.init.constant_(m.bias, 0)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        #return F.softmax(x, dim=1)
        return x

# -----------------------------------------------------------------------------

class Classifier_BBB(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.h1  = Linear_BBB(in_dim, hidden_dim)
        self.h2  = Linear_BBB(hidden_dim, hidden_dim)
        self.out = Linear_BBB(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        #x = x.view(-1, 28*28)
        x = torch.sigmoid(self.h1(x))
        x = torch.sigmoid(self.h2(x))
        x = F.log_softmax(self.out(x),dim=1)
        return x

    def log_prior(self):
        return self.h1.log_prior + self.h2.log_prior + self.out.log_prior

    def log_post(self):
        return self.h1.log_post + self.h2.log_post + self.out.log_post

    def log_like(self,outputs,target):
        return F.nll_loss(outputs, target, reduction='sum')

    def sample_elbo(self, input, target, samples, batch, num_batches, burnin=None):

        outputs = torch.zeros(samples, target.shape[0], self.out_dim)

        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)

        for i in range(samples):
            outputs[i] = self(input)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            log_likes[i] = self.log_like(outputs[i,:,:],target)

        # the mean of a sum is the sum of the means:
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()

        if burnin=="blundell":
            frac = 2**(num_batches - batch + 1)/2**(num_batches - 1)
        elif burnin==None:
            frac = 1./num_batches

        loss = frac*(log_post - log_prior) + log_like

        return loss, outputs


# -----------------------------------------------------------------------------

class LeNet_BBB(nn.Module):

    def __init__(self, in_chan, out_chan, kernel_size=5):
        super(LeNet_BBB, self).__init__()

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        self.fc1   = nn.Linear(16*3*3, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*3*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# -----------------------------------------------------------------------------

class HMT(nn.Module):

    """
    This network has been taken directly from "transfer learning for radio galaxy classification"
    https://arxiv.org/abs/1903.11921
    """

    def __init__(self):
        super(HMT,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(11,11),padding=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),padding=2,stride=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=24,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=24,out_channels=16,kernel_size=(3,3),padding=1,stride=1)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.mpool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.bnorm1 = nn.BatchNorm2d(6)
        self.bnorm2 = nn.BatchNorm2d(16)
        self.bnorm3 = nn.BatchNorm2d(24)
        self.bnorm4 = nn.BatchNorm2d(24)
        self.bnorm5 = nn.BatchNorm2d(16)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(400,256) #channel_size * width * height
        #self.fc1 = nn.Linear(16*7*7,256) #channel_size * width * height
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,2)
        self.dropout = nn.Dropout()

        # initialise weights and biases:
        self.init_xavier()

    def init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x  = F.relu(self.conv1(x))
        x = self.bnorm1(x)
        x = self.mpool1(x)
        x  = F.relu(self.conv2(x))
        x = self.bnorm2(x)
        x = self.mpool2(x)
        x  = F.relu(self.conv3(x))
        x = self.bnorm3(x)
        x  = F.relu(self.conv4(x))
        x = self.bnorm4(x)
        x  = F.relu(self.conv5(x))
        x = self.bnorm5(x)
        x = self.mpool3(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# -----------------------------------------------------------------------------

class HMT2(nn.Module):

    """
    This network has been taken directly from "transfer learning for radio galaxy classification"
    https://arxiv.org/abs/1903.11921
    """

    def __init__(self):
        super(HMT2,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(11,11),padding=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),padding=2,stride=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=24,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=24,out_channels=16,kernel_size=(3,3),padding=1,stride=1)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.mpool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.bnorm1 = nn.BatchNorm2d(6)
        self.bnorm2 = nn.BatchNorm2d(16)
        self.bnorm3 = nn.BatchNorm2d(24)
        self.bnorm4 = nn.BatchNorm2d(24)
        self.bnorm5 = nn.BatchNorm2d(16)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(400,256) #channel_size * width * height
        #self.fc1 = nn.Linear(16*7*7,256) #channel_size * width * height
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,2)
        self.dropout = nn.Dropout(p=0.5)

        self.init_xavier()

    def init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x  = F.relu(self.conv1(x))
        x = self.bnorm1(x)
        x = self.mpool1(x)
        x  = F.relu(self.conv2(x))
        x = self.bnorm2(x)
        x = self.mpool2(x)
        x  = F.relu(self.conv3(x))
        x = self.bnorm3(x)
        x  = F.relu(self.conv4(x))
        x = self.bnorm4(x)
        x  = F.relu(self.conv5(x))
        x = self.bnorm5(x)
        x = self.mpool3(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)

        return x
