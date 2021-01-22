import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    #def __init__(self, z_dim, hidden_dim):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, z_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        #have to here define the fully connected layers - 784 to 400
        #400 to 2
        
        # setup the non-linearities
        
        self.fc1 = nn.Linear(x_dim, h_dim1) #x_dim=10000 to h_dim1=4096 
        self.fc2 = nn.Linear(h_dim1, h_dim2) #h_dim1=4096 to h_dim2=2048
        self.fc3 = nn.Linear(h_dim2, h_dim3) #h_dim2=2048 to h_dim3=1024
        self.fc4 = nn.Linear(h_dim3, h_dim4) #h_dim3=1024 to h_dim4=512
        self.fc5 = nn.Linear(h_dim4, h_dim5) #h_dim4=512 to h_dim5=256
        self.fc61 = nn.Linear(h_dim5, z_dim) #h_dim5=256 to z_dim=2
        self.fc62 = nn.Linear(h_dim5, z_dim) #h_dim5=256 to z_dim=2
        
        self.softplus = nn.Softplus()

    def forward(self, x):
        
        x = x.reshape(-1,10000)
        
        slope_param=0.0001
        
        # then compute the hidden units
        # We use fully connected layers
        hidden = self.softplus(self.fc1(x))
        
        
       # hidden = F.leaky_relu(self.fc1(x),slope_param)
        hidden = F.leaky_relu(self.fc2(hidden),slope_param)
        hidden = F.leaky_relu(self.fc3(hidden),slope_param)
        hidden = F.leaky_relu(self.fc4(hidden),slope_param)
        hidden = F.leaky_relu(self.fc5(hidden),slope_param)
        
        z_loc = self.fc61(hidden)
        z_scale = torch.exp(self.fc62(hidden)) # mu, log_var

        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, z_dim):
        super(Decoder, self).__init__()
        
        self.fc7 = nn.Linear(z_dim, h_dim5) #z_dim=2 to h_dim5=256
        self.fc8 = nn.Linear(h_dim5, h_dim4) #h_dim5=256 to h_dim4=512
        self.fc9 = nn.Linear(h_dim4, h_dim3) #h_dim4=512 to h_dim3=1024
        self.fc10 = nn.Linear(h_dim3, h_dim2) #h_dim3=1024 to h_dim2=2048
        self.fc11 = nn.Linear(h_dim2, h_dim1) #h_dim2=2048 to h_dim1=4096
        self.fc12 = nn.Linear(h_dim1, x_dim)  #h_dim1=4096 to x_dim=10000
        
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        slope_param=0.0001
        hidden = F.leaky_relu(self.fc7(z),slope_param)
        hidden = F.leaky_relu(self.fc8(hidden),slope_param)
        hidden = F.leaky_relu(self.fc9(hidden),slope_param)
        hidden = F.leaky_relu(self.fc10(hidden),slope_param)
        hidden = F.leaky_relu(self.fc11(hidden),slope_param)
        
        loc_img = self.sigmoid(self.fc12(hidden))
        
        return  loc_img# <-------- To check this