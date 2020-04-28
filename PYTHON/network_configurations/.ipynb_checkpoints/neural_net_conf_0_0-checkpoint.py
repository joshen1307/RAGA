import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#------------------------------Network Configuration for Configuration 0_0---------------------------

class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            #Initialised the Network Size and Configuration
            
            x_dim = 10000
            y_dim =2
            
            h_dim1 =500 
            
            z_dim = 2
            
            
            # setup the two linear transformations used
            self.fc3 = nn.Linear(z_dim+y_dim, h_dim1) #z_dim=2 to h_dim5=500
            self.fc4 = nn.Linear(h_dim1, x_dim)  #h_dim1=4096 to x_dim=10000
        
            self.softplus = nn.Softplus()
            self.sigmoid = nn.Sigmoid()

        def forward(self,z_y_2):
            # define the forward computation on the latent z
            # first compute the hidden units
        
            [z,y]=z_y_2
        
            z = z.reshape(-1, 2) #@David Change this to reshape if something fucks up
            y = y.reshape(-1, 2)
            z_y_1 = torch.cat((z,y), dim=1)
            z_y_1 = z_y_1.view(z_y_1.size(0), -1)
        
            slope_param=0.0001
            hidden = F.leaky_relu(self.fc3(z_y_1),slope_param)
            loc_img = self.sigmoid(self.fc4(hidden))
            return loc_img



class EncoderZ(nn.Module):
        #def __init__(self, z_dim, hidden_dim):
        def __init__(self):
            super(EncoderZ, self).__init__()
            
            x_dim = 10000
            y_dim =2
            
            h_dim1 =500 
            
            z_dim = 2
            
            
            
            self.fc1 = nn.Linear(x_dim+y_dim, h_dim1) # x_dim=10000 + y_dim=2 to h_dim1=500
            self.fc21 = nn.Linear(h_dim1, z_dim) #h_dim5=500 to z_dim=2
            self.fc22 = nn.Linear(h_dim1, z_dim) #h_dim5=500 to z_dim=2
            self.softplus = nn.Softplus()

        def forward(self, x_y_2):
            [x,y]=x_y_2
        
            x = x.reshape(-1, 10000) 
            y = y.reshape(-1, 2) 
        
            x_y_1 = torch.cat((x,y), dim=1) 
            x_y_1 = x_y_1.view(x_y_1.size(0), -1)
        
            slope_param=0.0001
        
            # then compute the hidden units
            # We use fully connected layers
            hidden = self.softplus(self.fc1(x_y_1))
        
            z_loc = self.fc21(hidden)
            z_scale = torch.exp(self.fc22(hidden)) # mu, log_var
        
        
            return z_loc, z_scale

        
