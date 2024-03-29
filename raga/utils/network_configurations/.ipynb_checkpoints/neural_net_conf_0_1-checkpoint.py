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
            
            h_dim1 = 4096
            h_dim2 = 2048
            h_dim3 = 1024
            h_dim4 = 512
            h_dim5 = 256
            
            z_dim = 2
            
            
            # setup the two linear transformations used
            self.fc7 = nn.Linear(z_dim+y_dim, h_dim5) #z_dim=2 to h_dim5=256
            self.fc8 = nn.Linear(h_dim5, h_dim4) #h_dim5=256 to h_dim4=512
            self.fc9 = nn.Linear(h_dim4, h_dim3) #h_dim4=512 to h_dim3=1024
            self.fc10 = nn.Linear(h_dim3, h_dim2) #h_dim3=1024 to h_dim2=2048
            self.fc11 = nn.Linear(h_dim2, h_dim1) #h_dim2=2048 to h_dim1=4096
            self.fc12 = nn.Linear(h_dim1, x_dim)  #h_dim1=4096 to x_dim=10000
        
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
            hidden = F.leaky_relu(self.fc7(z_y_1),slope_param)
            hidden = F.leaky_relu(self.fc8(hidden),slope_param)
            hidden = F.leaky_relu(self.fc9(hidden),slope_param)
            hidden = F.leaky_relu(self.fc10(hidden),slope_param)
            hidden = F.leaky_relu(self.fc11(hidden),slope_param)
        
            loc_img = self.sigmoid(self.fc12(hidden))
            
            return loc_img



class EncoderZ(nn.Module):
        #def __init__(self, z_dim, hidden_dim):
        def __init__(self):
            super(EncoderZ, self).__init__()
            
            x_dim = 10000
            y_dim =2
            
            h_dim1 = 4096
            h_dim2 = 2048
            h_dim3 = 1024
            h_dim4 = 512
            h_dim5 = 256
            
            z_dim = 2
            
            
            self.fc1 = nn.Linear(x_dim+y_dim, h_dim1) # x_dim=10000 + y_dim=2 to h_dim1=4096 
            self.fc2 = nn.Linear(h_dim1, h_dim2) #h_dim1=4096 to h_dim2=2048
            self.fc3 = nn.Linear(h_dim2, h_dim3) #h_dim2=2048 to h_dim3=1024
            self.fc4 = nn.Linear(h_dim3, h_dim4) #h_dim3=1024 to h_dim4=512
            self.fc5 = nn.Linear(h_dim4, h_dim5) #h_dim4=512 to h_dim5=256
            self.fc61 = nn.Linear(h_dim5, z_dim) #h_dim5=256 to z_dim=2
            self.fc62 = nn.Linear(h_dim5, z_dim) #h_dim5=256 to z_dim=2
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
        
        
       # hidden = F.leaky_relu(self.fc1(x),slope_param)
            hidden = F.leaky_relu(self.fc2(hidden),slope_param)
            hidden = F.leaky_relu(self.fc3(hidden),slope_param)
            hidden = F.leaky_relu(self.fc4(hidden),slope_param)
            hidden = F.leaky_relu(self.fc5(hidden),slope_param)
        
            z_loc = self.fc61(hidden)
            z_scale = torch.exp(self.fc62(hidden)) # mu, log_var
        
        
            return z_loc, z_scale

        

        
