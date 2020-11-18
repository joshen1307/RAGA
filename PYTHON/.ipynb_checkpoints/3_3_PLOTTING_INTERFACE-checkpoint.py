import matplotlib.pyplot as plt


import os
import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam,Adagrad
import pyro.poutine as poutine
from torchvision.utils import save_image

import matplotlib.pylab as plt

from PIL import Image # Module for image rotation (making use of PIL)



pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

#from utils.MiraBest import MiraBest
from utils.MiraBest_full import MiraBest_full
from utils.FRDEEP import FRDEEPF
from utils.data_downloader import dataloader_first_noisy
from utils.data_downloader import dataloader_first_FRDEEP
from utils.classifier_fr_2 import classification_procedure

import network_configurations.neural_net_conf_0_2 as network #change this here to change configuration


#-----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------VAE BLOCK SECTION--------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # @David We have changed the latent space to 2-dimensional
    # and we use 400 hidden units
    def __init__(self, x_dim=10000,h_dim1=4096, h_dim2=2048, h_dim3=1024, h_dim4=512, h_dim5=256, z_dim=2, use_cuda=True): # z_dim has been changed to 2
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        #For the first images add addition hidden layers
        self.encoder = network.Encoder(x_dim, h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, z_dim) #To check the layering structure
        self.decoder = network.Decoder(x_dim, h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, z_dim) #To check the layering structure

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # Do not touch this part for the time being this will be modified when doing the Semi-Supervised VAE     
    # define the model p(x|z)p(z)
    #-------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------Model Chunck--------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 10000))
    #------------------------------------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------Guide Chunk-----------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------
    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
    #------------------------------------------------------------------------------------------------------------------


    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
    
#-----------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------





vae = VAE()

# Note that the mini batch logic is handled by the data loader, we should duplicate the same logic of the data loader with the FIRST Database. The core of the training loop is svi.step(x). This is the data entry point. It should be noted that we have to change the looping structure to that of the mini batch structure that is used for the FIRST database.

# To do evaluate part afterwards

def evaluate(svi, test_loader, use_cuda=True):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x) #Data entry point <---------------------------------Data Entry Point
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

def z_space_sampler(epoch):
    z = torch.randn(400, 2).cuda()
    count_x=-4.0
    count_y_2=-4.0
    count_y=count_y_2
    x=0
    y=0
    for i in range (0,20):
        for j in range (0,20):
            z[x,0]=count_x
            z[x,1]=count_y
            x=x+1
            count_y=count_y+0.4
        y=y+1
        count_x=count_x+0.4
        count_y=count_y_2
    sample = vae.decoder(z)
    save_image(sample.view(400,1,100,100),'/raid/scratch/davidb/1_DEVELOPMENT/VAE_FIRST/VAE-MSc/PYTHON/RESULTS/PLOTS_UNSUP_1/sample_image_z'+str(epoch)+'.png',nrow=20)

    
def single_image_sampler(z0=0.0,z1=0.0):
    z = torch.rand(1,2).cuda()
    z[0,0]=z0
    z[0,1]=z1
    single_sample_image = vae.decoder(z)
    image_array_single =single_sample_image.reshape(100,100).cpu().detach().numpy()
    temp_array=image_array_single
    plt.figure(300,figsize = (10,10))
    plt.imshow(image_array_single)
    plt.colorbar()
    plt.show()
    return image_array_single





def load_checkpoint():
    print("loading model from ...")
    vae.load_state_dict(torch.load('model_unsup'))
    print("loading optimizer states from ...")
    optimizer.load('model_unsup_opt')
    print("done loading model and optimizer states.")
    
    
    



def on_pick(event):
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
    
    
    # Include Plotting of Image Here
    image = single_image_sampler(xmouse,ymouse)
    
    

#---------------------------------------MAIN RUNNING SEQUENCE----------------------------------------

# Run options
LEARNING_RATE = 0.3e-3
USE_CUDA = True



# Core Training section
#--------------------------------------------------------Main VAE Section-----------------------------------------------------
train_loader,test_loader = dataloader_first_FRDEEP() #Loads data from FIRST data is strored in train_loader, This part should include 
                                  #a testing data part.

# clear param store
pyro.clear_param_store()

# setup the VAE
vae = VAE(use_cuda=USE_CUDA)

# setup the optimizer
adam_args = {"lr": LEARNING_RATE}
# optimizer = Adam(adam_args)   # The Adam optimizer is used as optimizer
optimizer = Adagrad(adam_args)
# setup the inference algorithm
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
#svi = SVI(vae.model, vae.guide, optimizer, loss=simple_elbo)
#svi = SVI(vae.model, vae.guide, optimizer, loss=simple_elbo_kl_annealing)

#svi.step(annealing_factor=0.2, latents_to_anneal=["my_latent"])




load_checkpoint()
    
    

    
    
# Plot Reduced Dimension of Real Data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
trainset = FRDEEPF(root='./FIRST_data', train=True, download=True, transform=transform)  
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=2, batch_size=len(trainset))

for x in trainloader:
    x = x[0].cuda()
    
array_train= next(iter(trainloader))[0].numpy()

array_train=(array_train-np.min(array_train))/(np.max(array_train)-np.min(array_train))

array_train_cuda = torch.from_numpy(array_train).float().to("cuda:0")
array_ttrain_labels= next(iter(trainloader))[1].numpy()

reduced_dimension_array = np.zeros((550,2))

for ii in range (0,550):
    reduced_dimension_array[ii,:]=vae.encoder(array_train_cuda[ii][0][25:125,25:125])[0].cpu().detach().numpy()
    


    
fig, ax = plt.subplots()
plt.figure(200)

tolerance = 100 # points
ax.plot(reduced_dimension_array[:,0],reduced_dimension_array[:,1], 'ro', picker=100000000)

fig.canvas.callbacks.connect('pick_event', on_pick)

plt.show()