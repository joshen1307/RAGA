#Parse in number of dimensions and laerning rate
#Arguments to parse 
#i. Latent dimensions d
#ii. Learning Rate lr
#iii. Number of epoch to training
#iv. Loss/Inception score saving rate
#v. CUDA-True or False
#vi. model saving frequency
#vii. log file directory


import os
import pandas as pd
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

from utils.models import HMT

from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp

import sys
import argparse

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

#from utils.MiraBest import MiraBest
from utils.MiraBest_full import MiraBest_full
from utils.FRDEEP import FRDEEPF
from utils.data_downloader import dataloader_first_FRDEEP
from utils.data_downloader import dataloader_first
from utils.classification_function import classfication_procedure

import network_configurations.neural_net_conf_0_1_dropout as network1

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-d","--latent_dimensions", help="Define the number of latent dimensions")
    options = parser.parse_args(args)
    return options


options = getOptions(sys.argv[1:])


d = int(options.latent_dimensions)

def inception_score():
    model = HMT().cuda()
    model.load_state_dict(torch.load('model.mod'))
    
    return model.eval()

inception_classifier = inception_score()


def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score


#---------------------------------------------------VAE CLASS DECLARATION------------------------------------------------------------
class VAE(nn.Module):
# by default our latent space is 50-dimensional
    # @David We have changed the latent space to 2-dimensional
    # and we use 400 hidden units
    def __init__(self, x_dim,h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, y_dim, z_dim, use_cuda=True): # z_dim has been changed to 2
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        #For the first images add addition hidden layers
        
        self.encoder_z = network1.EncoderZ(x_dim, h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, y_dim, z_dim)
        
        self.decoder = network1.Decoder(x_dim, h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, y_dim, z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            
            
        self.use_cuda = use_cuda
        self.z_dim = z_dim
                
    # define the model p(x|z)p(z)
    def model(self, xs, ys):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)
        batch_size = xs.size(0)
        

        output_size = 2
        
        
        
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
          
            # sample the handwriting style from the constant prior distribution
            prior_loc = xs.new_zeros([batch_size, self.z_dim])
            prior_scale = xs.new_ones([batch_size, self.z_dim])
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = xs.new_ones([batch_size, output_size]) / (1.0 * output_size)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)
            
            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            loc = self.decoder.forward([zs, ys])
            pyro.sample("x", dist.Bernoulli(loc).to_event(1), obs=xs)
            
    def guide(self, xs, ys):
        with pyro.plate("data"):
           # if the class label (the digit) is not supervised, sample
           # (and score) the digit with the variational distribution
           # q(y|x) = categorical(alpha(x))
           
            #-------------------REMOVED THIS PART FOR THE CLASSIFIER ASSUME ALL DATA ARE LABELLED---------

           # sample (and score) the latent handwriting-style with the variational
           # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, xs, ys):
        # encode image x
        z_loc, z_scale = self.encoder_z.forward([xs,ys])
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder.forward([zs,ys])
        
        return loc_img



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

def train(svi, train_loader, use_cuda=True):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, y in train_loader:
       # if on GPU put mini-batch into CUDA memory
       if use_cuda:
            x = x.cuda()
       # do ELBO gradient and accumulate loss
       labels_y = torch.tensor(np.zeros((y.shape[0],2)))
       y_2=torch.Tensor.cpu(y.reshape(1,y.size()[0])[0]).numpy().astype(int)  
       labels_y=np.eye(2)[y_2]
       labels_y = torch.from_numpy(labels_y)   
         
       epoch_loss += svi.step(x.reshape(-1,10000),labels_y.cuda().float())

        # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


# model_unsup >>>>> model for FRDEEP with 2 latent space. Conf = 0:2
# model_unsup_10 >>>>>> model for FRDEEP with 10 latent space. Conf = 0:2 
# model_unsup_MIRABEST_2 >>>>> model for MIRABEST with 2 latent space Cond = 0:2\


#--------------------------------------------------Dataloader initialisation---------------------------------------------    
    
train_loader,test_loader = dataloader_first_FRDEEP()

learning_rates = []
train_losses = []
LEARNING_RATE = 0.0005


#--------------------------------------------Defining the dataframe to save the logs--------------------------------------
data = np.zeros((2,100)).T

columns = ['learning_rate','train_loss']
df = pd.DataFrame(data, columns=columns)
count = 0
for k in range(0,100):
    
    pyro.clear_param_store() #VVIP clears out parameters eash time looping sequence is engaged
    
    # ------------------------------------------------Initialisation of all the parameters-----------------------------------
    # Fully Connected Layer network architecture parameterization
    
    #Learning Rate Configuration
    print(LEARNING_RATE)
    #Use CUDA configuration
    USE_CUDA = True

    # Run only for a single iteration for testing 

    # Define 
    NUM_EPOCHS = 10 #Remove the hardcoded [8000]
    TEST_FREQUENCY = 5

    #-----------------------------------------------------VAE initialisation--------------------------------------------------



    vae = VAE(x_dim=10000,h_dim1=4096,h_dim2=2048,h_dim3=1024,h_dim4=512,h_dim5=256,y_dim=2,z_dim=d,use_cuda=USE_CUDA)




    # Core Training section
    #--------------------------------------------------------Main VAE Section-----------------------------------------------------

    # setup the optimizer
    adam_args = {"lr": LEARNING_RATE}
    # optimizer = Adam(adam_args)   # The Adam optimizer is used as optimizer
    optimizer = Adagrad(adam_args)
    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
    


    train_elbo = []
    test_elbo = []
    # training loop


    for epoch in range(NUM_EPOCHS):
        # total_epoch_loss_train = train(svi, epoch, train_loader)
        # ------------ To include Training Loop just for test -------------------
        total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
        
      
        train_elbo.append(-total_epoch_loss_train)
    
    
        # --------------------------Do testing for each epoch here--------------------------------
        test_loss = 0.
        # compute the loss over the entire test set
        for x_test,y_test in test_loader:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
            # compute ELBO estimate and accumulate loss
            labels_y_test = torch.tensor(np.zeros((y_test.shape[0],2)))
            y_test_2=torch.Tensor.cpu(y_test.reshape(1,y_test.size()[0])[0]).numpy().astype(int)  
            labels_y_test=np.eye(2)[y_test_2]
            labels_y_test = torch.from_numpy(labels_y_test)
        
            test_loss += svi.evaluate_loss(x_test.reshape(-1,10000),labels_y_test.cuda().float()) 
        
        
        normalizer_test = len(test_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test
    
        print("[epoch %03d]  average training loss: %.4f testing loss: %.4f" % (epoch, total_epoch_loss_train,total_epoch_loss_test))
    df['learning_rate'][count]=LEARNING_RATE
    df['train_loss'][count]=total_epoch_loss_train
    count = count + 1

    print('+++++++++++++++++++++++++++++++++++++Incrementing Learning Rate++++++++++++++++++++++++++++++++++++')
    learning_rates.append(LEARNING_RATE)
    train_losses.append(total_epoch_loss_train)
    
    df.to_csv('data_lr_experiment_sup_d'+str(d)+'.csv')
    
    LEARNING_RATE = LEARNING_RATE + 0.00001
    