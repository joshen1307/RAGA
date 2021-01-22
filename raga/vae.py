#Parse in number of dimensions and laerning rate
#Arguments to parse 
#i. Latent dimensions d
#ii. Learning Rate lr
#iii. Number of epoch to training
#iv. CUDA-True or False
#v. log file directory


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

from raga.utils.models import HMT #Check if path to this file is correct

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
from raga.utils.MiraBest_full import MiraBest_full
from raga.utils.FRDEEP import FRDEEPF
from raga.utils.data_downloader import dataloader_first_FRDEEP
from raga.utils.data_downloader import dataloader_first
from raga.utils.classification_function import classfication_procedure

from raga.utils.network_configurations import neural_net_conf_0_2_dropout as network


def inception_score():
    model = HMT().cuda()
    model.load_state_dict(torch.load('raga/utils/model.mod'))
    
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
    def __init__(self, x_dim,h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, z_dim, use_cuda=True): # z_dim has been changed to 2
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







# model_unsup >>>>> model for FRDEEP with 2 latent space. Conf = 0:2
# model_unsup_10 >>>>>> model for FRDEEP with 10 latent space. Conf = 0:2 
# model_unsup_MIRABEST_2 >>>>> model for MIRABEST with 2 latent space Cond = 0:2\


#--------------------------------------------------Dataloader initialisation---------------------------------------------  








def lrsearch(latent_dimensions,training_epoch,log_dir): 

    d = latent_dimensions
    training_epochs = training_epoch
    
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
        NUM_EPOCHS = training_epoch #Remove the hardcoded [8000]
        TEST_FREQUENCY = 5

        #-----------------------------------------------------VAE initialisation--------------------------------------------------



        vae = VAE(x_dim=10000,h_dim1=4096,h_dim2=2048,h_dim3=1024,h_dim4=512,h_dim5=256,z_dim=d,use_cuda=USE_CUDA)




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
             epoch_loss = 0.
   
             for x in train_loader: # Note that _ is the labels only and x are the images.
             # if on GPU put mini-batch into CUDA memory
        
                 x = x[0].cuda()
                 # do ELBO gradient and accumulate loss
                 epoch_loss += svi.step(x)
        
             # return epoch loss
             normalizer_train = len(train_loader.dataset)
    
             total_epoch_loss_train = epoch_loss / normalizer_train
      
             train_elbo.append(-total_epoch_loss_train)
    
    
             # --------------------------Do testing for each epoch here--------------------------------
             # initialize loss accumulator
             test_loss = 0.
             # compute the loss over the entire test set
             for x_test in test_loader:
                 # if on GPU put mini-batch into CUDA memory

                 x_test = x_test[0].cuda()
                 # compute ELBO estimate and accumulate loss
                 test_loss += svi.evaluate_loss(x_test) #Data entry point <---------------------------------Data Entry Point
             normalizer_test = len(test_loader.dataset)
             total_epoch_loss_test = test_loss / normalizer_test
    
    
             print("[epoch %03d]  average training loss: %.4f testing loss: %.4f" % (epoch, total_epoch_loss_train,total_epoch_loss_test))
        
        count = count + 1

        df['learning_rate'][count]=LEARNING_RATE
        df['train_loss'][count]=total_epoch_loss_train
        print('+++++++++++++++++++++++++++++++++++++Incrementing Learning Rate++++++++++++++++++++++++++++++++++++')
        learning_rates.append(LEARNING_RATE)
        train_losses.append(total_epoch_loss_train)
    
        df.to_csv(log_dir+'/data_lr_experiment_exp0_d'+str(d)+'.csv')
    
        LEARNING_RATE = LEARNING_RATE + 0.00001



def vaetrain(path,LEARNING_RATE,d,NUM_EPOCHS,log_path):
    train_loader,test_loader = dataloader_first_FRDEEP()



    # Core Training section
    #--------------------------------------------------------Main VAE Section-----------------------------------------------------

    # setup the optimizer
    adam_args = {"lr": LEARNING_RATE}
    # optimizer = Adam(adam_args)   # The Adam optimizer is used as optimizer
    optimizer = Adagrad(adam_args)
    # setup the inference algorithm
    TEST_FREQUENCY = 5

#--------------------------------------------Defining the dataframe to save the logs--------------------------------------
    data = np.zeros((5,8000)).T

    columns = ['Epoch','Train_Loss', 'Test_Loss', 'Sigma_clipped','Inception_score']
    df = pd.DataFrame(data, columns=columns)
    vae = VAE(x_dim=10000,h_dim1=4096,h_dim2=2048,h_dim3=1024,h_dim4=512,h_dim5=256,z_dim=d,use_cuda=True)
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
    
    def single_image_sampler(z):
         single_sample_image = vae.decoder(z)
         image_array_single =single_sample_image.reshape(100,100).cpu().detach().numpy()
         temp_array=image_array_single
         return image_array_single

# model_unsup >>>>> model for FRDEEP with 2 latent space. Conf = 0:2
# model_unsup_10 >>>>>> model for FRDEEP with 10 latent space. Conf = 0:2 
# model_unsup_MIRABEST_2 >>>>> model for MIRABEST with 2 latent space Cond = 0:2\

#++++++++++++++++++++++++++++++Recode this a function that create the new directory should be create+++++++++++++++++++++++
#+++++++++++++++++++++++++++++++A function that log about the experiment description should be coded+++++++++++++++++++++++


    def save_checkpoint(epoch,path):
         print("saving model to ...")
         torch.save(vae.state_dict(), path+'/'+'model_unsup_main_'+str(epoch)+'.mod')
         print("saving optimizer states...")
         optimizer.save(path+'/model_unsup_main_'+str(epoch)+'_opt.opt')
         print("done saving model and optimizer checkpoints to disk.")
    

    # The different SVI methods have been commented
    #svi = SVI(vae.model, vae.guide, optimizer, loss=simple_elbo)
    #svi = SVI(vae.model, vae.guide, optimizer, loss=simple_elbo_kl_annealing)
    #svi.step(annealing_factor=0.2, latents_to_anneal=["my_latent"])

    def inception_scoring(d,limits):
        z_fr = torch.randn(100, d)

        for i in range (0,100):
            for j in range (0,d):
                z_fr[i,j] = np.random.uniform(limits[0,j],limits[1,j])
    
        sample1 = vae.decoder(z_fr.cuda()).cpu().detach().numpy().reshape(100,1,100,100)
    
        fullsize_image = np.zeros((100,1,150,150))

        for i in range (0,100):
             fullsize_image[i,0,25:125,25:125]=sample1[i,0,:,:]
    
        array_generated= torch.from_numpy(fullsize_image).float().to("cpu")

        valid_pred = inception_classifier(array_generated.cuda())
        m = nn.Softmax(dim=1)
        values=m(valid_pred).cpu().detach().numpy()
        score = calculate_inception_score(values)

        return score

    pyro.clear_param_store() #VVIP clears out parameters eash time looping sequence is engaged

    train_elbo = []
    test_elbo = []
    # training loop
    

    for epoch in range(NUM_EPOCHS):
        # total_epoch_loss_train = train(svi, epoch, train_loader)
        # ------------ To include Training Loop just for test -------------------
        epoch_loss = 0.
   
        for x in train_loader: # Note that _ is the labels only and x are the images.
        # if on GPU put mini-batch into CUDA memory
        
            x = x[0].cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)
        
        # return epoch loss
        normalizer_train = len(train_loader.dataset)
    
        total_epoch_loss_train = epoch_loss / normalizer_train
      
        train_elbo.append(-total_epoch_loss_train)
    
    
        # --------------------------Do testing for each epoch here--------------------------------
        # initialize loss accumulator
        test_loss = 0.
        # compute the loss over the entire test set
        for x_test in test_loader:
            # if on GPU put mini-batch into CUDA memory

            x_test = x_test[0].cuda()
            # compute ELBO estimate and accumulate loss
            test_loss += svi.evaluate_loss(x_test) #Data entry point <---------------------------------Data Entry Point
        normalizer_test = len(test_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test
        incept_score = 0 
    
    
        # This loop fixes the limits for the random number generator
        #On first run the 
        limits = np.zeros((2,d))
        for i in range (0,d):
             limits[0,i]= -4
             limits[1,i]= 4
    
    
        incept_score = inception_scoring(d,limits) #Calls the inception score and calculates it.
    
    
    
        print("[epoch %03d]  average training loss: %.4f testing loss: %.4f inception score: %.4f" % (epoch, total_epoch_loss_train,total_epoch_loss_test,incept_score))
        df['Epoch'][epoch]=epoch
        df['Train_Loss'][epoch]=total_epoch_loss_train
        df['Test_Loss'][epoch]=total_epoch_loss_test
        df['Inception_score'][epoch]=incept_score
    
        if epoch%10 == 0:
            df.to_csv(log_path+'/data_unsupervised_d'+str(d)+'.csv')
    
        if epoch%50 == 0:
            save_checkpoint(epoch,path)
            print('checkpoint saved to '+path)


    