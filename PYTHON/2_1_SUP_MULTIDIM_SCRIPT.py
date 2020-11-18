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



import argparse
import sys


from datetime import date

today = date.today()
print("Today's date:", today)



def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-lr", "--learning_rate", help="Lerning Rate of VAE")
    parser.add_argument("-moddir","--model_directory", help="Model Saving Directory")
    parser.add_argument("-d","--latent_dimensions", help="Define the number of latent dimensions")
    options = parser.parse_args(args)
    return options

options = getOptions(sys.argv[1:])


LEARNING_RATE=float(options.learning_rate)
model_directory=options.model_directory
d = int(options.latent_dimensions)

string_lr = str(LEARNING_RATE).replace(".", "-")
print(string_lr)

#---------------------------------Create directory to save models------------------------------------

path = model_directory+"/"+"SUP-MODELS-"+"d"+str(d)+"-"+string_lr+"-"+str(today)
print(path)
try:
    os.mkdir(path)
except OSError:
    if os.path.isdir(path):
         print ('Directory Already Exist Ovewriting Existing Directory Content')
    else:
         print ("Creation of the directory %s failed" % path)
else:
    
    print ("Successfully created the directory %s " % path)



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




# ------------------------------------------------Initialisation of all the parameters-----------------------------------
# Fully Connected Layer network architecture parameterization
#d = 4 #Define the number of dimensions in the Bottleneck of the Variational Auto Encoder

#Learning Rate Configuration
print(LEARNING_RATE)
#Use CUDA configuration
USE_CUDA = True

# Run only for a single iteration for testing 

# Define 
NUM_EPOCHS = 1 if smoke_test else 8000 #Remove the hardcoded [8000]
TEST_FREQUENCY = 5

#--------------------------------------------Defining the dataframe to save the logs--------------------------------------
data = np.zeros((7,8000)).T

columns = ['Epoch','Train_Loss', 'Test_Loss', 'Sigma_clipped','Inception_score','number_FR1','number_FR2']
df = pd.DataFrame(data, columns=columns)


#-----------------------------------------------------VAE initialisation--------------------------------------------------

vae = VAE(x_dim=10000,h_dim1=4096,h_dim2=2048,h_dim3=1024,h_dim4=512,h_dim5=256,y_dim=2,z_dim=d,use_cuda=USE_CUDA)
#The VAE has to be initialised right before the evaluate(), single_image_sampler() and save_checkpoint() as they make use of VAE instance

# Note that the mini batch logic is handled by the data loader, we should duplicate the same logic of the data loader with the FIRST Database. The core of the training loop is svi.step(x). This is the data entry point. It should be noted that we have to change the looping structure to that of the mini batch structure that is used for the FIRST database.

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



# To do evaluate part afterwards
def evaluate(svi, test_loader, use_cuda=True):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x,y in test_loader:
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

#++++++++++++++++++++++++++++++Recode this a function that create the new directory should be create+++++++++++++++++++++++
#+++++++++++++++++++++++++++++++A function that log about the experiment description should be coded+++++++++++++++++++++++


def save_checkpoint(epoch):
    print("saving model to ...")
    torch.save(vae.state_dict(), path+'/'+'model_unsup_main_'+str(epoch)+'.mod')
    print("saving optimizer states...")
    optimizer.save(path+'/model_unsup_main_'+str(epoch)+'_opt.opt')
    print("done saving model and optimizer checkpoints to disk.")


#--------------------------------------------------Dataloader initialisation---------------------------------------------    
pyro.clear_param_store() 
#VVIP clears out parameters eash time looping sequence is engaged

train_loader,test_loader = dataloader_first_FRDEEP()


# Core Training section
#--------------------------------------------------------Main VAE Section-----------------------------------------------------

# setup the optimizer
adagrad_args = {"lr": LEARNING_RATE}
# optimizer = Adam(adam_args)   # The Adam optimizer is used as optimizer
optimizer = Adagrad(adagrad_args)
# setup the inference algorithm
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

# The different SVI methods have been commented
#svi = SVI(vae.model, vae.guide, optimizer, loss=simple_elbo)
#svi = SVI(vae.model, vae.guide, optimizer, loss=simple_elbo_kl_annealing)
#svi.step(annealing_factor=0.2, latents_to_anneal=["my_latent"])


#INCEPTION SCORER TEMPORARILY REMOVED FOR SMOKE TEST


def inception_scoring(d,limits):
    
    #Create the latent sample z to be used to sample the radio sources from the decoder architecture
    z_fr1 = torch.randn(100, d)
    z_fr2 = torch.randn(100, d)
    
    
    for i in range (0,100):
        for j in range (0,d):
            z_fr1[i,j] = np.random.uniform(limits[0,j],limits[1,j])
            z_fr2[i,j] = np.random.uniform(limits[0,j],limits[1,j])
            
    #Defines the labels 
    
    labels_y1 = torch.tensor(np.zeros((100,2)))
    labels_y2 = torch.tensor(np.zeros((100,2)))
        
    labels_y1[:,1] = 0
    labels_y2[:,1] = 0

    labels_y1[:,0] = 0
    labels_y2[:,0] = 0

    labels_y1[:,1] = 1  #To check this in more detail so as not to create confucion
    labels_y2[:,0] = 1
    
    #Sample the two sets of sources
    
    sample_fr1 = vae.decoder([z_fr1.cuda(),labels_y1.cuda().float()])
    img1=sample_fr1.reshape(100,100,100).cpu().detach().numpy()
 

    sample_fr2 = vae.decoder([z_fr2.cuda(),labels_y2.cuda().float()])
    
    img2=sample_fr2.reshape(100,100,100).cpu().detach().numpy()

    
        
    array_fr1 = np.zeros(100)
    array_fr2 = np.zeros(100)

        
    fullsize_image = np.zeros((200,1,150,150))

    for i in range (0,100):
        fullsize_image[i,0,25:125,25:125]=img1[i,:,:]
        
    for i in range (100,200):
        fullsize_image[i,0,25:125,25:125]=img2[i-100,:,:]
        
    
    array_generated= torch.from_numpy(fullsize_image).float().to("cpu")

    valid_pred = inception_classifier(array_generated.cuda())
    m = nn.Softmax(dim=1)
    values=m(valid_pred).cpu().detach().numpy()
    score = calculate_inception_score(values)
    
    array_generated_fr1 = torch.from_numpy(fullsize_image[0:100,:,:,:]).float().to("cpu")
    fr1_pred = inception_classifier(array_generated_fr1.cuda())
    values=m(fr1_pred).cpu().detach().numpy()
    print(values.shape)
    num_fr1=len(values[values[:,0]>0.5])
    
    array_generated_fr2 = torch.from_numpy(fullsize_image[100:200,:,:,:]).float().to("cpu")
    fr2_pred = inception_classifier(array_generated_fr2.cuda())
    values=m(fr2_pred).cpu().detach().numpy()
    num_fr2=len(values[values[:,1]>0.5])
    
    fullsize_image_reshaped_fr1 = img1.reshape(100*100*100,1)
    fullsize_image_reshaped_fr2 = img2.reshape(100*100*100,1)
    
    fr1_clipped = len(fullsize_image_reshaped_fr1[fullsize_image_reshaped_fr1<0.00001])
    fr2_clipped = len(fullsize_image_reshaped_fr2[fullsize_image_reshaped_fr2<0.00001])
   
    
    return score,num_fr1,num_fr2,(fr1_clipped+fr2_clipped)/(2*100000)



train_elbo = []
test_elbo = []
# training loop


for epoch in range(NUM_EPOCHS):
   # total_epoch_loss_train = train(svi, epoch, train_loader)
    # ------------ To include Training Loop just for test -------------------
    
    total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)

        
    
    
    train_elbo.append(-total_epoch_loss_train)
    
    
# -----------------------------------------Test Loop------------------------------------------
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
    incept_score = 0 
     
       
    # This loop fixes the limits for the random number generator
    #On first run the 
    limits = np.zeros((2,d))
    for i in range (0,d):
        limits[0,i]= -4
        limits[1,i]= 4
    
    
    incept_score,num_fr1,num_fr2,sigma_clipped = inception_scoring(d,limits) #Calls the inception score and calculates it.
       
    
#    print("[epoch %03d]  average training loss: %.4f testing loss: %.4f inception score: %.4f" % (epoch, total_epoch_loss_train,total_epoch_loss_test,inception_score))

    print("[epoch %03d]  average training loss: %.4f testing loss: %.4f inception score: %.4f Number of FRI: %.4f Number of FRII: %.4f Sigma Clipped: %.4f" % (epoch, total_epoch_loss_train,total_epoch_loss_test,incept_score,num_fr1,num_fr2,sigma_clipped))
    
    df['Epoch'][epoch]=epoch
    df['Train_Loss'][epoch]=total_epoch_loss_train
    df['Test_Loss'][epoch]=total_epoch_loss_test
    df['Inception_score'][epoch]=incept_score
    df['Sigma_clipped'][epoch]=sigma_clipped
    df['number_FR1'][epoch]=num_fr1
    df['number_FR2'][epoch]=num_fr2
    
    if epoch%10 == 0:
        df.to_csv('data_supervised_d'+str(d)+'.csv')
    
    if epoch%50 == 0:
        save_checkpoint(epoch)
        print('checkpoint saved to '+path)

