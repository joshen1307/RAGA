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


from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
import math
from sklearn import datasets, linear_model
from astropy.io import fits
from PIL import Image


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

from raga.utils.network_configurations import neural_net_conf_0_1_dropout as network1

from datetime import date

from astropy.io import fits

today = date.today()


def inception_score(model_inception_name):
    model = HMT().cuda()
    model.load_state_dict(torch.load(model_inception_name))
    
    return model.eval()

#inception_classifier = inception_score(model_inception_name) #<---------------------------------TO MOVE THIS LINE


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
class CVAE(nn.Module):
    # by default our latent space is 50-dimensional
    # @David We have changed the latent space to 2-dimensional
    # and we use 400 hidden units
    def __init__(self, x_dim,h_dim1, h_dim2, h_dim3, h_dim4, h_dim5, y_dim, z_dim, use_cuda=True): # z_dim has been changed to 2
        super(CVAE, self).__init__()
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

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------Learning Rate Search Code Section---------------------------------------------  
#--------------------------------------------------------------------------------------------------------------------------------


def lrsearch(latent_dimensions,training_epoch,log_dir):
    d = latent_dimensions
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



        cvae = CVAE(x_dim=10000,h_dim1=4096,h_dim2=2048,h_dim3=1024,h_dim4=512,h_dim5=256,y_dim=2,z_dim=d,use_cuda=USE_CUDA)




        # Core Training section
        #--------------------------------------------------------Main VAE Section-----------------------------------------------------

        # setup the optimizer
        adam_args = {"lr": LEARNING_RATE}
        # optimizer = Adam(adam_args)   # The Adam optimizer is used as optimizer
        optimizer = Adagrad(adam_args)
        # setup the inference algorithm
        svi = SVI(cvae.model, cvae.guide, optimizer, loss=Trace_ELBO())
    


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

#---------------------------------------------------------------------------------------------------------
#-------------------------------------------Train Function------------------------------------------------
#---------------------------------------------------------------------------------------------------------

def cvaetrain(path,LEARNING_RATE,d,NUM_EPOCHS,log_path,model_inception_name):


    string_lr = str(LEARNING_RATE).replace(".", "-")
    print(LEARNING_RATE)
    #Use CUDA configuration
    USE_CUDA = True

    # Run only for a single iteration for testing 

    # Define 
    TEST_FREQUENCY = 5

    #--------------------------------------------Defining the dataframe to save the logs--------------------------------------
    data = np.zeros((7,8000)).T

    columns = ['Epoch','Train_Loss', 'Test_Loss', 'Sigma_clipped','Inception_score','number_FR1','number_FR2']
    df = pd.DataFrame(data, columns=columns)
    #---------------------------------Create directory to save models------------------------------------

    path = path+"/"+"SUP-MODELS-"+"d"+str(d)+"-"+string_lr+"-"+str(today)
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


    train_loader,test_loader = dataloader_first_FRDEEP()



    # Core Training section
    #--------------------------------------------------------Main VAE Section-----------------------------------------------------

    # setup the optimizer
    adagrad_args = {"lr": LEARNING_RATE}
    # optimizer = Adam(adam_args)   # The Adam optimizer is used as optimizer
    optimizer = Adagrad(adagrad_args)
    # setup the inference algorithm

#--------------------------------------------Defining the dataframe to save the logs--------------------------------------
    
    cvae = CVAE(x_dim=10000,h_dim1=4096,h_dim2=2048,h_dim3=1024,h_dim4=512,h_dim5=256,y_dim=2,z_dim=d,use_cuda=USE_CUDA)

    svi = SVI(cvae.model, cvae.guide, optimizer, loss=Trace_ELBO())
    
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
         torch.save(cvae.state_dict(), path+'/'+'model_sup_main_'+str(epoch)+'.mod')
         print("saving optimizer states...")
         optimizer.save(path+'/model_sup_main_'+str(epoch)+'_opt.opt')
         print("done saving model and optimizer checkpoints to disk.")
    

    # The different SVI methods have been commented
    #svi = SVI(vae.model, vae.guide, optimizer, loss=simple_elbo)
    #svi = SVI(vae.model, vae.guide, optimizer, loss=simple_elbo_kl_annealing)
    #svi.step(annealing_factor=0.2, latents_to_anneal=["my_latent"])

    def inception_scoring(d,limits,model_inception_name):
    
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
    
        sample_fr1 = cvae.decoder([z_fr1.cuda(),labels_y1.cuda().float()])
        img1=sample_fr1.reshape(100,100,100).cpu().detach().numpy()
 

        sample_fr2 = cvae.decoder([z_fr2.cuda(),labels_y2.cuda().float()])
    
        img2=sample_fr2.reshape(100,100,100).cpu().detach().numpy()

    
        
        array_fr1 = np.zeros(100)
        array_fr2 = np.zeros(100)

        
        fullsize_image = np.zeros((200,1,150,150))

        for i in range (0,100):
             fullsize_image[i,0,25:125,25:125]=img1[i,:,:]
        
        for i in range (100,200):
             fullsize_image[i,0,25:125,25:125]=img2[i-100,:,:]
        
    
        array_generated= torch.from_numpy(fullsize_image).float().to("cpu")
        inception_classifier = inception_score(model_inception_name) #<-----------------Check this here
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

    pyro.clear_param_store() #VVIP clears out parameters eash time looping sequence is engaged

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
    
    
        incept_score,num_fr1,num_fr2,sigma_clipped = inception_scoring(d,limits,model_inception_name) #Calls the inception score and calculates it.
       
    
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
            save_checkpoint(epoch,path)
            print('checkpoint saved to '+path)

def inference_function(latent_dimension,weights_file,optimizer_file):
    USE_CUDA = True
    LEARNING_RATE =0.00001
    # setup the VAE
    cvae = CVAE(x_dim=10000,h_dim1=4096,h_dim2=2048,h_dim3=1024,h_dim4=512,h_dim5=256,y_dim=2,z_dim=latent_dimension,use_cuda=USE_CUDA)

    # setup the optimizer
    adam_args = {"lr": LEARNING_RATE}
    # optimizer = Adam(adam_args)   # The Adam optimizer is used as optimizer
    optimizer = Adagrad(adam_args)
    # setup the inference algorithm
    svi = SVI(cvae.model, cvae.guide, optimizer, loss=Trace_ELBO())

    print("loading model from ...")
    cvae.load_state_dict(torch.load(weights_file))
    print("loading optimizer states from ...")
    optimizer.load(optimizer_file)
    print("done loading model and optimizer states.")

    return cvae

def inference_function_main(latent_dimension,weights_file,optimizer_file,number_of_sources,fits_directory,source_class):
    CVAE_MOD=inference_function(latent_dimension,weights_file,optimizer_file)
    
    z_fr = torch.randn(number_of_sources,latent_dimension)
    for i in range (0,number_of_sources):
        for j in range (0,latent_dimension):
            z_fr[i,j] = np.random.uniform(-4,4)
    
    labels = torch.tensor(np.zeros((number_of_sources,2)))
     
        
    labels[:,1] = 0

    labels[:,0] = 0
       

    labels[:,source_class] = 1  #To check this in more detail so as not to create confucion

    
    #Sample the two sets of sources
    
    sample1 = CVAE_MOD.decoder([z_fr.cuda(),labels.cuda().float()]).cpu().detach().numpy().reshape(number_of_sources,1,100,100)



    rows = number_of_sources
    columns = 4
    image_set = 0
    j = 0

    image_file_rotated_all = np.zeros((number_of_sources,100,100))




    for n in range (0,rows):
        data_image=sample1[n,0,:,:]
        blobs_dog = blob_dog(data_image.astype('double'), max_sigma=1., threshold=0.005)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
        blobs = blobs_dog
        x_list=[]
        y_list=[]
    
        for blob in blobs:
            y, x, r = blob
            x_list.append(x)
            y_list.append(y)
        
        number_blobs = len(x_list)
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(np.array(x_list).reshape(-1, 1), np.array(y_list).reshape(-1, 1))

        # Make predictions using the testing set
        y_pred = regr.predict(np.array(x_list).reshape(-1, 1))
 
        image_object=Image.fromarray(data_image)
    
        rotated=image_object.rotate(math.degrees(math.atan(regr.coef_[0][0])))

    
        image_array_unrot= data_image
        image_array = np.array(rotated)
        image_file_rotated_all[n,:,:]=image_array


        


        hdu = fits.PrimaryHDU()

    
        header = hdu.header

        header.append("DATE") 
        header.append("OBJECT")
        header.append("OBJTX") 
        header.append("OBJTY") 
        header.append("XCORN")  
        header.append("YCORN")
        header.append("TELESCOP")
        header.append("INSTRUME")
        header.append("OBSERVER")
        header.append("DATE-OBS")
        header.append("DATE-MAP")
        header.append("BUNIT")
        header.append("EPOCH")
        header.append("EQUINOX")
        header.append("DATAMAX")
        header.append("DATAMIN")
        header.append("CTYPE1")
        header.append("CRVAL1")
        header.append("CDELT1")
        header.append("CRPIX1")
        header.append("CROTA1")
        header.append("CTYPE2")
        header.append("CRVAL2")
        header.append("CDELT2")
        header.append("CRPIX2")
        header.append("CROTA2")
        header.append("CTYPE3")
        header.append("CRVAL3")
        header.append("CDELT3")
        header.append("CRPIX3")
        header.append("CROTA3")
        header.append("CTYPE4")
        header.append("CRVAL4")
        header.append("CDELT4")
        header.append("CRPIX4")
        header.append("CROTA4")
        header.append("BMAJ")
        header.append("BMIN")
        header.append("BPA")
        header.append("SIZE")
        header.append("CLASS")

        header["DATE"]='2020-07-30'
        header["OBJECT"]='None'
        header["OBJTX"]=50
        header["OBJTY"]=50
        header["XCORN"]=1101
        header["YCORN"]=359
        header["TELESCOP"]='VAESourceSimulator'
        header["INSTRUME"]='VAESourceSimulator'
        header["OBSERVER"]='Simulated'
        header["DATE-OBS"]='00000000'
        header["DATE-MAP"]='00000000'
        header["BUNIT"]='JY/BEAM'
        header["EPOCH"]=2000.00
        header["EQUINOX"]=2000.00
        header["DATAMAX"]=1.0
        header["DATAMIN"]=0.0
        header["CTYPE1"]='RA---SIN'
        header["CRVAL1"]=162.750000000
        header["CDELT1"]=-0.000500000
        header["CRPIX1"]=-326.737
        header["CROTA1"]=0.00000
        header["CTYPE2"]='DEC--SIN'
        header["CRVAL2"]=30.7600000000
        header["CDELT2"]=0.000500000
        header["CRPIX2"]=216.790
        header["CROTA2"]=0.00000
        header["CTYPE3"]='FREQ'
        header["CRVAL3"]=1364900000.00
        header["CDELT3"]=2.18750E+07
        header["CRPIX3"]=1.00000
        header["CROTA3"]=0.00000
        header["CTYPE4"]='STOKES'
        header["CRVAL4"]=1.00000000000
        header["CDELT4"]=1.00000
        header["CRPIX4"]=1.00000
        header["CROTA4"]=0.00000
        header["BMAJ"]=1.5000E-03
        header["BMIN"]=1.5000E-03
        header["BPA"]=0.00
        header["SIZE"]=52.8
        header["CLASS"]=source_class
    
        if fits_directory != False:
            hdu = fits.PrimaryHDU(data=image_file_rotated_all[n],header=header)
            hdu.writeto(fits_directory+'/'+str(n)+'.fits')

        j = j + 1



    return image_file_rotated_all
