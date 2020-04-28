import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision.utils import save_image


from utils.FRDEEP import FRDEEPF

from utils.MiraBest_full import MiraBest_full

from PIL import Image



def dataloader_first():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])]) #Defines the Transformation for the dataframe (only makes use of normalisation)

    trainset = MiraBest_full(root='./FIRST_data', train=True, download=True, transform=transform) #Downloads the trainset from Mirabest Full

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=2, batch_size=len(trainset))

    classes = ('FRI', 'FRII') #First class if FR1 and second class is FR2

    array_train= next(iter(trainloader))[0].numpy() # Training Datasets is loaded in numpy array
    array_label= next(iter(trainloader))[1].numpy()


    new_label_train=np.copy(array_label)


    #---------------Labels Recoding Sequence------------------
    new_label_train[new_label_train==0]=1 #Confident Standard FRI (297)
    new_label_train[new_label_train==1]=1 #Confident WAT FRI (43)
    new_label_train[new_label_train==2]=1 #Confident Head-Tail FRI (8)
    new_label_train[new_label_train==3]=0 #Confident Standard FRII (167)
    new_label_train[new_label_train==4]=0 #Confident DD FRII (2)
    new_label_train[new_label_train==5]=-99 #Confident Hybrids (378)
    new_label_train[new_label_train==6]=-99 #Uncertain Standard FRI (3)
    new_label_train[new_label_train==7]=-99 #Uncertain WAT FRI (171)
    new_label_train[new_label_train==8]=-99 #Uncertain Standard FRII (17)
    new_label_train[new_label_train==9]=-99 #Uncertain Hybrid (13)


    new_data_train = array_train[new_label_train != -99]

    new_label_train = new_label_train[new_label_train != -99]

    augmented_data=np.zeros((len(new_data_train)*36,1,100,100))
    
    augmented_data_label = np.zeros((len(new_label_train)*36,1))


    count=0
    
    for j in range(0,len(new_data_train)):
        image_object=Image.fromarray(new_data_train[j,0,:,:])
        for i in range(0,36):
            rotated=image_object.rotate(i*10)
            imgarr = np.array(rotated)
            temp_img_array=imgarr[25:125,25:125]
            augmented_data[count,0,:,:]=temp_img_array
            augmented_data_label[count,:]=new_label_train[j]
            count+=1
        
    augmented_data=(augmented_data-np.min(augmented_data))/(np.max(augmented_data)-np.min(augmented_data))
    
    X=augmented_data
    Y=augmented_data_label
    
    #X_random_mix=np.take(X,np.random.RandomState(seed=42).permutation(X.shape[0]),axis=0,out=X)
    #Y_random_mix=np.take(Y,np.random.RandomState(seed=42).permutation(Y.shape[0]),axis=0,out=Y)
    
    X_random_mix = X
    Y_random_mix = Y
    
    tensor_x = torch.stack([torch.Tensor(i) for i in X_random_mix])
    tensor_y = torch.stack([torch.Tensor(i) for i in Y_random_mix])
    
    first_augmented_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    
    first_dataloader = torch.utils.data.DataLoader(first_augmented_dataset,batch_size=100, shuffle=True) # create your dataloader



    #-------------------------------------------------TESTING PART OF CODE----------------------------------------------------

    # Cropping of the Testing Images to 100 by 100 pixels    
    testset = MiraBest_full(root='./FIRST_data', train=False, download=True, transform=transform) 
    
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=2, batch_size=len(testset))

    array_test= next(iter(testloader))[0].numpy()
    
    test_labels = next(iter(testloader))[1].numpy()

    new_label_test=np.copy(test_labels)


    #---------------Labels Recoding Sequence------------------
    new_label_test[new_label_test==0]=1 #Confident Standard FRI (297)
    new_label_test[new_label_test==1]=1 #Confident WAT FRI (43)
    new_label_test[new_label_test==2]=1 #Confident Head-Tail FRI (8)
    new_label_test[new_label_test==3]=0 #Confident Standard FRII (167)
    new_label_test[new_label_test==4]=0 #Confident DD FRII (2)
    new_label_test[new_label_test==5]=-99 #Confident Hybrids (378)
    new_label_test[new_label_test==6]=-99 #Uncertain Standard FRI (3)
    new_label_test[new_label_test==7]=-99 #Uncertain WAT FRI (171)
    new_label_test[new_label_test==8]=-99 #Uncertain Standard FRII (17)
    new_label_test[new_label_test==9]=-99 #Uncertain Hybrid (13)


    #------------Filter Out based on previous Criteria-------------
    new_data_test = array_test[new_label_test != -99]
    new_label_test = new_label_test[new_label_test != -99]

    test_data_reduced=np.zeros((len(new_data_test),1,100,100))
    test_data_label = np.zeros((len(new_data_test),1))
    
    for k in range (0,len(new_data_test)):
        test_data_reduced[k][0][:][:] = new_data_test[k][0][25:125,25:125]
        test_data_label[k,:]=new_label_test[k]
    
    test_data_reduced=(test_data_reduced-np.min(test_data_reduced))/(np.max(test_data_reduced)-np.min(test_data_reduced))
    
    
    tensor_test = torch.stack([torch.Tensor(i) for i in test_data_reduced])
    tensor_test_label = torch.stack([torch.Tensor(i) for i in test_data_label])
    
    first_augmented_dataset_test = torch.utils.data.TensorDataset(tensor_test,tensor_test_label) # create your datset
    
    first_dataloader_test = torch.utils.data.DataLoader(first_augmented_dataset_test,batch_size=50, shuffle=True) # create your dataloader
    
    return first_dataloader,first_dataloader_test