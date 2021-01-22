import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision.utils import save_image


from raga.utils.FRDEEP import FRDEEPF

from raga.utils.MiraBest_full import MiraBest_full

from PIL import Image

import pickle

import matplotlib.pyplot as plt

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
    new_label_train[new_label_train==3]=-99 #Confident Standard FRII (167)
    new_label_train[new_label_train==4]=-99 #Confident DD FRII (2)
    new_label_train[new_label_train==5]=0 #Confident Hybrids (378)
    new_label_train[new_label_train==6]=0 #Uncertain Standard FRI (3)
    new_label_train[new_label_train==7]=-99 #Uncertain WAT FRI (171)
    new_label_train[new_label_train==8]=-99 #Uncertain Standard FRII (17)
    new_label_train[new_label_train==9]=-99 #Uncertain Hybrid (13)


    new_data_train = array_train[new_label_train != -99]

    new_label_train = new_label_train[new_label_train != -99]

    augmented_data=np.zeros((len(new_data_train)*36,1,100,100))
    
    
    
    
    count=0
    
    for j in range(0,550):
        image_object=Image.fromarray(array_train[j,0,:,:])
        for i in range(0,36):
            rotated=image_object.rotate(i*10)
            imgarr = np.array(rotated)
            temp_img_array=imgarr[25:125,25:125]
            augmented_data[count,0,:,:]=temp_img_array
            count+=1
    
    print('MIRABEST Dataset 1')
    
    augmented_data=(augmented_data-np.min(augmented_data))/(np.max(augmented_data)-np.min(augmented_data))
    
    augmented_data_label = np.zeros((len(new_label_train)*36,1))

    X=augmented_data
    
    plt.imshow(X[0][0])
    plt.savefig('test.png')
    
    X_random_mix=np.take(X,np.random.permutation(X.shape[0]),axis=0,out=X);
    
    # Have to add the test data to the code. The test data has to be mixed with the training data. 
    
    tensor_x = torch.stack([torch.Tensor(i) for i in X_random_mix])
    
    first_augmented_dataset = torch.utils.data.TensorDataset(tensor_x) # create your datset
    
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
    new_label_test[new_label_test==3]=-99 #Confident Standard FRII (167)
    new_label_test[new_label_test==4]=-99 #Confident DD FRII (2)
    new_label_test[new_label_test==5]=0 #Confident Hybrids (378)
    new_label_test[new_label_test==6]=0 #Uncertain Standard FRI (3)
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


def dataloader_first_2():
    
    with open('/raid/scratch/davidb/1_DEVELOPMENT/FIRST_DATASETS/MIRABEST/mirabest.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # 100 Confident FRI Standard >> 1
    # 102 Confident FRI Wide-Angle Tail >> 1
    # 104 Confident FRI Head-Tail >> 1
    # 200 Confident FRII Standard >> 0
    # 201 Confident FRII Double-Double >> 0
    # 300 Confident Hybrid >> -99
    # 110 Uncertain FRI Standard >> -99
    # 112 Uncertain FRI Wide-Angle Tail >> -99
    # 210 Uncertain FRII Standard >> -99
    # 310 Uncertain Hybrid >> -99
    
    labels=np.zeros(data[0].shape[0])
    
    labels[:] = -99
    
    data.append(labels)
    
    data[0][np.isnan(data[0])]=0.
    data[0] = (data[0]-np.min(data[0]))/(np.max(data[0])-np.min(data[0]))
    
    data[2][data[1][:,1]==100]=1
    data[2][data[1][:,1]==102]=1
    data[2][data[1][:,1]==104]=1
    data[2][data[1][:,1]==200]=0
    data[2][data[1][:,1]==201]=0
    
    data_x_first=data[0][data[2] != -99]
    
    data_y_first=data[2][data[2] != -99]
    
    temp_data_x = data_x_first
    temp_data_y = data_y_first
    
    train_fr1 = temp_data_x[temp_data_y == 1][0:327]
    test_fr1 = temp_data_x[temp_data_y == 1][327:409]
    train_fr2 = temp_data_x[temp_data_y == 0][0:357]
    test_fr2 = temp_data_x[temp_data_y == 0][357:446]
    
    train_data = np.vstack((train_fr1, train_fr2))
    train_labels = np.zeros((train_data.shape[0]))
    train_labels[0:train_fr1.shape[0]] = 1

    test_data = np.vstack((test_fr1,test_fr2))
    test_labels = np.zeros((test_data.shape[0]))
    test_labels[0:test_fr1.shape[0]]=1
    
    new_data_train = train_data.copy()
    new_label_train = train_labels.copy()
    
    new_data_test = test_data.copy()
    new_label_test = test_labels.copy()
    
    # --------------------------------------Preprocessing of Train Data-------------------------------------

    
    augmented_data=np.zeros((len(new_data_train)*36,1,100,100))
    augmented_data_label = np.zeros((len(new_label_train)*36,1))
    
    count=0

    for j in range(0,len(new_data_train)):
        image_object=Image.fromarray(new_data_train[j,:,:])
        for i in range(0,36):
            rotated=image_object.rotate(i*10)
            imgarr = np.array(rotated)
            temp_img_array=imgarr[25:125,25:125]
            augmented_data[count,0,:,:]=temp_img_array
            augmented_data_label[count,:]=new_label_train[j]
            count+=1
        
      #augmented_data=(augmented_data-np.min(augmented_data))/(np.max(augmented_data)-np.min(augmented_data))
    
    X=augmented_data
    Y=augmented_data_label
    
    X_random_mix=np.take(X,np.random.RandomState(seed=42).permutation(X.shape[0]),axis=0,out=X)
    Y_random_mix=np.take(Y,np.random.RandomState(seed=42).permutation(Y.shape[0]),axis=0,out=Y)
    
    #X_random_mix = X
    #Y_random_mix = Y
    
    tensor_x = torch.stack([torch.Tensor(i) for i in X_random_mix])
    tensor_y = torch.stack([torch.Tensor(i) for i in Y_random_mix])
    
    first_augmented_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    
    first_dataloader = torch.utils.data.DataLoader(first_augmented_dataset,batch_size=100, shuffle=True) # create your dataloader
    
    # --------------------------------------Preprocessing of Test Data-------------------------------------
    
    test_data_reduced=np.zeros((len(new_data_test),1,100,100))
    test_data_label = np.zeros((len(new_data_test),1))
    
    for k in range (0,len(new_data_test)):
        test_data_reduced[k][0][:][:] = new_data_test[k][25:125,25:125]
        test_data_label[k,:]=new_label_test[k]
    
    #test_data_reduced=(test_data_reduced-np.min(test_data_reduced))/(np.max(test_data_reduced)-np.min(test_data_reduced))
    
    
    tensor_test = torch.stack([torch.Tensor(i) for i in test_data_reduced])
    tensor_test_label = torch.stack([torch.Tensor(i) for i in test_data_label])
    
    first_augmented_dataset_test = torch.utils.data.TensorDataset(tensor_test,tensor_test_label) # create your datset
    
    first_dataloader_test = torch.utils.data.DataLoader(first_augmented_dataset_test,batch_size=50, shuffle=True) # create your dataloader
    
    return first_dataloader,first_dataloader_test



def dataloader_first_noisy():
    
    with open('/raid/scratch/davidb/1_DEVELOPMENT/FIRST_DATASETS/MIRABEST/mirabest.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # 100 Confident FRI Standard >> 1
    # 102 Confident FRI Wide-Angle Tail >> 1
    # 104 Confident FRI Head-Tail >> 1
    # 200 Confident FRII Standard >> 0
    # 201 Confident FRII Double-Double >> 0
    # 300 Confident Hybrid >> -99
    # 110 Uncertain FRI Standard >> -99
    # 112 Uncertain FRI Wide-Angle Tail >> -99
    # 210 Uncertain FRII Standard >> -99
    # 310 Uncertain Hybrid >> -99
    
    labels=np.zeros(data[0].shape[0])
    
    labels[:] = -99
    
    data.append(labels)
    
    data[0][np.isnan(data[0])]=0.
    data[0] = (data[0]-np.min(data[0]))/(np.max(data[0])-np.min(data[0]))
    
    log_data=np.log(data[0])
    log_data[np.isinf(log_data)] = -5.903
    data[0]= log_data.copy()
    
    data[0] = (data[0]-np.min(data[0]))/(np.max(data[0])-np.min(data[0]))
    
    data[2][data[1][:,1]==100]=1
    data[2][data[1][:,1]==102]=1
    data[2][data[1][:,1]==104]=1
    data[2][data[1][:,1]==200]=0
    data[2][data[1][:,1]==201]=0
    
    data_x_first=data[0][data[2] != -99]
    
    data_y_first=data[2][data[2] != -99]
    
    temp_data_x = data_x_first
    temp_data_y = data_y_first
    
    train_fr1 = temp_data_x[temp_data_y == 1][0:327]
    test_fr1 = temp_data_x[temp_data_y == 1][327:409]
    train_fr2 = temp_data_x[temp_data_y == 0][0:357]
    test_fr2 = temp_data_x[temp_data_y == 0][357:446]
    
    train_data = np.vstack((train_fr1, train_fr2))
    train_labels = np.zeros((train_data.shape[0]))
    train_labels[0:train_fr1.shape[0]] = 1

    test_data = np.vstack((test_fr1,test_fr2))
    test_labels = np.zeros((test_data.shape[0]))
    test_labels[0:test_fr1.shape[0]]=1
    
    new_data_train = train_data.copy()
    new_label_train = train_labels.copy()
    
    new_data_test = test_data.copy()
    new_label_test = test_labels.copy()
    
    # --------------------------------------Preprocessing of Train Data-------------------------------------

    
    augmented_data=np.zeros((len(new_data_train)*36,1,100,100))
    augmented_data_label = np.zeros((len(new_label_train)*36,1))
    
    count=0

    for j in range(0,len(new_data_train)):
        image_object=Image.fromarray(new_data_train[j,:,:])
        for i in range(0,36):
            rotated=image_object.rotate(i*10)
            imgarr = np.array(rotated)
            temp_img_array=imgarr[25:125,25:125]
            augmented_data[count,0,:,:]=temp_img_array
            augmented_data_label[count,:]=new_label_train[j]
            count+=1
        
      #augmented_data=(augmented_data-np.min(augmented_data))/(np.max(augmented_data)-np.min(augmented_data))
    
    X=augmented_data
    Y=augmented_data_label
    
    X_random_mix=np.take(X,np.random.RandomState(seed=42).permutation(X.shape[0]),axis=0,out=X)
    Y_random_mix=np.take(Y,np.random.RandomState(seed=42).permutation(Y.shape[0]),axis=0,out=Y)
    
    #X_random_mix = X
    #Y_random_mix = Y
    
    tensor_x = torch.stack([torch.Tensor(i) for i in X_random_mix])
    tensor_y = torch.stack([torch.Tensor(i) for i in Y_random_mix])
    
    first_augmented_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    
    first_dataloader = torch.utils.data.DataLoader(first_augmented_dataset,batch_size=100, shuffle=True) # create your dataloader
    
    # --------------------------------------Preprocessing of Test Data-------------------------------------
    
    test_data_reduced=np.zeros((len(new_data_test),1,100,100))
    test_data_label = np.zeros((len(new_data_test),1))
    
    for k in range (0,len(new_data_test)):
        test_data_reduced[k][0][:][:] = new_data_test[k][25:125,25:125]
        test_data_label[k,:]=new_label_test[k]
    
    #test_data_reduced=(test_data_reduced-np.min(test_data_reduced))/(np.max(test_data_reduced)-np.min(test_data_reduced))
    
    
    tensor_test = torch.stack([torch.Tensor(i) for i in test_data_reduced])
    tensor_test_label = torch.stack([torch.Tensor(i) for i in test_data_label])
    
    first_augmented_dataset_test = torch.utils.data.TensorDataset(tensor_test,tensor_test_label) # create your datset
    
    first_dataloader_test = torch.utils.data.DataLoader(first_augmented_dataset_test,batch_size=50, shuffle=True) # create your dataloader
    
    return first_dataloader,first_dataloader_test

def dataloader_first_FRDEEP():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
    testset = FRDEEPF(root='./FIRST_data', train=False, download=True, transform=transform) 
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=2, batch_size=len(testset))

    trainset = FRDEEPF(root='./FIRST_data', train=True, download=True, transform=transform)  
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=2, batch_size=len(trainset))


    classes = ('FRI', 'FRII')
    
    array_train= next(iter(trainloader))[0].numpy() # Training Datasets is loaded in numpy array
    
    array_label= next(iter(trainloader))[1].numpy() # Training Datasets labels is loaded in seperate numpy array
    
    augmented_data=np.zeros((19800,1,100,100))
    augmented_labels = np.zeros((19800))
    
    #-------------------------------Labels Missing for FRDEEP Dataset-----------------------------------
    
    count=0
    
    for j in range(0,550):
        image_object=Image.fromarray(array_train[j,0,:,:])
        for i in range(0,36):
            rotated=image_object.rotate(i*10)
            imgarr = np.array(rotated)
            temp_img_array=imgarr[25:125,25:125]
            augmented_data[count,0,:,:]=temp_img_array
            augmented_labels[count]= array_label[j]
            count+=1

    augmented_data=(augmented_data-np.min(augmented_data))/(np.max(augmented_data)-np.min(augmented_data))

    X=augmented_data
    Y=augmented_labels

    Y=Y.reshape(19800,1)

    # Have to add the test data to the code. The test data has to be mixed with the training data. 
    
    tensor_x = torch.stack([torch.Tensor(i) for i in X])
    tensor_y = torch.stack([torch.Tensor(i) for i in Y])


    #--------------------Put the label set here as tensor y-----------------------------------
    first_augmented_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y.reshape(19800)) # create your datset
    
    first_dataloader = torch.utils.data.DataLoader(first_augmented_dataset,batch_size=100, shuffle=True) # create your dataloader
    
    
    # Cropping of the Testing Images to 100 by 100 pixels

    array_test= next(iter(testloader))[0].numpy()

    array_test_label = next(iter(testloader))[1].numpy()

    test_data_reduced=np.zeros((50,1,100,100))
    for k in range (0,50):
        test_data_reduced[k][0][:][:] = array_test[k][0][25:125,25:125]
    
    test_data_reduced=(test_data_reduced-np.min(test_data_reduced))/(np.max(test_data_reduced)-np.min(test_data_reduced))

    array_test_label = array_test_label.reshape(50,1)
    
    
    tensor_test = torch.stack([torch.Tensor(i) for i in test_data_reduced])

    tensor_test_label = torch.stack([torch.Tensor(i) for i in array_test_label])



    
    first_augmented_dataset_test = torch.utils.data.TensorDataset(tensor_test,tensor_test_label.reshape(50)) # create your datset
    
    first_dataloader_test = torch.utils.data.DataLoader(first_augmented_dataset_test,batch_size=50, shuffle=True) # create your dataloader

    
    return first_dataloader, first_dataloader_test