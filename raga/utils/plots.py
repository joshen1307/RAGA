import pylab as pl
import numpy as np
import scipy.stats as stats
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import seaborn as sns

def positionimage(x, y, ax, ar, zoom=0.5):
    """Place image from file `fname` into axes `ax` at position `x,y`."""

    im = OffsetImage(ar, zoom=zoom)
    im.image.axes = ax

    ab = AnnotationBbox(im, (x,y), xycoords='data')
    ax.add_artist(ab)


def make_linemarker(x,y,dx,col,ax):

    xs = [x-0.5*dx,x+0.5*dx]
    for i in range(0,y.shape[0]):
        ys = [y[i],y[i]]
        ax.plot(xs,ys,marker=",",c=col,alpha=0.1,lw=5)

    return

def plot_error(accs1, accs2):

    train_err = 100.*(1 - np.array(accs1))
    test_err = 100.*(1 - np.array(accs2))

    pl.subplot(111)
    pl.plot(train_err,label="train")
    pl.plot(test_err,label="test")
    #pl.axis([0,len(test_err),0.8,2.2])
    pl.xlabel("Epochs")
    pl.ylabel("Test Error (%)")
    pl.legend()
    pl.savefig("./outputs/plot_err.png")
    pl.show()

    np.savez("./outputs/testerr.npz",test_err)

    return

def plot_weights(model):

    weights = np.array([])
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "w_mu" in name or "weight" in name:
                weights = np.concatenate((weights,param.data.numpy().ravel()))

    num_bins = 200
    n, bins, patches = pl.hist(weights, num_bins, facecolor='blue', alpha=0.5)
    pl.savefig("./outputs/plot_weights.png")
    pl.show()

    return

def plot_snr(model):

    snr = np.array([])
    layers = list(dict(model.named_children()).keys())
    for layer in layers:
        for name, param in model.named_parameters():
            if layer in name and param.requires_grad:
                if name==layer+".w_mu":
                    means = param.data.numpy().ravel()
                if name==layer+".w_rho":
                    sigmas= np.log(1 + np.exp(param.data.numpy().ravel()))

        snr = np.concatenate((snr,means/sigmas))

    num_bins = 200
    n, bins, patches = pl.hist(snr, num_bins, facecolor='blue', alpha=0.5)
    pl.savefig("./outputs/plot_snr.png")
    pl.show()

    return

def uncertainty_test(model, test_loader):
    model.train()
    T = 100
    rotation_list = range(0, 180, 10)
    dataiter = iter(test_loader)
    data, target = dataiter.next()
    image_list = []
    pred_list = []
    outp_list = []
    inpt_list = []
    for r in rotation_list:

        # make rotated image:
        rotation_matrix = torch.Tensor([[[np.cos(r/360.0*2*np.pi), -np.sin(r/360.0*2*np.pi), 0],
                                        [np.sin(r/360.0*2*np.pi), np.cos(r/360.0*2*np.pi), 0]]])
        grid = F.affine_grid(rotation_matrix, data.size(), align_corners=True)
        data_rotate = F.grid_sample(data, grid, align_corners=True)
        image_list.append(data_rotate)

        # run 100 stochastic forward passes:
        output_list, input_list = [], []
        for i in range(T):
            x = model(data_rotate)
            input_list.append(torch.unsqueeze(x, 0))
            output_list.append(torch.unsqueeze(F.softmax(x,dim=1), 0))

        # calculate the mean output for each target:
        output_mean = np.squeeze(torch.cat(output_list, 0).mean(0).data.numpy())

        # append per rotation output into list:
        pred_list.append(output_mean.argmax())
        outp_list.append(np.squeeze(torch.cat(output_list, 0).data.numpy()))
        inpt_list.append(np.squeeze(torch.cat(input_list, 0).data.numpy()))

        print ('rotation degree', str(r), 'Predict : {}'.format(output_mean.argmax()))

    # find unique predictions:
    pred_list = np.array(pred_list)
    preds = np.unique(pred_list)

    outp_list = np.array(outp_list)
    inpt_list = np.array(inpt_list)
    rotation_list = np.array(rotation_list)

    colours=["b","r","g","orange","b","r","g","orange","b","r","g","orange"]

    fig1, (a0, a1) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [8,1]})
    fig2, (a2, a3) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [8,1]})

    a0.set_title("Input")
    a2.set_title("Output")

    dx = 0.8*(rotation_list[1]-rotation_list[0])
    for pred in preds:
        col = colours[pred]
        a0.plot(rotation_list[0],inpt_list[0,0,pred],marker=",",c=col,label=str(pred))
        a2.plot(rotation_list[0],outp_list[0,0,pred],marker=",",c=col,label=str(pred))
        for i in range(rotation_list.shape[0]):
            make_linemarker(rotation_list[i],inpt_list[i,:,pred],dx,col,a0)
            make_linemarker(rotation_list[i],outp_list[i,:,pred],dx,col,a2)

    a0.legend()
    a2.legend()
    #a0.axis([0,180,0,1])
    a0.set_xlabel("Rotation [deg]")
    a2.set_xlabel("Rotation [deg]")
    a1.axis([0,180,0,1])
    a3.axis([0,180,0,1])
    a1.axis('off')
    a3.axis('off')
    for i in range(len(rotation_list)):
        inc = 0.5*(180./len(rotation_list))
        positionimage(rotation_list[i]+inc, 0., a1, image_list[i][0, 0, :, :].data.numpy())
        positionimage(rotation_list[i]+inc, 0., a3, image_list[i][0, 0, :, :].data.numpy())

    fig1.tight_layout()
    fig2.tight_layout()
    pl.show()

def fr_rotation_test(model, test_loader):
    model.train()
    T = 100
    rotation_list = range(0, 180, 10)
    dataiter = iter(test_loader)
    data, target = dataiter.next()
    print("True classification: ",target[0].item())

    image_list = []
    outp_list = []
    inpt_list = []
    for r in rotation_list:

        # make rotated image:
        rotation_matrix = torch.Tensor([[[np.cos(r/360.0*2*np.pi), -np.sin(r/360.0*2*np.pi), 0],
                                        [np.sin(r/360.0*2*np.pi), np.cos(r/360.0*2*np.pi), 0]]])
        grid = F.affine_grid(rotation_matrix, data.size(), align_corners=True)
        data_rotate = F.grid_sample(data, grid, align_corners=True)
        image_list.append(data_rotate)

        # run 100 stochastic forward passes:
        output_list, input_list = [], []
        for i in range(T):
            x = model(data_rotate)
            input_list.append(torch.unsqueeze(x, 0))
            output_list.append(torch.unsqueeze(F.softmax(x,dim=1), 0))

        # calculate the mean output for each target:
        output_mean = np.squeeze(torch.cat(output_list, 0).mean(0).data.numpy())

        # append per rotation output into list:
        outp_list.append(np.squeeze(torch.cat(output_list, 0).data.numpy()))
        inpt_list.append(np.squeeze(torch.cat(input_list, 0).data.numpy()))

        print ('rotation degree', str(r), 'Predict : {}'.format(output_mean.argmax()))

    preds = np.array([0,1])

    outp_list = np.array(outp_list)
    inpt_list = np.array(inpt_list)
    rotation_list = np.array(rotation_list)

    colours=["b","r","g","orange","b","r","g","orange","b","r","g","orange"]

    fig1, (a0, a1) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [8,1]})
    fig2, (a2, a3) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [8,1]})

    a0.set_title("Input")
    a2.set_title("Output")

    dx = 0.8*(rotation_list[1]-rotation_list[0])
    for pred in preds:
        col = colours[pred]
        a0.plot(rotation_list[0],inpt_list[0,0,pred],marker=",",c=col,label=str(pred))
        a2.plot(rotation_list[0],outp_list[0,0,pred],marker=",",c=col,label=str(pred))
        for i in range(rotation_list.shape[0]):
            make_linemarker(rotation_list[i],inpt_list[i,:,pred],dx,col,a0)
            make_linemarker(rotation_list[i],outp_list[i,:,pred],dx,col,a2)

    a0.legend()
    a2.legend()
    #a0.axis([0,180,0,1])
    a0.set_xlabel("Rotation [deg]")
    a2.set_xlabel("Rotation [deg]")
    a1.axis([0,180,0,1])
    a3.axis([0,180,0,1])
    a1.axis('off')
    a3.axis('off')
    for i in range(len(rotation_list)):
        inc = 0.5*(180./len(rotation_list))
        positionimage(rotation_list[i]+inc, 0., a1, image_list[i][0, 0, :, :].data.numpy(), zoom=0.32)
        positionimage(rotation_list[i]+inc, 0., a3, image_list[i][0, 0, :, :].data.numpy(), zoom=0.32)

    fig1.tight_layout()
    fig2.tight_layout()
    pl.show()
