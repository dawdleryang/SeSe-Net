#!/usr/bin/env python
# coding: utf-8

# # AVJ_new

# ## Load Necessary Libraries 

# In[ ]:

from __future__ import division
from PIL import Image
import glob
import copy
from tqdm import *
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os

import torchvision.transforms
import torchvision
from sklearn.preprocessing import StandardScaler
# Choose Pytorch library for CNN
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import  Dataset, DataLoader

from pandas import DataFrame
from pandas.io.parsers import read_csv
import scipy.ndimage as ndi

import skimage.io as skio
import skimage.transform as skt
from torchsummary import summary

from torch.nn import functional as F
import torchvision.transforms.functional as TF
import random
import torchvision.transforms.functional as TF
import random


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_HEIGHT = 128
INPUT_WIDTH = 128
OUTPUTS = 4
LR = 0.001
EPOCHS = 20001
BATCH_SIZE = 320
pre = False


# ## Load Data

# ### Get Selected Data Path

# In[ ]:


def load_data(data_dir, data_csv, load_pts=True):
    
    df = read_csv(data_csv)  # load pandas dataframe
    img_ids = df['ID']
    hR = 96    
    xCnt = 128
    yCnt = 128
    imgs = []
    
    for img_name in img_ids:
        # read in as grey img [0, 1]
        img = skio.imread('%s/%s.jpg' % (data_dir, img_name), as_gray=True)
        #img1 = img[yCnt-hR:yCnt+hR,xCnt-hR:xCnt+hR]      
        height, width = np.shape(img)[0:2] 
        img = skt.resize(img, (INPUT_HEIGHT, INPUT_WIDTH)) 
        imgs.append(img)
    #import pdb; pdb.set_trace()
    fScale = INPUT_HEIGHT*1.0/height
    if load_pts:
        # pts are not normalized
        x1 = np.array(df['X1'].values * fScale)
        y1 = np.array(df['Y1'].values * fScale)
        x2 = np.array(df['X2'].values * fScale)
        y2 = np.array(df['Y2'].values * fScale)
        pts1 = np.stack((x1, y1), axis=1)
        pts2 = np.stack((x2, y2), axis=1)
        #pts0 = np.stack((pts1, pts2), axis=1)

    print('Num of images: {}'.format(len(imgs)))

    if load_pts:
        return img_ids, imgs, pts1, pts2
    else:
        return img_ids, imgs


# In[ ]:


data_dir='/home/xulei/projects/nhcs/AVJ_new/data/training/frames'
data_csv='/home/xulei/projects/nhcs/AVJ_new/data/training/ch234.csv'
_, imgs, pts1, pts2 = load_data(data_dir, data_csv)
pts00 = np.append(pts1, pts2, axis=1)
trainData = np.asarray(imgs)
trainLabel= torch.tensor(np.asarray(pts00))


# In[ ]:


#mean = np.mean(trainData)
#std = np.std(trainData)
#trainData = (trainData-mean)/std


# In[ ]:


#from __future__ import division

def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

#standardized_images = normalize_meanstd(rgb_images, axis=(1,2))
trainData = normalize_meanstd(trainData, axis=(1,2))



# In[ ]:


print(trainData.shape)
#print(trainData_norm.shape)


# In[ ]:


data_dir='/home/xulei/projects/nhcs/AVJ_new/data/testing/frames'
data_csv='/home/xulei/projects/nhcs/AVJ_new/data/testing/ch234.csv'
_, imgs, pts1, pts2 = load_data(data_dir, data_csv)
pts01 = np.append(pts1, pts2, axis=1)
testData = np.asarray(imgs)
testLabel= torch.tensor(np.asarray(pts01))
#testData = (testData-mean)/std
testData = normalize_meanstd(testData, axis=(1,2))


# In[ ]:


def show_img(img, pts1, pts2):
    import sys
    import matplotlib.pyplot as plt
    #import matplotlib.patches as patches

    def press(event):
        if event.key == 'q':
            print('Terminated by user')
            sys.exit()
        elif event.key == 'c':
            plt.close()


    x1, y1 = pts1
    x2, y2 = pts2

    plt.ioff()
    fig = plt.figure(frameon=False)
    #fig.canvas.set_window_title('Image')
    fig.canvas.mpl_connect('key_press_event', press)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='gray')
    ax.plot([x1, x2], [y1, y2], 'r-', lw=2)
    print(x1, y1, x2, y2)
    plt.scatter([x1, x2], [y1, y2], s=20)
    plt.show()


# In[ ]:


def address_rotation(address,angle,c,rate):
    x1 = address[1] - c
    y1 = c - address[0]
    #print('x1:%1f y1:%1f' % (x1,y1))
    pi = np.pi
    radian = angle/180 * pi
    #print(torch.cos(radian))
    tx1 = x1*torch.cos(radian)+y1*torch.sin(radian)
    ty1 = y1*torch.cos(radian)-x1*torch.sin(radian)
    #print('tx1:%1f ty1:%1f' % (tx1,ty1))
    address[1] = (c*rate + tx1)/rate
    address[0] = (c*rate - ty1)/rate
    #print('tttx1:%1f ty1:%1f' % (address[0],address[1]))

    return address


def my_rotation_transforms(image, label):
    if random.random() > 0:
        tlabel=torch.tensor(np.zeros(4))
        angle = random.randint(-30, 30)
        image = torch.unsqueeze(image.clone().detach(),0)
        image = TF.rotate(image, angle,interpolation  = torchvision.transforms.InterpolationMode.BILINEAR, expand = False)
        rate = copy.deepcopy(image.shape[1])/128
        newimage = TF.resize(image, (128,128))
        c = 127/2
        angle = torch.tensor(angle)
        ad1 = copy.deepcopy(label[:2])
        rad1 = address_rotation(ad1,angle,c,rate)
        ad2 = copy.deepcopy(label[2:4])
        rad2 = address_rotation(ad2,angle,c,rate)
        tlabel[:2] = rad1.clone().detach()
        tlabel[2:4] = rad2.clone().detach()
        image=torch.squeeze(image)
    # more transforms ...
    else:
        tlabel=torch.tensor(np.zeros(4))
        tlabel[:] = label[:].clone().detach()
        newimage = copy.deepcopy(image.clone().detach())
    return newimage, tlabel


def my_flip(image, label):
    tlabel=torch.tensor(np.zeros(4))
    c = 127/2
    image = image.clone().detach()
    if random.random() >= 0.5:
        newimage = TF.hflip(image)
        tlabel[0] = copy.deepcopy(2*c-label[0])
        tlabel[1] = copy.deepcopy(label[1])
        tlabel[2] = copy.deepcopy(2*c-label[2])
        tlabel[3] = copy.deepcopy(label[3])
    else:
        tlabel[:] = copy.deepcopy(label[:].clone().detach())
        newimage = copy.deepcopy(image)

    if random.random() > 0.5:
        newimage = TF.vflip(newimage)
        tlabel[0] = copy.deepcopy(tlabel[0])
        tlabel[1] = copy.deepcopy(2*c-tlabel[1])
        tlabel[2] = copy.deepcopy(tlabel[2])
        tlabel[3] = copy.deepcopy(2*c-tlabel[3])
    else:
        tlabel[:] = copy.deepcopy(tlabel[:].clone().detach())
        newimage = copy.deepcopy(newimage)

    return newimage, tlabel

def my_shift(image, label):
    if random.random() >= 0:
        rand1 = 0.4 * (random.random()-0.5)
        rand2 = 0.4 * (random.random()-0.5)
        theta = torch.tensor([
                [1,0,rand1],
                [0,1,rand2]
            ], dtype=torch.float)
        grid = F.affine_grid(theta.unsqueeze(0), [1,1,128,128],align_corners=False)
        image= image.double()
        image = torch.unsqueeze(image,0)
        image = torch.unsqueeze(image,0).double()
        newimage = F.grid_sample(image, grid.double(),align_corners=False)
        tlabel=torch.tensor(np.zeros(4))
        #print(("rand1 : %2f rand2 : %2f")%(rand1*128,rand2*128))
        tlabel[0] = copy.deepcopy(label[0]-rand1*64)
        tlabel[1] = copy.deepcopy(label[1]-rand2*64)
        tlabel[2] = copy.deepcopy(label[2]-rand1*64)
        tlabel[3] = copy.deepcopy(label[3]-rand2*64)
        newimage = torch.squeeze(newimage)
    else:
        tlabel=torch.tensor(np.zeros(4))
        newimage = copy.deepcopy(image)
        tlabel[:] = copy.deepcopy(label[:].clone().detach())
    
    return newimage,tlabel

def my_transform(image, label):
    
    simg, slabel = my_shift(image, label)
    simg, slabel = my_flip(simg, slabel)
    simg, slabel = my_rotation_transforms(simg, slabel)
    #print(simg.shape)
    #simg, slabel = my_rotation_transforms(simg, slabel)
    #print(simg.shape)
    
    #print(simg.shape)
    return torch.squeeze(simg), slabel


# In[ ]:


simg, slabel = my_transform(torch.tensor(trainData[1044]), trainLabel[1044])
print(simg.shape)
show_img(trainData[1044], trainLabel[1044][0:2], trainLabel[1044][2:4])
show_img(simg, slabel[0:2], slabel[2:4])
print(pts1[1044])
print(pts2[1044])
print(trainLabel[1044])


# In[ ]:


class trainDataset(Dataset):
    def __init__(self, input, label):
        super(Dataset).__init__()
        self.input = input
        self.label = label

    def __getitem__(self, item):
        img = self.input[item,:,:]
        lab = self.label[item,:]
        trans_img, trans_label = my_transform(img, lab)


        return trans_img, trans_label

    def __len__(self):
        return self.input.shape[0]

train_iter = \
    DataLoader(trainDataset(torch.tensor(trainData).float(), trainLabel.float()),
                    batch_size=BATCH_SIZE,
                    shuffle=True)


# In[ ]:


class Dataset(Dataset):
    def __init__(self, input, label):
        super(Dataset).__init__()
        self.input = input
        self.label = label

    def __getitem__(self, item):
        #input = self.input[item,:,:]
        #label = self.label[item,:]

        return self.input[item,:,:], self.label[item,:]

    def __len__(self):
        return self.input.shape[0]


test_iter = \
    DataLoader(Dataset(torch.tensor(testData).float(), testLabel.float()),
                    batch_size=BATCH_SIZE,
                    shuffle=True)


# In[ ]:


def net_accurary(data_iter, loss_function, net):
    net.eval()
    pixel_loss, loss, n = 0.0, 0.0, 0
    for X, y in data_iter:
        if torch.cuda.is_available():
            X = X.to(device)
            y = y.to(device)

        y_hat = net(X)
        loss += loss_function(y_hat, y).item()
        #pixel_loss += torch.sum(torch.abs(y-y_hat))
        n += 1
    return  loss / n


# In[ ]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # The first Convolution layer (1,20) 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(in_features=32768, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=OUTPUTS)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        input = self.conv1(x)
        input = self.bn1(input)
        input = self.relu(input)
        input = self.pool_1(input)
        input = self.dropout1(input)

        input = self.conv2(input)
        input = self.bn2(input)
        input = self.relu(input)
        input = self.pool_2(input)
        input = self.dropout2(input)

        input = self.conv3(input)
        input = self.bn3(input)
        input = self.relu(input)
        input = self.pool_3(input)
        input = self.dropout3(input)
        input = input.flatten(start_dim=1)

        input = self.fc1(input)
        input = self.relu(input)
        input = self.dropout4(input)
        input = self.fc2(input)
        input = self.relu(input)
        input = self.dropout5(input)
        input = self.fc3(input)
        output = self.relu(input)


        return output




# In[ ]:


if pre:
            
    net = torch.load('./best_model.pkl')
else:
    net = Model()

if torch.cuda.is_available():
    net.cuda()
# Cross Entropy Loss function
loss_function = nn.MSELoss()
loss_function_test = nn.L1Loss()

# SGD optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay =1e-5)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.95)
summary(net, (128, 128))


# In[ ]:


def train(net):
    
    best_tn = 1000
    for epoch in range(EPOCHS):
        total_loss = 0.0
        n1 = 0
        for X, y in train_iter:
            if torch.cuda.is_available():
                X = X.to(device)
                y = y.to(device)
            #print(X.dtype)
            n1 += 1
            y_hat = net(X.float())
            #print('success')
            l = loss_function(y_hat.float(), y.float())
            #print(l.dtype)
            total_loss += loss_function(y_hat, y).item()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()


        

            
        
        test_loss = net_accurary(test_iter, loss_function, net)
        
        if best_tn > test_loss:
            best_tn = test_loss
            torch.save(net,str('./best_model.pkl'))


        if epoch %5==0:
            text = 'epoch: %d,   ' % (epoch + 1)
            
            text += 'train_loss: %.8f; test_loss: %.8f'% (total_loss/n1,test_loss)
            text += ('best_test_accuracy: %.8f'%(best_tn))
            print(text)
            

        
        

        


train(net)
            


# In[ ]:


class trainDataset(Dataset):
    def __init__(self, input, label):
        super(Dataset).__init__()
        self.input = input
        self.label = label

    def __getitem__(self, item):
        img = self.input[item,:,:]
        lab = self.label[item,:]
        trans_img, trans_label = my_transform(img, lab)


        return self.input[item,:,:], self.label[item,:]

    def __len__(self):
        return self.input.shape[0]

train_iter = \
    DataLoader(trainDataset(torch.tensor(testData).float(), testLabel.float()),
                    batch_size=1,
                    shuffle=True)




for X, y in train_iter:
    if torch.cuda.is_available():
        X = X.to(device)
        y = y.to(device)
            #print(X.dtype)
    y_hat = net(X.float())
    for k in range(X.shape[0]):
        show_img(X[k].cpu(), y[k][0:2].cpu(), y[k][2:4].cpu())
        show_img(X[k].cpu(), y_hat[k][0:2].cpu().detach().numpy(), y_hat[k][2:4].cpu().detach().numpy())
    break

