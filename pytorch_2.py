import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange
from time import sleep
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
use_gpu = torch.cuda.is_available()
import os
import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import math
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

def import_img(path):
    training_data = []
    for img in os.listdir(path):
        try :
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
            img_array = cv2.resize(img_array, (64, 64))
            img_array = np.squeeze(img_array).astype('float64')
            img_array /= 255
            training_data.append(img_array)
        except:
            pass      
    
    return training_data

# path = '/home/atik/Documents/UMAML_FSL/data/train/n01532829'

X1 = import_img('/home/atik/Documents/UMAML_FSL/data/train/n01532829')
#y1 = tf.convert_to_tensor(np.zeros(np.array(X1.shape[0]).astype('int32')).astype('float32'))
y1 = np.zeros(len(X1)).astype('float32')
X2 = import_img('/home/atik/Documents/UMAML_FSL/data/train/n02113712')
y2 = np.zeros(len(X2)).astype('float32')+1
    
# X = np.concatenate([X1, X2], axis=0)
# y = np.concatenate([y1, y2], axis=0)

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.9, shuffle=True)

# trainx = torch.from_numpy(x_train).float()
# testx = torch.from_numpy(x_test).float()

# trainx = trainx.permute(0,3,1,2).cuda()
# testx = testx.permute(0,3,1,2).cuda()

# y_train = torch.from_numpy(y_train).float().cuda()
# y_test = torch.from_numpy(y_test).float().cuda()

trainx1 = torch.from_numpy(np.array(X1[0:50])).float()
trainx1 = trainx1.permute(0,3,1,2).cuda()
y_train1 = torch.from_numpy(y1[0:50]).float().cuda()

trainx2 = torch.from_numpy(np.array(X2[0:50])).float()
trainx2 = trainx2.permute(0,3,1,2).cuda()
y_train2 = torch.from_numpy(y2[0:50]).float().cuda()


""""""
from torch.utils.data import DataLoader
trainloader = DataLoader(trainx1,
                        shuffle=True,
                        num_workers=8,
                        batch_size=5)


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)
    
    
def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(proto, samples):
    
    logits = []
    for i in range(samples.size(0)):
        logit = -((proto - samples[i])**2).sum(dim=0)
        logits.append(logit)    
    
    logits = torch.FloatTensor(logits)
        
    return logits

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.reshape(x.size(0), -1)

model = Convnet().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

max_epoch = 10
for epoch in range(1, max_epoch + 1):
    lr_scheduler.step()
    
    model.train()
    
    proto1 = model(trainx1)
    proto1 = proto1.mean(dim=0)
    
    proto2 = model(trainx2)
    proto2 = proto2.mean(dim=0)
    
    label1 = torch.tensor(y_train1,dtype=torch.long)
    label2 = torch.tensor(y_train2,dtype=torch.long)
    
    samples1 = model(trainx1)
    
    dist1 = euclidean_metric(proto1,samples1)
    dist2 = euclidean_metric(proto2,samples1)
    
    dist = torch.stack((dist1,dist2), dim = 1).cuda()
  
    Softmax = torch.nn.Softmax(dim=1)
    logits = Softmax(dist)
      
    #loss = F.cross_entropy(logits, label.long())
    
    loss = F.cross_entropy(logits, label1)
    
    print('loss: ', loss)
    
    #output = loss(logits, label1)
    
    acc = count_acc(logits, label1)
    
    print('acc: ', acc)
    
    optimizer.zero_grad()
    loss.requires_grad = True
    loss.backward()
    optimizer.step()

















