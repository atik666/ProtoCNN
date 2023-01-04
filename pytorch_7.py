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
from pytorch_metric_learning import losses
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


trainx1 = torch.from_numpy(np.array(X1[1:500])).float()
trainx1 = trainx1.permute(0,3,1,2).cuda()
y_train1 = torch.from_numpy(y1[1:500]).float().cuda()

trainx2 = torch.from_numpy(np.array(X2[1:500])).float()
trainx2 = trainx2.permute(0,3,1,2).cuda()
y_train2 = torch.from_numpy(y2[1:500]).float().cuda()


""""""
from torch.utils.data import DataLoader, TensorDataset 

label1 = torch.tensor(y_train1,dtype=torch.long)
label2 = torch.tensor(y_train2,dtype=torch.long)

dataset = TensorDataset(trainx1, label1, trainx2, label2)

trainloader = DataLoader(dataset,
                        shuffle=True,
                        batch_size=20)

""""""
val1 = torch.from_numpy(np.array(X1[500:600])).float()
val1 = val1.permute(0,3,1,2).cuda()
y_val1 = torch.from_numpy(y1[500:600]).float().cuda()

val2 = torch.from_numpy(np.array(X2[500:600])).float()
val2 = val2.permute(0,3,1,2).cuda()
y_val2 = torch.from_numpy(y2[500:600]).float().cuda()


""""""
from torch.utils.data import DataLoader, TensorDataset 

y_val1 = torch.tensor(y_val1,dtype=torch.long)
y_val2 = torch.tensor(y_val2,dtype=torch.long)

val_dataset = TensorDataset(val1, y_val1, val2, y_val2)

val_loader = DataLoader(val_dataset,
                        shuffle=True,
                        batch_size=20)
    
    
def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def euclidean_metric(proto, samples):
    
    logits = []
    for i in range(samples.size(0)):
        logit = -((proto - samples[i])**2).sum(dim=0)
        logits.append(logit)    
    
    logits = torch.FloatTensor(logits)
        
    return logits

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
    
def loss_dist(batch, target=0):
    
    train1, train2 = batch[0].cuda(), batch[2].cuda()
    
    if target == 0:
        
        x1, x2 = train1, train2
        
    elif target == 1:
        
        x1, x2 = train2, train1
        
        
    proto = model(x1, x2)
    proto1 = proto[0].mean(dim=0)
    #proto2 = proto[1].mean(dim=0)
    
    dist1 = euclidean_metric(proto1,proto[0])
    #dist2 = euclidean_metric(proto1,proto[1])
    
    #dist = torch.stack((dist1,dist2), dim = 1).cuda()
        
    return dist1

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

    def forward(self, x1, x2):
        
        x1 = self.encoder(x1)
        x1 = x1.reshape(x1.size(0), -1)
        #x = x.mean(dim=0)
        
        x2 = self.encoder(x2)
        x2 = x2.reshape(x2.size(0), -1)
        #x = x.mean(dim=0)
        
        return x1, x2

model = Convnet().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
criterion = ContrastiveLoss()

max_epoch = 5
for epoch in range(1, max_epoch + 1):
    lr_scheduler.step()
    
    model.train()
    
    for i, batch in enumerate(trainloader):

        
        dist0 = loss_dist(batch, target = 0)
        dist1 = loss_dist(batch, target = 1)
        
        #print(dist0, "__", dist1)
        
        trainx0 = batch[0].cuda()
        label0 = batch[1].cuda()
        trainx1 = batch[2].cuda()
        label1 = batch[3].cuda()
        
        proto0, proto1 = model(trainx1, trainx1)
        proto0_mean = proto0.mean(dim=0)
        proto1_mean = proto1.mean(dim=0)
        
        loss0 = criterion(proto0, proto1, label0)
        loss1 = criterion(proto1, proto0, label1)
      
        Softmax = torch.nn.Softmax(dim=1)
        logits0 = Softmax(dist0)
        logits1 = Softmax(dist1)
        
        acc0 = count_acc(logits0, label0)
        acc1 = count_acc(logits1, label1)
        
        #print('acc: ', acc)
        
        print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
          .format(epoch, i, len(trainloader), loss0.item(), acc0))
        
        print('epoch1 {}, train1 {}/{}, loss1={:.4f} acc1={:.4f}'
          .format(epoch, i, len(trainloader), loss1.item(), acc1))
        
        optimizer.zero_grad()
        #loss.requires_grad = False
        loss0.backward()
        loss1.backward()
        optimizer.step()

    for i, batch in enumerate(val_loader):
        
        trainx1 = batch[0].cuda()      
        trainx2 = batch[2].cuda()
        
        label1 = batch[1].cuda()
        label2 = batch[3].cuda()
        
        proto = model(trainx1, trainx2)
        proto1 = proto[0].mean(dim=0)
        proto2 = proto[1].mean(dim=0)
        
        dist1 = euclidean_metric(proto1,proto[0])
        dist2 = euclidean_metric(proto2,proto[1])
        
        dist = torch.stack((dist1,dist2), dim = 1).cuda()
      
        Softmax = torch.nn.Softmax(dim=1)
        logits = Softmax(dist)
        
        loss = criterion(proto[0], proto[1], label1)
        
        #print('loss: ', loss)
        
        acc = count_acc(logits, label1)
        
        #print('acc: ', acc)
        
        print('epoch {}, val {}/{}, loss={:.4f} acc={:.4f}'
          .format(epoch, i, len(val_loader), loss.item(), acc))

        












