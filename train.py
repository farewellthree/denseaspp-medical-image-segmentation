import os
import sys
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dt
from torch.autograd import Variable

from CToperation import *
from DenseASPP-master.cfgs.DenseASPP161 import Model_CFG
from DenseASPP-master.models.DenseASPP import DenseASPP


def read_data(num_data,root='.'):
    val_list = np.random.choice(np.arange(1,num_data+1),int(num_data/5))
    train_list = no.setdiff1d(np.arange(1,num_data+1),val_list)
    
    data_train = [os.path.join(root,'train','data'+str(i)+'.npy')for i in train_list]
    label_train = [os.path.join(root,'label','label'+str(i)+'.npy')for i in train_list]
    data_val = [os.path.join(root,'train','data'+str(i)+'.npy')for i in val_list]
    label_val = [os.path.join(root,'label','label'+str(i)+'.npy')for i in val_list]
    
    return data_train,label_train,data_val,label_val

def normalization(img):
    liver_win = [-200, 250]
    newimg = (img - liver_win[0]) / (liver_win[1] - liver_win[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg * 2 - 1

def transforms(data,label):
    data = normalization(data)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return data,label
    
    
class dataset(dt.Dataset):
    def __init__(self,train,transform=None):
        data_train,label_train,data_val,label_val=read_data(num_data)
        
        self.data_list = data_train if train else data_val
        self.label_list = label_train if train else label_val
        
        self.transform = transform
        
    def __getitem__(self,idx):
        data = np.load(self.data_list[idx])
        label = np.load(self.data_list[idx])
        
        if transform is not None:
            data,label = self.transform(data,label)
        return data,label
    
    def __len__(self):
        return len(self.data_list)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()

    def forward(self,data,label):
        #batch size
        N = label.size(0)
        data_matrix = data.view(N,-1)
        label_matrix = label.view(N,-1)
        
        intersection = data_matrix*label_matrix
        loss = 2*intersection.sum(1)/(data_matrix.sum(1)+label_matrix.sum(1))
        loss = 1-loss.sum()/N 
        return loss

def accuracy(data,label):
    N = data.shape[0]
    
    intersection = np.dot(data,label)
    acc = 2*np.sum(intersection,axis=1)/(np.sum(data,axis=1)+np.sum(label,axis=1))
    acc = np.sum(acc)
    return acc

    
def train():
    voc_train = dataset(True,transforms)
    voc_test = dataset(False,transforms)    
    train_data = dt.DataLoader(voc_train, batch_size=8, num_workers=2)
    valid_data = dt.DataLoader(voc_test, batch_size=8, num_workers=2)

    model = DenseASPP(Model_CFG, n_class=2, output_stride=8)
    criterion = DiceLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

    for epoch in range(50):
        if epoch == 30:
            optimizer.set_learning_rate(optimizer.learning_rate*0.1)
        prev_time = datetime.now()
        
        train_loss = 0
        train_acc = 0
        model = model.train()
        for Data in train_data:
            data_batch = Variable(Date[0].cuda())
            label_batch = Variable(Data[1].cuda())

            output = model(data_batch)
            output = F.log_softmax(output,dim=1)
            loss = criterion(output,label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]

            train_pred = output.data.cpu().numpy() #加.max[1]是啥意思？
            train_true = label_batch.data.cpu().numpy()
            train_acc += accuracy(train_pred,train_true)

        model = model.eval()
        eval_loss = 0
        eval_acc = 0
        for Data in valid_data:
            data_batch = Variable(Date[0].cuda())
            label_batch = Variable(Data[1].cuda())

            output = model(data_batch)
            output = F.log_softmax(output,dim=1)
            loss = criterion(output,label_batch)
            eval_loss += loss.data[0]

            eval_pred = output.data.cpu().numpy()
            eval_true = label_batch.data.cpu().numpy()
            eval_acc += accuracy(eval_pred,eval_true)
        
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        epoch_str = ('Epoch:{},Train loss:{:.5f},Train acc:{:.5f},Eval loss:{:.5f},Eval acc:{:.5f}'.format(
            epoch,train_loss/len(train_data),train_acc/len(train_data),eval_loss/len(valid_data),eval_acc/len(valid_data)))

        print (epoch_str+time_str)


if __name__ == '__main__':
    train()
        






            
#path = 'F:\\lib and data\\beilun'
#a,b = find_files(path)
#produce_data(a,b)
      
        
        
    





                
        
