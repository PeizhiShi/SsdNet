from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.utils import progress_bar
import cupy as cp
import random

import time


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
#parser.add_argument('--dataset_root', default=VOC_ROOT,
#                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
#parser.add_argument('--start_iter', default=0, type=int,
#                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
#parser.add_argument('--cuda', default=True, type=str2bool,
#                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
#parser.add_argument('--gamma', default=0.1, type=float,
#                    help='Gamma update for SGD')
#parser.add_argument('--visdom', default=False, type=str2bool,
#                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)



def remove_redudant(imgs, labels):
    #print(imgs.shape)
    #print(labels)
    
    sel = []
    targets = []
    for i in range(len(labels)):
        if labels[i].shape[0] != 0:
            sel.append(i)
            targets.append(labels[i])
    return imgs[sel,:,:],targets

def get_lvec(labels):
    results = np.zeros(24)
    
    for i in labels:
        results[int(i)] += 1
    
    return results

def eval_metric(pre,trul,tp):
    precision = tp/pre
    precision[np.isnan(precision)]=0
    precision[precision>1]=1
    
    recall = tp/trul
    recall[np.isnan(recall)] = 0
    recall[recall>1] = 1
    
    return precision, recall

def rg(val,factor):
    
    if val < 0:
        val = 0
    
    if val > 1:
        val = 1
    
    return int(val * factor)

def rotate_sample(sample,rotation, reverse = False):

    if reverse:
        if rotation == 1:
            sample = np.rot90(sample, -2, (0,1)).copy()  
        elif rotation == 2:
            sample = np.rot90(sample, -1, (0,1)).copy()  
        elif rotation == 3:
            sample = np.rot90(sample, -1, (1,0)).copy()  
        elif rotation == 4:
            sample = np.rot90(sample, -1, (2,0)).copy()  
        elif rotation == 5:
            sample = np.rot90(sample, -1, (0,2)).copy() 
    else:
        if rotation == 1:
            sample = np.rot90(sample, 2, (0,1)).copy()  
        elif rotation == 2:
            sample = np.rot90(sample, 1, (0,1)).copy()  
        elif rotation == 3:
            sample = np.rot90(sample, 1, (1,0)).copy()  
        elif rotation == 4:
            sample = np.rot90(sample, 1, (2,0)).copy()  
        elif rotation == 5:
            sample = np.rot90(sample, 1, (0,2)).copy() 
        
    return sample

def val(net, criterion):
    
    
    dataset = VOCDetection(None, transform=SSDAugmentation(cfg['min_dim'],MEANS,'val'),phase='val')
    data_loader = data.DataLoader(dataset, 1,num_workers=1,collate_fn=detection_collate,pin_memory=True)
    
    data_size = len(dataset)
    
    batch_iterator = iter(data_loader)
  
    loss_all = 0
    
    with torch.no_grad():
        for step in range(0, data_size):
            images, targets = next(batch_iterator)
            
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [ann.cuda() for ann in targets]
            
            
            out = net(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            
            loss_all = loss_all + loss.data
  
            progress_bar(step, data_size, ' avg loss %.4f' % (loss_all/step))
      
    return loss_all/data_size
        


def tensor_to_float(val):
    
    if val < 0:
        val = 0
    
    if val > 1:
        val = 1
    
    return float(val)



def set_lr(lr,optimizer):
    for g in optimizer.param_groups:
        g['lr'] = args.lr
  

def train():
    
    
    args.dataset = 'VOC'
    
    args.resume = 'vocinit'
    
 
    args.batch_size = 16    
    args.num_workers = 2 
    
 
    ssd_net = build_ssd(cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

    if args.resume == 'vocinit':
        args.resume = 'weights/base/ssd_300_VOC0712.pth'
        
        print('Resuming training, loading {}...'.format(args.resume))
        
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        
        ssd_net.load_weights(args.resume, True)
    elif args.resume == 'ours':
        args.resume = 'weights/VOC.pth' 
        
        print('Resuming training, loading {}...'.format(args.resume))
        
        ssd_net.load_weights(args.resume, False)
        
    else:   #random init
        ssd_net.vgg.apply(weights_init)
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    net = net.cuda()
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr) 
   
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, 3)
    
    net.train()
    
    valloss = 10000000000
   
    start_time = time.time()
    for epoch in range(0,4):

        dataset = VOCDetection(None, transform=SSDAugmentation(cfg['min_dim'],MEANS), phase = 'train')
        data_loader = data.DataLoader(dataset, args.batch_size,
                                      num_workers=args.num_workers,collate_fn=detection_collate,
                                      pin_memory=True)
        print('epoch ',epoch)
        
        loss_all = 0
        
        if epoch <= 1:
            args.lr = 0.0001
            set_lr(args.lr,optimizer)
        elif epoch == 2:
            args.lr = 0.00001
            set_lr(args.lr,optimizer)
            
        
        print('learning rate:',  args.lr)
        for iteration, (images,targets) in enumerate(data_loader):
            images, targets = remove_redudant(images,targets)
    
            images = Variable(images.cuda())
            
            
            
            with torch.no_grad():
                targets = [ann.cuda() for ann in targets]
            
    
            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            
            loss_all = loss_all + loss.data
            

    
            if iteration % 500 == 0:
                print('iter %4d'%iteration, ' || Loss: %2.4f ||' % (loss.data))

            
            if iteration % 2000 == 0 and iteration > 0:
                print("---total training time: %s seconds ---" % (time.time() - start_time))
                curloss = val(net, criterion)
                if valloss > curloss:
                    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.dataset +'.pth')
                    valloss = curloss
                    print('model saved')
                


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()



if __name__ == '__main__':
    train()























