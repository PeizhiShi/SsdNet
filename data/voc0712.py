"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


import utils.binvox_rw
import csv
import random
import cupy as cp
import os
from pathlib import Path
import matplotlib.pyplot as plt

#VOC_CLASSES = (  # always index 0
#    'aeroplane', 'bicycle', 'bird', 'boat',
#    'bottle', 'bus', 'car', 'cat', 'chair',
#    'cow', 'diningtable', 'dog', 'horse',
#    'motorbike', 'person', 'pottedplant',
#    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASSES = ('O ring', 'Through hole', 'Blind hole', 
               'Triangular passage', 'Rectangular passage', 
               'Circular through slot', 'Triangular through slot', 
               'Rectangular through slot', 'Rectangular blind slot',
               'Triangular pocket', 'Rectangular pocket', 
               'Circular end pocket', 'Triangular blind step', 
               'Circular blind step', 'Rectangular blind step', 
               'Rectangular through step' , '2-sides through step', 
               'Slanted through step', 'Chamfer', 'Round', 
               'Vertical circular end blind slot', 
               'Horizontal circular end blind slot', 
               '6-sides passage', '6-sides pocket')
  

# note: if you used our download scripts, this should be right
#VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


#class VOCAnnotationTransform(object):
#    """Transforms a VOC annotation into a Tensor of bbox coords and label index
#    Initilized with a dictionary lookup of classnames to indexes
#
#    Arguments:
#        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
#            (default: alphabetic indexing of VOC's 20 classes)
#        keep_difficult (bool, optional): keep difficult instances or not
#            (default: False)
#        height (int): height
#        width (int): width
#    """
#
#    def __init__(self, class_to_ind=None, keep_difficult=False):
#        self.class_to_ind = class_to_ind or dict(
#            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
#        self.keep_difficult = keep_difficult
#
#    def __call__(self, target, width, height):
#        """
#        Arguments:
#            target (annotation) : the target annotation to be made usable
#                will be an ET.Element
#        Returns:
#            a list containing lists of bounding boxes  [bbox coords, class name]
#        """
#        res = []
#        for obj in target.iter('object'):
#            difficult = int(obj.find('difficult').text) == 1
#            if not self.keep_difficult and difficult:
#                continue
#            name = obj.find('name').text.lower().strip()
#            bbox = obj.find('bndbox')
#
#            pts = ['xmin', 'ymin', 'xmax', 'ymax']
#            bndbox = []
#            for i, pt in enumerate(pts):
#                cur_pt = int(bbox.find(pt).text) - 1
#                # scale height or width
#                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
#                bndbox.append(cur_pt)
#            label_idx = self.class_to_ind[name]
#            bndbox.append(label_idx)
#            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
#            # img_id = target.find('filename').text[:-4]
#
#        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

def rotate_sample24(sample):
    
#    strategy = random.randint(0,3)
#    
#    if strategy <= 2:
#        sample = cp.flip(sample,strategy).copy()
    
     
    rotation = random.randint(0,23)
    
    if rotation == 1:
        sample = cp.rot90(sample,1,(1,2)).copy()  
    elif rotation == 2:
        sample = cp.rot90(sample,2,(1,2)).copy()  
    elif rotation == 3:
        sample = cp.rot90(sample,1,(2,1)).copy()  
    elif rotation == 4:
        sample = cp.rot90(sample,1,(0,1)).copy()  
    elif rotation == 5:
        sample = cp.rot90(sample,1,(0,1)).copy()  
        sample = cp.rot90(sample,1,(1,2)).copy()  
    elif rotation == 6:
        sample = cp.rot90(sample,1,(0,1)).copy()  
        sample = cp.rot90(sample,2,(1,2)).copy()  
    elif rotation == 7:
        sample = cp.rot90(sample,1,(0,1)).copy()  
        sample = cp.rot90(sample,1,(2,1)).copy()  
    elif rotation == 8:
        sample = cp.rot90(sample,1,(1,0)).copy()  
    elif rotation == 9:
        sample = cp.rot90(sample,1,(1,0)).copy()  
        sample = cp.rot90(sample,1,(1,2)).copy()  
    elif rotation == 10:
        sample = cp.rot90(sample,1,(1,0)).copy()  
        sample = cp.rot90(sample,2,(1,2)).copy()  
    elif rotation == 11:
        sample = cp.rot90(sample,1,(1,0)).copy()  
        sample = cp.rot90(sample,1,(2,1)).copy()  
    elif rotation == 12:
        sample = cp.rot90(sample,2,(1,0)).copy()  
    elif rotation == 13:
        sample = cp.rot90(sample,2,(1,0)).copy()  
        sample = cp.rot90(sample,1,(1,2)).copy()  
    elif rotation == 14:
        sample = cp.rot90(sample,2,(1,0)).copy()  
        sample = cp.rot90(sample,2,(1,2)).copy()  
    elif rotation == 15:
        sample = cp.rot90(sample,2,(1,0)).copy()  
        sample = cp.rot90(sample,1,(2,1)).copy()  
    elif rotation == 16:
        sample = cp.rot90(sample,1,(0,2)).copy()  
    elif rotation == 17:
        sample = cp.rot90(sample,1,(0,2)).copy()  
        sample = cp.rot90(sample,1,(1,2)).copy()  
    elif rotation == 18:
        sample = cp.rot90(sample,1,(0,2)).copy()  
        sample = cp.rot90(sample,2,(1,2)).copy()  
    elif rotation == 19:
        sample = cp.rot90(sample,1,(0,2)).copy()  
        sample = cp.rot90(sample,1,(2,1)).copy()  
    elif rotation == 20:
        sample = cp.rot90(sample,1,(2,0)).copy()  
    elif rotation == 21:
        sample = cp.rot90(sample,1,(2,0)).copy()  
        sample = cp.rot90(sample,1,(1,2)).copy()  
    elif rotation == 22:
        sample = cp.rot90(sample,1,(2,0)).copy()  
        sample = cp.rot90(sample,2,(1,2)).copy()  
    elif rotation == 23:
        sample = cp.rot90(sample,1,(2,0)).copy()  
        sample = cp.rot90(sample,1,(2,1)).copy()  
        
    return sample


def rotate_sample(sample,rotation, reverse = False):

    if reverse:
        if rotation == 1:
            sample = cp.rot90(sample, -2, (0,1)).copy()  
        elif rotation == 2:
            sample = cp.rot90(sample, -1, (0,1)).copy()  
        elif rotation == 3:
            sample = cp.rot90(sample, -1, (1,0)).copy()  
        elif rotation == 4:
            sample = cp.rot90(sample, -1, (2,0)).copy()  
        elif rotation == 5:
            sample = cp.rot90(sample, -1, (0,2)).copy() 
    else:
        if rotation == 1:
            sample = cp.rot90(sample, 2, (0,1)).copy()  
        elif rotation == 2:
            sample = cp.rot90(sample, 1, (0,1)).copy()  
        elif rotation == 3:
            sample = cp.rot90(sample, 1, (1,0)).copy()  
        elif rotation == 4:
            sample = cp.rot90(sample, 1, (2,0)).copy()  
        elif rotation == 5:
            sample = cp.rot90(sample, 1, (0,2)).copy() 
        
    return sample

def get_label_from_csv(filename):
    
    retarr = np.zeros((0,7))
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            items = row[0].split(',')
            retarr = np.insert(retarr,0,np.asarray(items),0)
            
    
    return retarr[:,6]


def achieve_legal_model(list_IDs, list_size, factor):
    
    while True:
        filename = list_IDs[random.randint(0,len(list_IDs)-1)]
        
        cur_factor = list_size[str(filename)]
        
        
        if cur_factor >= factor:
            continue
        
        
        label = int(os.path.basename(filename).split('_')[0])
        with open(filename, 'rb') as f:
            model = utils.binvox_rw.read_as_3d_array(f).data
        
        return label, model
    



def achieve_random_model(list_IDs, list_size):
    
    #num_features = 2
    num_features = random.randint(2,10)
    factor = 76 - 6*num_features
    
    #print('n features:', num_features)
    
    components = np.zeros((num_features,64,64,64))
    
    ret_model = np.ones((64,64,64))
    
    model_label = np.zeros((0))
    
    for i in range(num_features):
        
        label, model = achieve_legal_model(list_IDs, list_size, factor)
    
        model_label = np.append(model_label,label)
        components[i,:,:,:] = rotate_sample24(model)
        ret_model = ret_model * components[i,:,:,:]
            
    return ret_model, model_label, components


    
def create_img(obj3d, rotation, grayscale = False):
    
    cursample = obj3d.copy()
    #cursample = cp.array(cursample)
    cursample = rotate_sample(cursample,rotation)
    
    img0 = np.zeros((cursample.shape[1],cursample.shape[2]))
    
    for i in range(img0.shape[0]):
        for j in range(img0.shape[1]):
            for d in range(cursample.shape[0]):
                
                if cursample[d,i,j] == True:
                    img0[i,j] = d/cursample.shape[0]
                    break
                
                if d == cursample.shape[0] - 1:
                    img0[i,j] = 1
                    break
    
    if img0.mean() == 0:
        flag = False
    else:
        flag = True
                    
                
    if grayscale == False:
        img1 = img0.copy()
        img2 = img0.copy()   
    
        img = np.stack((img0,img1,img2), axis=2)
    else:
        img = img0
    
    img = img * 255
    
    
    return img, flag

def achieve_model_gt(model_label, model_components, rotation):
    
    img_label = np.zeros((10,6))
    img_nbox = 0
    
    for i in range(len(model_label)):
        cur_component = rotate_sample(model_components[i,:,:,:], rotation)
        
        region = np.where(cur_component[0,:,:]==0)
        
        if len(region[0]) == 0:
            continue
        
        img_label[img_nbox,5] = model_label[i] #+ 1
        img_label[img_nbox,0] = (region[1].min())/64.0
        img_label[img_nbox,1] = (region[0].min())/64.0
        img_label[img_nbox,2] = (region[1].max()+1)/64.0
        img_label[img_nbox,3] = (region[0].max()+1)/64.0
        
        region = np.where(cur_component[:,:,:]==0)
        img_label[img_nbox,4] = (region[0].max())/64.0
        img_nbox += 1
    
    return img_label[0:img_nbox,:]

def create_partition(num_train_per_class = 30, num_val_per_class = 30):
    
    num_classes = 24
    counter = cp.zeros(num_classes)
    partition = {}
    for i in range(num_classes): 
        partition['train',i] = []
        partition['val',i] = []
        
#    for i in range(1,12):
#        partition['test',i] = []
        
    with open(os.devnull, 'w') as devnull:
        for filename in Path('data/FNSet/').glob('*.binvox'):
            namelist = os.path.basename(filename).split('_')
            
            
            label = int(namelist[0])
                
            counter[label] += 1
            
            items = [filename]
            
            if counter[label] % 10 < 9:
                partition['train',label] += items
            elif counter[label] % 10 == 9:
                partition['val',label] += items
    

    
    ret = {}
    ret['train'] = []
    ret['val'] = []
    
#    for testidx in range(1,12):
#        ret['test', testidx] = partition['test', testidx]
    
           
    for i in range(num_classes):      
        random.shuffle(partition['train',i])  
        random.shuffle(partition['val',i])  
        
        ret['train'] += partition['train',i][0:num_train_per_class]
        ret['val'] += partition['val',i][0:num_val_per_class]
    
    random.shuffle(ret['train'])  
    random.shuffle(ret['val'])  
        
    return ret

def create_test(testidx):
    partition = []
    with open(os.devnull, 'w') as devnull:
        for filename in Path('data/MulSet/set' + str(testidx) + '/').glob('*.binvox'):
            partition += [filename]
    return partition


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, list_IDs = None, transform=None, phase = 'train'):
        
        self.transform = transform
        self.list_IDs = list_IDs
        self.phase = phase
        
        if phase == 'test':
            self.num_samples = len(list_IDs)*6
            #self.is_test = True
        elif phase == 'val':
            self.DIR = 'data/ValSet/'
            self.num_samples = int(len([name for name in os.listdir(self.DIR) if os.path.isfile(os.path.join(self.DIR, name))])/2)
            
        elif phase == 'train':
            self.DIR = 'data/TrSet/'
            self.num_samples = int(len([name for name in os.listdir(self.DIR) if os.path.isfile(os.path.join(self.DIR, name))])/2)
            
            #self.num_samples = 143469
            #self.is_test = False
        
#        with open('data/minfo.csv', mode='r') as infile:
#            reader = csv.reader(infile)
#            self.list_size = {rows[0]:int(rows[1]) for rows in reader}



    def __getitem__(self, idx):
        
        
        if self.phase == 'test':
            rotation = int(idx%6)
            m_idx = int(idx/6)
        
            if rotation == 0:
                filename = self.list_IDs[m_idx]
                with open(filename, 'rb') as f:
                    self.cur_model = utils.binvox_rw.read_as_3d_array(f).data
                    self.cur_model_label = get_label_from_csv(str(filename).replace('.binvox','.csv'))
                    
            img, _ = create_img(self.cur_model, rotation)
            
            img, _, _ = self.transform(img, 0, 0)
            
        
        else: #train, val
            
            filename = self.DIR+str(idx)
            img = plt.imread(filename+'.png',format='grayscale')
            target=np.load(filename+'.npy')
                
            img = img[:,:,:3]
            
            
            #da strategy 3
            if self.phase == 'train' and random.randint(0,1) == 0:

                
                filename2 = self.DIR+str(random.randint(0,self.num_samples-1))
                img2 = plt.imread(filename2+'.png',format='grayscale')
                target2=np.load(filename2+'.npy')
                    
                img2 = img2[:,:,:3]
                
                target = np.concatenate((target,target2),axis=0)
                
#                org_img = img[:,:,0]/255
#                tmp = np.ones((66,66))
#                tmp[1:65,1:65] = org_img
#                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#                ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#                plt.show()
                
                img = np.maximum(img,img2)
                
#                org_img = img2[:,:,0]/255
#                tmp = np.ones((66,66))
#                tmp[1:65,1:65] = org_img
#                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#                ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#                plt.show()
#                
#                org_img = img[:,:,0]/255
#                tmp = np.ones((66,66))
#                tmp[1:65,1:65] = org_img
#                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#                ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#                plt.show()
                #print(img.max())
                #print(img.min())
                
                
            
            img, boxes, labels = self.transform(img, target[:, :5], target[:, 5])
            
            self.cur_model_label = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
            
        return torch.from_numpy(img).permute(2, 0, 1).float(), self.cur_model_label

    def __len__(self):
        return self.num_samples

