import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import matplotlib.pyplot as plt



class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels



class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        
#        org_img = image[:,:,0]/255
#        tmp = np.ones((66,66))
#        tmp[1:65,1:65] = org_img
#        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#        ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#        plt.show()
        
        return image, boxes, labels




class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels




class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        #print(image.shape)
        
#        org_img = image[:,:,0]/255
#        tmp = np.ones((66,66))
#        tmp[1:65,1:65] = org_img
#        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#        ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#        plt.show()
        
        height, width, depth = image.shape
        ratio = random.uniform(1, 2)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
#
        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image
        
        

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:4] += (int(left), int(top))
        
        #print(image.shape)

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        height, width, _ = image.shape
        strategy = random.randint(4)
        
#        org_img = image[:,:,0]/255
#        tmp = np.ones((66,66))
#        tmp[1:65,1:65] = org_img
#        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#        ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#        
#        plt.show()
        
        if strategy == 0:
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0:4:2] = width - boxes[:, 2::-2]
        elif strategy == 1:
            image = image[::-1, :]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
            
        
            
#        org_img = image[:,:,0]/255
#        tmp = np.ones((66,66))
#        tmp[1:65,1:65] = org_img
#        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#        ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#        
#        plt.show()
        return image, boxes, classes


    

class SSDAugmentation(object):
    def __init__(self, size=64, mean=(104, 117, 123), phase = 'train'):
        self.mean = mean
        self.size = size
        if phase == 'train':
            self.augment = Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                RandomMirror(),
                Expand(self.mean),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean)
            ])
        else:
            self.augment = Compose([
                ConvertFromInts(),
                #ToAbsoluteCoords(),
                #PhotometricDistort(),
                #Expand(self.mean),
                #RandomSampleCrop(),
                #RandomMirror(),
                #ToPercentCoords(),
                #Resize(self.size),
                SubtractMeans(self.mean)
            ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
