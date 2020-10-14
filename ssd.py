import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import cfg#, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        #self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg#[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size
        
        #

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm512 = L2Norm(512, 20)
        self.L2Norm256 = L2Norm(256, 20)
        self.L2Norm128 = L2Norm(128, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        #if phase == 'test':
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 300, 0.01, 0.45)

    def forward(self, x, phase = 'train'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(9):
            x = self.vgg[k](x)
        
        s = self.L2Norm128(x)
        #print(s.shape)
        #print(s)
        sources.append(s)
        
        
        for k in range(9,16):
            x = self.vgg[k](x)
            
        s = self.L2Norm256(x)
        #print(s.shape)
        sources.append(s)
        
        
        for k in range(16,23):
            x = self.vgg[k](x)
            
        s = self.L2Norm512(x)
        #print(s.shape)
        sources.append(s)
        

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        
        #print(x.shape)
        sources.append(x)
        
        #print(x.shape)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                #print(x.shape)
                sources.append(x)
                
            

        
        #print('...................')


        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            #print(l)
            #print(l(x).shape)
            #print(c)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        
#        print(loc.shape)
#        print(conf.shape)
        
        #print('prior shape:',self.priors.shape)
        
        if phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 5),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 5),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file, remove_layers = False):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
#            self.load_state_dict(torch.load(base_file,
#                                 map_location=lambda storage, loc: storage),strict=False)
            
            checkpoint = torch.load(base_file)
            
            if remove_layers:
                del checkpoint['loc.0.weight']
                del checkpoint['loc.0.bias']
                del checkpoint['loc.1.weight']
                del checkpoint['loc.1.bias']
                del checkpoint['loc.2.weight']
                del checkpoint['loc.2.bias']
                del checkpoint['loc.3.weight']
                del checkpoint['loc.3.bias']
                del checkpoint['loc.4.weight']
                del checkpoint['loc.4.bias']
                del checkpoint['loc.5.weight']
                del checkpoint['loc.5.bias']
                del checkpoint['conf.0.weight']
                del checkpoint['conf.0.bias']
                del checkpoint['conf.1.weight']
                del checkpoint['conf.1.bias']
                del checkpoint['conf.2.weight']
                del checkpoint['conf.2.bias']
                del checkpoint['conf.3.weight']
                del checkpoint['conf.3.bias']
                del checkpoint['conf.4.weight']
                del checkpoint['conf.4.bias']
                del checkpoint['conf.5.weight']
                del checkpoint['conf.5.bias']
            
            
            self.load_state_dict(checkpoint, strict=False)
  
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [7, 14, 21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 5, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 5, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '64': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '64': [256, 'S', 512, 128, 'S', 256],
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '64': [6, 6, 6, 6, 6, 6], 
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(size=64, num_classes=21):
#    if phase != "test" and phase != "train":
#        print("ERROR: Phase: " + phase + " not recognized")
#        return
#    if size != 64:
#        print("ERROR: You specified size " + repr(size) + ". However, " +
#              "currently only SSD (size=64) is supported!")
#        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(size, base_, extras_, head_, num_classes)
