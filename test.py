import pyvista as pv
import numpy as np
import csv
import utils.binvox_rw
from utils.augmentations import SSDAugmentation
from data import *
import torch
import torch.backends.cudnn as cudnn
from ssd import build_ssd
from torch.autograd import Variable
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pow
import matplotlib.patches as mpatches


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
def get_gt_label(filename):
    
    
    retarr = np.zeros((0,7))
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            items = row[0].split(',')
            retarr = np.insert(retarr,0,np.asarray(items),0)
    

    retarr[:,0:6] = retarr[:,0:6] * 1000
    
    return retarr




def load_pretrained_model():
    ssd_net = build_ssd(cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True
    
    #ssd_net.load_weights('weights/512-exp2-notlda/VOC.pth')
    ssd_net.load_weights('weights/VOC.pth')
    
    #print(net)
    
    return net.cuda()


def tensor_to_float(val):
    
    if val < 0:
        val = 0
    
    if val > 1:
        val = 1
    
    return float(val)

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

def soft_nms_pytorch(boxes, box_scores, sigma=0.5):

    dets = boxes[:,0:6].copy()*1000
    
    N = dets.shape[0]
    
    indexes = torch.arange(0, N, dtype=torch.double).view(N, 1).cpu()
    dets = torch.from_numpy(dets).double().cpu()
    scores = torch.from_numpy(box_scores.copy()).double().cpu()
        
    dets = torch.cat((dets, indexes), dim=1).cpu()
    

    z1 = dets[:, 0]
    y1 = dets[:, 1]
    x1 = dets[:, 2]
    z2 = dets[:, 3]
    y2 = dets[:, 4]
    x2 = dets[:, 5]
    #scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)* (z2 - z1 + 1)

    for i in range(N):
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()


        # IoU calculate
        zz1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        yy1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        zz2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 4].to("cpu").numpy(), dets[pos:, 4].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 5].to("cpu").numpy(), dets[pos:, 5].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        l = np.maximum(0.0, zz2 - zz1 + 1)
        inter = torch.tensor(w * h * l).cpu()
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter)).cpu()

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma).cpu()
        
        
        
        scores[pos:] = weight * scores[pos:]


    #print(scores)
    max_margin = 0
    thresh = 0
    for i in range(scores.shape[0]-1):
        if scores[i] - scores[i + 1] > max_margin:
            max_margin = scores[i] - scores[i + 1]
            thresh = (scores[i] + scores[i+1])/2
        
    #thresh = (scores[1] + scores[2])/2
       
    
    
    keep = dets[:, 6][scores > thresh].int()
    
    #print(keep.shape)
    
    

    return keep.to("cpu").numpy()

def get_predicted_label(filename, net):
    with open(filename+'.binvox', 'rb') as f:
        model = utils.binvox_rw.read_as_3d_array(f).data
    
    transform=SSDAugmentation(cfg['min_dim'],MEANS,phase='test')
    
    images = []
    
    for rot in range(6):
        img, _ = create_img(model, rot)
                
        img, _, _ = transform(img, 0, 0)
        
        images.append(img)
        
    images = torch.tensor(images).permute(0, 3, 1, 2).float()
    
    
    images = Variable(images.cuda())
    #images = images.cuda()
                    
    out = net(images, 'test')
    out.cuda()
    
    cur_boxes = np.zeros((0,8))
    
    
    for i in range(6):

        
        for j in range(out.shape[1]):
            label = out[i,j,1].detach().cpu()
            
                    
            if label == 0:
                continue
            
            
            
                    
            score = out[i,j,0].detach().cpu()
                    
            x1 = tensor_to_float(out[i,j,2])
            y1 = tensor_to_float(out[i,j,3])
            x2 = tensor_to_float(out[i,j,4])
            y2 = tensor_to_float(out[i,j,5])
            z1 = 0.0
            z2 = tensor_to_float(out[i,j,6])
            
            if x1 >= x2 or y1 >= y2 or z2 <= 0:
                continue
            
          
            a = z1
            b = y1
            c = x1
            d = z2
            e = y2
            f = x2
        
            if i == 1:
                a = 1-z2
                b = 1-y2
                c = x1
                d = 1-z1
                e = 1-y1
                f = x2
            elif i == 2:
                a = y1
                b = 1-z2
                c = x1
                d = y2
                e = 1-z1
                f = x2
            elif i == 3:
                a = 1-y2
                b = z1
                c = x1
                d = 1-y1
                e = z2
                f = x2
            elif i == 4:
                a = 1-x2
                b = y1
                c = z1
                d = 1-x1
                e = y2
                f = z2
            elif i == 5:
                a = x1
                b = y1
                c = 1-z2
                d = x2
                e = y2
                f = 1-z1
            
            cur_boxes = np.append(cur_boxes,np.array([a,b,c,d,e,f,label-1,score]).reshape(1,8),axis=0)
    
    keepidx = soft_nms_pytorch(cur_boxes[:,:7], cur_boxes[:,-1])
    cur_boxes = cur_boxes[keepidx,:]
    
    
    cur_boxes[:,0:6] = 10000*cur_boxes[:,0:6]
    
    
    return cur_boxes

def get_lvec(labels):
    results = np.zeros(24)
    
    
    for i in labels:
        results[int(i)] += 1
    
    return results.astype(int)


    
def eval_metric(pre,trul,tp):
    precision = tp/pre
    
    recall = tp/trul
    
    return precision, recall


def cal_detection_performance(pred_boxes_labels, gt_boxes_labels):
    
    gt = get_lvec(gt_boxes_labels[:,6])
    pred = get_lvec(pred_boxes_labels[:,6])
    tp = np.minimum(gt,pred)
    
    return pred, gt, tp

def cal_localization_performance(pred_boxes_labels, gt_boxes_labels):
    gt = get_lvec(gt_boxes_labels[:,6])
    pred = get_lvec(pred_boxes_labels[:,6])
    tp = np.zeros(24).astype(int)
    threshold = 0.5
    #
    gt_boxes_labels = gt_boxes_labels[gt_boxes_labels[:,6].argsort()]
    
    pred_boxes_labels = pred_boxes_labels[pred_boxes_labels[:,7].argsort()[::-1]]
    pred_boxes_labels = pred_boxes_labels[pred_boxes_labels[:,6].argsort(kind='stable')]
    
    #print(gt_boxes_labels[:,-1])
    #print(pred_boxes_labels[:,-2:])
    
    # calculate the area of prediction and gt boxes
    pred_z1 = pred_boxes_labels[:, 0]
    pred_y1 = pred_boxes_labels[:, 1]
    pred_x1 = pred_boxes_labels[:, 2]
    pred_z2 = pred_boxes_labels[:, 3]
    pred_y2 = pred_boxes_labels[:, 4]
    pred_x2 = pred_boxes_labels[:, 5]
    pred_areas = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)* (pred_z2 - pred_z1 + 1)
    
    
    gt_z1 = gt_boxes_labels[:, 0]
    gt_y1 = gt_boxes_labels[:, 1]
    gt_x1 = gt_boxes_labels[:, 2]
    gt_z2 = gt_boxes_labels[:, 3]
    gt_y2 = gt_boxes_labels[:, 4]
    gt_x2 = gt_boxes_labels[:, 5]
    gt_areas = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)* (gt_z2 - gt_z1 + 1)
    
    found_gt = np.zeros(gt_boxes_labels.shape[0])
    
    #print(pred_areas)
    #for each detection
    for i in range(pred_boxes_labels.shape[0]):
        
        pred_label = pred_boxes_labels[i,6].astype(int)
        found = -1
        max_iou = 0
        
        #among the ground-truths, choose one that belongs to the same class and has the highest IoU with the detection        
        for j in range(gt_boxes_labels.shape[0]):
            
            gt_label = gt_boxes_labels[j,6].astype(int)
            
            if pred_label != gt_label or found_gt[j] == 1:
                continue
            
            zz1 = np.maximum(pred_z1[i], gt_z1[j])
            yy1 = np.maximum(pred_y1[i], gt_y1[j])
            xx1 = np.maximum(pred_x1[i], gt_x1[j])
            zz2 = np.minimum(pred_z2[i], gt_z2[j])
            yy2 = np.minimum(pred_y2[i], gt_y2[j])
            xx2 = np.minimum(pred_x2[i], gt_x2[j])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            l = np.maximum(0.0, zz2 - zz1 + 1)
            inter = w * h * l
            iou = inter/(pred_areas[i] + gt_areas[j] - inter)
            
            if iou > max_iou:
                found = j
                max_iou = iou
        
        #if no ground-truth can be chosen or IoU < threshold (e.g., 0.5):
            #the detection is a false positive
        #else:
            #the detection is a true positive
          
        
        if max_iou >= threshold and found >= 0:
            found_gt[found] = 1
            tp[pred_label] += 1
    
    return pred, gt, tp


def test_ssdnet(sidx, metric = cal_detection_performance):
    
    net = load_pretrained_model()
    
    predictions = np.zeros(24)
    truelabels = np.zeros(24)
    truepositives = np.zeros(24)
    
    with torch.no_grad():
        with open(os.devnull, 'w') as devnull:
            for filename in Path('data/MulSet/set' + str(sidx)+ '/').glob('*.STL'):
                filename = str(filename).replace('.STL','')
                
                pred, trul, tp = metric(get_predicted_label(filename, net), get_gt_label(filename+'.csv'))
                
                predictions += pred
                truelabels += trul
                truepositives += tp
                
                print(filename)
                    
                print(trul)
                print(pred)
                print(tp)
                
              
    
    precision, recall = eval_metric(predictions, truelabels, truepositives)
    print('Precision scores')
    precision = precision.mean()
    print(precision)
    print('Recall scores')
    recall = recall.mean()
    print(recall.mean())
    print('F scores')
    print((2*recall*precision)/(recall+precision))



import time
start_time = time.time()

data_group = 10

test_ssdnet(data_group,cal_detection_performance) #feature recognition
#test_ssdnet(data_group,cal_localization_performance) #feature localisation

print("--- %s seconds ---" % (time.time() - start_time))











