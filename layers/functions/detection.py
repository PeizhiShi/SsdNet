import torch
from torch.autograd import Function
from ..box_utils import decode#, nms, nms2
from data import cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        #self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
            
            output: shape [batch, top_k, 6] # score, k, label+boxes
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.top_k, 7)
        
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            
            conf_scores, idx = conf_scores.max(0)
            
            mask = idx.gt(0)
            labels = idx[mask]
            conf_scores = conf_scores[mask]
            decoded_boxes = decoded_boxes[mask]
            
            if labels.shape[0] == 0:
                continue
            
#            print(labels.shape)
#            print(decoded_boxes.shape)
            
            #keep, count = nms(decoded_boxes, conf_scores, 0.6)
#            print(decoded_boxes.shape)
#            print(conf_scores.shape)
            #keep, count = nms2(decoded_boxes, conf_scores, 0.5)
#            print(decoded_boxes.shape)
#            print(conf_scores.shape)
            
            #tmp = torch.cat((conf_scores[keep].unsqueeze(1),labels[keep].float().unsqueeze(1),decoded_boxes[keep,:]), 1)
            
#            print(tmp.shape)
#            print(count)
            
            #output[i, :count] = torch.cat((conf_scores[keep].unsqueeze(1),labels[keep].float().unsqueeze(1),decoded_boxes[keep,:]), 1)
            
            #keep, count = nms(decoded_boxes, conf_scores, 0.6)
            count = labels.shape[0]
            output[i, :count] = torch.cat((conf_scores.unsqueeze(1),labels.float().unsqueeze(1),decoded_boxes), 1)
            
                
         
        return output























