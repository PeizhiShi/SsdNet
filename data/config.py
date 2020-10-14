# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
cfg = {
    'num_classes': 25,
    #'lr_steps': (80000, 100000, 120000),
    #'max_iter': 120000,
    'feature_maps': [32, 16, 8, 4, 2, 1],
    'min_dim': 64,
    'steps': [2, 4, 8, 16, 32, 64],
    'min_sizes': [6, 12, 23, 34, 44, 56],
    'max_sizes': [12, 23, 34, 44, 56, 67],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


