## SsdNet
Created by Peizhi Shi at University of Huddersfield. If you have any questions about the code, please feel free to contact me (p.shi@hud.ac.uk).

### Introduction

The SsdNet is a novel learning-based intersecting feature recognition and localisation method. At the time of its release, the SsdNet achieves the state-of-the-art intersecting feature recognition and localisation performance. This is an open access peer-reviewed paper, which will be available online soon. This repository provides the source codes of the SsdNet. 

### Experimental configuration

1. CUDA (10.0.130)
2. cupy-cuda100 (6.2.0)
3. numpy (1.17.4)
4. Pillow (6.2.1)
5. python (3.6.8)
6. pyvista (0.22.4)
7. scikit-image (0.16.2)
8. scipy (1.3.3)
9. torch (1.1.0)
10. torchvision (0.3.0)

All the experiments mentioned in our paper are conducted on Ubuntu 18.04 under the above experimental configurations. If you run the code on the Windows or under different configurations, slightly different results might be achieved.


### Training (optional)

1. Get the SsdNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/SsdNet.git`.
2. Download the FeatureNet [dataset](https://github.com/madlabub/Machining-feature-dataset), and convert them into voxel models via [binvox](https://www.patrickmin.com/binvox/). The filename format is `label_index.binvox`. Then put all the `*.binvox` files in a same folder `??`. This folder is supposed to contain 24,000 `*.binvox` files. Please note there are some unlabelled/mislabelled files in category 8 (rectangular_blind_slot) and 12 (triangular_blind_step). Before moving these files in the same folder, please correct these filenames.
3. Run `python ??.py` and `python ??.py` to create training and validation sets respectively. Please note that training set creation process is time-consuming.
4. Run `python ??.py` to train the neural network. 


### Intersecting feature recognition and localisation

1. Get the SsdNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/SsdNet.git`.
2. Download the benchmark multi-feature [dataset](https://1drv.ms/u/s!At5UoWCCWHUKafomIKnOJnsl0Dg?e=lbK8iw), and put them in the folder `data/`.
3. Download the pretrained SsdNet model, and put it into the folder `??`.  
4. Run `python multi_test.py` to test the performances of the SsdNet for intersecting feature recognition and localisation.


