## SsdNet

Created by Peizhi Shi at University of Huddersfield. 

### Introduction

The SsdNet is a novel learning-based intersecting feature recognition and localisation method. At the time of its release, the SsdNet achieves the state-of-the-art intersecting feature recognition and localisation performance. This repository provides the source codes of the SsdNet. 

If this project is useful to you, please consider citing our paper:

    @ARTICLE{shi2020intersecting,
	    author={Shi, Peizhi and Qi, Qunfen and Qin, Yuchu and Scott, Paul and Jiang, Xiangqian},
	    journal={IEEE Transactions on Industrial Informatics}, 
	    title={Intersecting machining feature localization and recognition via single shot multibox detector}, 
	    year={2021},
	    volume={17},
	    number={5},
	    pages={3292--3302}
    }

This is a peer-reviewed paper, which is available [online](https://doi.org/10.1109/TII.2020.3030620). 

### Experimental configuration

1. CUDA (10.0.130)
2. cupy-cuda100 (6.2.0)
3. numpy (1.17.4)
4. **python (3.6.8)**
5. scikit-image (0.16.2)
6. scipy (1.3.3)
7. **torch (1.1.0)**
8. torchvision (0.3.0)
9. **matplotlib (3.1.2)**

All the experiments mentioned in our paper are conducted on Ubuntu 18.04 under the above experimental configurations. An Intel i9-9900X PC with a 128 GB memory and NVIDIA RTX 2080ti GPU is employed in this paper. If you run the code on the Windows or under different configurations, slightly different results might be achieved.


### Training (optional)

1. Get the SsdNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/SsdNet.git`.
2. Create the following folders: `data/TrSet`, `data/ValSet`, `data/FNSet`, `weights` and `weights/base`. 
3. Download the single feature [dataset](https://github.com/madlabub/Machining-feature-dataset) (originally from the [FeatureNet](https://doi.org/10.1016/j.cad.2018.03.006)), and convert unzipped STL models into voxel models via [binvox](https://www.patrickmin.com/binvox/). The filename format is `label_index.binvox`. Then put all the `*.binvox` files in a same folder `data/FNSet`. This folder is supposed to contain 24,000 `*.binvox` files. Please note there are some unlabelled/mislabelled files in category 8 (rectangular_blind_slot) and 12 (triangular_blind_step). Before moving these files in the same folder, please correct these filenames.
4. Run `python create_tr_set.py` and `python create_val_set.py` to create training and validation sets respectively. Please note that training set creation process is time-consuming.
5. Download the pretrained SSD300 [basenet](https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth), and put it in the folder `weights/base`. This pretrained model is utilised for transfer learning.  
6. Run `python train.py` to train the neural network. 


### Intersecting feature recognition and localisation

1. Get the SsdNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/SsdNet.git`.
2. Create the folder named `data/MulSet`.
3. Download the benchmark multi-feature [dataset](https://1drv.ms/u/s!At5UoWCCWHUKafomIKnOJnsl0Dg?e=lbK8iw), and put them in the folder `data/MulSet`.
4. Download our pretrained SsdNet [model](https://1drv.ms/u/s!At5UoWCCWHUKedwHDIt8BLUTw5E?e=SbR0Xh), and then put the unzipped file into the folder `weights`. This model allows for achieving the experimental results reported in our IEEE TII paper. This step could be skipped if you have trained the neural network by yourself.  
5. Run `python test.py` to test the performances of the SsdNet for intersecting feature recognition and localisation.

If you have any questions about the code, please feel free to contact me (p.shi@hud.ac.uk).
