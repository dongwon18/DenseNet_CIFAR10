# DenseNet_CIFAR10
build, train DenseNet with CIFAR 10 dataset to get accuracy over than 93%

## Parameters
- batch size: 128
- epoch: 200
- learning rate: 0.1
- dropout: 0.2
- growth rate: 12
- reduction: 0.5

# Dataset

## Download Data
1. load CIFAR10 dataset from torchvision.datasets
2. change dataset to tenser and normalize with meand and std  
[reference](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
    - image: 3 x 32 x32
    - total 10 classes
3. download train and dataset

## Shuffle and Split train, validation data
1. by using train_test_split, shuffle the dataset and split data
    - validation size = 10% of train data
    - split data to have similar ratio of targets using stratify option  
2. set samplers for each dataset using SubsetRandomSampler 

## Set DataLoader
1. using the samplers, set train, val loader
2. since test dataset is not shuffled and splited, no need to use a sampler

# Model
## DenseNet
- make DenseNet using Bottleneck and transition block
- refering the paper, 
    - set θ as 0.5, growth rate as 12
    - set 1st 3 x 3 Conv layer's output channels as twice the growth rate
    - for Dense121, 
        - no. of Dense block: 6 12 24 16 

> conv → dense1 → transition1 → dense2 → transition2 → dense3 → transition3 → dense4 → classification layer

# Train
## Loss function and optimizer
- refering the paper
    - use SGD
    - learning rate: 0.1, weight decay: 10^(-4)

## Train
- set epoch to 200
    - although the paper set epoch 300 for CIFAR10, by experiment, because of overfitting, set to 200 
    - train accuracy becomes 1 after 185 epoch and loss of validation set doesn't decrease
- not using early stopping to train the model enough

# Result
- test accuracy: 0.93
- detailed results are included in the ipynb file.

# 배운 점
- bottleneck block을 이용하여 feature map을 줄이면서(1x1 conv layer) 성능은 유지할 수 있다.
- DenseNet에서는 compression(2x2 average pooling)을 위해 transition block을 이용한다.
- droprate 또한 추가하였지만 원 논문에서는 bottleneck block으로 충분하다고 명시하였다.
- block을 이용하여 코드를 작성함으로써 모듈화 및 가독성을 높일 수 있다. 블록 단위로 코드를 작성하여 사용하면 논문에서 설명된 architecture를 그대로 설계하기 편하다.

# 한계점
- VGG16을 먼저 사용하였지만 목표 정확도를 도달하지 못했다.
- 당시 SOTA였던 ResNet을 이기기 위해 DenseNet을 만든 만큼 원 논문에는 ResNet과 비교하여 서술하는 부분이 많았다. 그러나 ResNet에 대한 이해가 부족했기 때문에 ResNet과 대비되는 DenseNet의 특징을 완벽히 이해하기 어려웠다.

