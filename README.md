# Distilled-Models
This repository contains knowledge-distillated models implemented in PyTorch and trained using Pytorch-Lightning.

- [Install](README.md#install)
- [Available Methods](README.md#methods)
- [Bug/Improvement](README.md#bug-or-improvement)

## Install
1. Clone the repository:
    > git clone https://github.com/blurry-mood/Distilled-Models
2. Install requirements:
    > cd Distilled-Models  
    > pip install -r requirements.txt
3. Setup CIFAR-100:
    > cd datasets/cifar-100  
    > ./setup.sh
    
## Train:
To reproduce the results of an experiment mentioned in the table below, execute the command existing at the top of the corresponding python script.
    
## Experiments:
| Paper | Dataset | Teacher | Student | Accuracy@Top-1 | Accuracy@Top-5 |
| ----- | ------- | ------- | ------- | -------------- | -------------- |
| Baseline | CIFAR-100 | Resnet32 | - | 69.59 | 91.39 |
| [Deep Mutual Learning](https://arxiv.org/abs/1706.00384) (Redo) | CIFAR-100 | Resnet32 | Resnet32 | 69.31 | 91.86 |
| [Online Knowledge Distillation with Diverse Peers](https://arxiv.org/abs/1912.00350) | CIFAR-100 | Resnet32 | Resnet32 | N/A | N/A |


## Bug or improvement:
This repository is still under developement, thus if you encounter a bug or would like to request a feature, please feel free to open an issue [here](https://github.com/blurry-mood/Distilled-Models/issues).