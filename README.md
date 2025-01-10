# Prerequisites

The following prerequisites are required to run the scripts:

- [PyTorch and torchvision](https://pytorch.org)

- Dataset: Please download the dataset
  from [Google Cloud](https://drive.google.com/drive/folders/1elbJ6aHxtKGutzOxlXA7QwV45EZEqNxq?usp=drive_link) and
  modify the corresponding dataset path in method folder.

- Install packages

```shell
pip install -r requirements.txt
```

# Start

Please excute following commands in order.

## Computing Centers

```shell
python method/compute_center.py
--dataset CIFAR-FS/FC100
--backbone resnet12/resnet50/swin
```

## Training

```shell
python method/train.py
--max-epoch 50
--mode clip
--semantic-size 512
--text_type gpt
--shot 1
--step-size 40
--test-batch 600
--batch-size 128
--num-workers 8
--dataset CIFAR-FS/FC100
--lr 1e-4
--backbone resnet12/resnet50/swin
```

## Evaluation

```shell
python method/test.py
--dataset CIFAR-FS/FC100
--shot 1
--backbone resnet12/resnet50/swin
```