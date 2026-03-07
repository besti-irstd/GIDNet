# GIDNet: Infrared Small Target Detection Network Based on  Gradient-Intensity Decoupled
This repository contains the official PyTorch implementation of GIDNet, a deep learning model designed for Infrared Small Target Detection (IRSTD). GIDNet leverages spatial domain enhancements, utilizing generalized central difference convolution (GISC), multi-scale dilated convolution (MSDC) and shallow feature projection (SFP) strategy to effectively capture the structural edge information of small infrared targets. 
# Prerequisites
requirements.txt
# Datasets
Download the datasets and put them to './datasets': 
[IRSTD-1k](https://github.com/RuiZhang97/ISNet),
[NUAA-SIRST](https://github.com/YimianDai/sirst), 
[NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Or test according to the train/val split ratio provided in the datasets directory.
# Prerequisite
Trained and tested on PyTorch 2.0.0 Python, 3.8(ubuntu20.04), CUDA  11.8 torch 2.0.0+cu118, torchaudio 2.0.1+cu118, torchvision 0.15.1+cu118, and 1×NVIDIA 4090.
# Training
The training command is very simple like this:
python main --dataset-dir --batch-size --epochs --mode 'train'
For example:
python main.py --dataset-dir './dataset/IRSTD-1k' --batch-size 4 --epochs 800 --mode 'train'
# Testing
You can test the model with the following command:
python main.py --dataset-dir './dataset/IRSTD-1k' --batch-size 1 --mode 'test' --weight-path './weight/irstd.pkl'
# Quantative Results
| Dataset    | mIoU (x10(-2)) | Pd (x10(-2)) | Fa (x10(-6)) |                                               Weights                                               |
| ---------- | :------------: | :----------: | :----------: | :-------------------------------------------------------------------------------------------------: |
| IRSTD-1k   |     69.01      |    93.54     |     10.32     |  [IRSTD-1k]()  |
| NUAA-SIRST |     78.16      |     100      |     6.74     | [NUAA-SIRST]() |
| NUDT-SIRST |     83.51      |    98.41     |     2.80     | [NUDT-SIRST]() |
# Loss
* GIDNet employs the SLS loss and further improves the network architecture based on [MSHNet](https://github.com/Lliu666/MSHNet). Thanks to Qiankun Liu.
# Citation
Please kindly cite the papers if this code is useful and helpful for your research.



