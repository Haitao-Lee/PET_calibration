# coding = utf-8
import argparse
import os
import torch.nn as nn
import torch
# from initialize import *


dir_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
parser = argparse.ArgumentParser()

parser.add_argument('--final_models', default=dir_path + '/trained_model')
parser.add_argument('--DenseNet121_mean', default=dir_path + '/trained_model/denseNet121_mean.pth')
parser.add_argument('--DenseNet121', default=dir_path + '/trained_model/denseNet121.pth')
parser.add_argument('--finalNet_mean', default=dir_path + '/trained_model/finalNet_mean.pth')
parser.add_argument('--stRegNet_mean', default=dir_path + '/trained_model/stRegNet_mean.pth')
parser.add_argument('--ResNet18', default=dir_path + '/trained_model/resnet18.pth')
parser.add_argument('--ResNet34', default=dir_path + '/trained_model/resnet34.pth')
parser.add_argument('--UNet', default=dir_path + '/trained_model/UNet.pth')
parser.add_argument('--UNet_mean', default=dir_path + '/trained_model/UNet_mean.pth')
parser.add_argument('--UNet3plus', default=dir_path + '/trained_model/UNet3plus.pth')
parser.add_argument('--UNet3plus_mean', default=dir_path + '/trained_model/UNet3plus_mean.pth')
parser.add_argument('--Laddernet', default=dir_path + '/trained_model/Laddernet.pth')
parser.add_argument('--Laddernet_mean', default=dir_path + '/trained_model/Laddernet_mean.pth')
parser.add_argument('--Convnext', default=dir_path + '/trained_model/Convnext.pth')
parser.add_argument('--Convnext_mean', default=dir_path + '/trained_model/Convnext_mean.pth')

# hardware setting
parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),)
args = parser.parse_args()


