# coding = utf-8
import torch 
import torch.nn as nn
from config import args
import models.densenet
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
import models.satten_RegNet
import models.denseRegNet
import models.finalnet
import models.denseNet_reg
import models.denseNet_seg
import models.resnet_reg
import models.unet_reg
import models.wnetl
import models.cbw_models.deeplabv3_resnet.nets.deeplabv3_plus as deeplabv3_plus
from models.cbw_models.resnet.nets.resnet import resnet18, resnet34
from models.cbw_models.unet.nets.unet import Unet

matplotlib.use('TkAgg')


def predict():
    device = args.device
    
    nets = []
    denseNet121_reg = models.denseNet_reg.denseNet121_reg().to(device)
    denseNet121_reg.load_state_dict(torch.load(args.denseNet121_reg_no_mean), strict=False)
    nets.append(denseNet121_reg)
    denseNet121_seg = models.denseNet_seg.denseNet121_seg().to(device)
    denseNet121_seg.load_state_dict(torch.load(args.denseNet121_seg), strict=False)
    nets.append(denseNet121_seg)
    denseNet169_reg = models.denseNet_reg.denseNet169_reg().to(device)
    denseNet169_reg.load_state_dict(torch.load(args.denseNet169_reg_no_mean), strict=False)
    nets.append(denseNet169_reg)
    denseNet169_seg = models.denseNet_seg.denseNet169_seg().to(device)
    denseNet169_seg.load_state_dict(torch.load(args.denseNet169_seg), strict=False)
    nets.append(denseNet169_seg)
    finalNet_reg_mean = models.finalnet.finalnet().to(device)
    finalNet_reg_mean.load_state_dict(torch.load(args.finalNet_reg_mean), strict=False)
    nets.append(finalNet_reg_mean)
    stRegNet_mean = models.satten_RegNet.st_RegNet().to(device)
    stRegNet_mean.load_state_dict(torch.load(args.stRegNet_mean), strict=False)
    nets.append(stRegNet_mean)
    resnet18_reg = models.resnet_reg.resnet18_reg().to(device)
    resnet18_reg.load_state_dict(torch.load(args.resnet18_reg), strict=False)
    nets.append(resnet18_reg)
    resnet18_seg = resnet18(False, False).to(device)
    resnet18_seg.load_state_dict(torch.load(args.resnet18_seg), strict=False)
    nets.append(resnet18_seg)
    resnet34_reg = models.resnet_reg.resnet34_reg().to(device)
    resnet34_reg.load_state_dict(torch.load(args.resnet34_reg), strict=False)
    nets.append(resnet34_reg)
    resnet34_seg = resnet34(False, False).to(device)
    resnet34_seg.load_state_dict(torch.load(args.resnet34_seg), strict=False)
    nets.append(resnet34_seg)
    unet_reg = models.unet_reg.Unet().to(device)
    unet_reg.load_state_dict(torch.load(args.unet_reg), strict=False)
    nets.append(unet_reg)
    unet_seg = Unet(2).to(device)
    unet_seg.load_state_dict(torch.load(args.unet_seg), strict=False)
    nets.append(unet_seg)
    wzl_reg = models.wnetl.Net().to(device)
    wzl_reg.load_state_dict(torch.load(args.wzl_reg), strict=False)
    nets.append(wzl_reg)
    deeplabv3_resnet18_seg = deeplabv3_plus._segm_resnet(backbone_name='resnet18', num_classes=2, output_stride=16).to(device)
    deeplabv3_resnet18_seg.load_state_dict(torch.load(args.deeplabv3_resnet18_seg), strict=False)
    nets.append(deeplabv3_resnet18_seg)
    deeplabv3_resnet34_seg = deeplabv3_plus._segm_resnet(backbone_name='resnet34', num_classes=2, output_stride=16).to(device)
    deeplabv3_resnet34_seg.load_state_dict(torch.load(args.deeplabv3_resnet34_seg), strict=False)
    nets.append(deeplabv3_resnet34_seg)
    
    luts = []
    for path in os.listdir(args.final_lut):
        luts.append(np.load(args.final_lut + '/' + path))
    peaks = []
    for path in os.listdir(args.final_peak):
        peaks.append(np.load(args.final_peak + '/' + path))
    maps = []
    for path in os.listdir(args.final_map):
        maps.append(np.load(args.final_map + '/' + path))
    
    if not os.path.exists(args.result + '/MSELoss/'):
        os.makedirs(args.result + '/MSELoss/')
    if not os.path.exists(args.result + '/Dice/'):
        os.makedirs(args.result + '/Dice/')
    if not os.path.exists(args.result + '/L1Loss/'):
        os.makedirs(args.result + '/L1Loss/')
    if not os.path.exists(args.result + '/psnr/'):
        os.makedirs(args.result + '/psnr/')
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        denseNet121_reg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = denseNet121_reg(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        predict_peak = predict_peak[0,0,:]
        predict_peak.resize([256,2]) #[256,2]
        predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/denseNet121_reg', dice)
    np.save(args.result + '/MSELoss/denseNet121_reg', mse)
    np.save(args.result + '/L1Loss/denseNet121_reg', l1)
    np.save(args.result + '/psnr/denseNet121_reg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        denseNet121_seg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = denseNet121_seg(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        predict_peak = predict_peak[0,0,:]
        # predict_peak.resize([256,2]) #[256,2]
        # predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        # predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/denseNet121_seg', dice)
    np.save(args.result + '/MSELoss/denseNet121_seg', mse)
    np.save(args.result + '/L1Loss/denseNet121_seg', l1)
    np.save(args.result + '/psnr/denseNet121_seg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        denseNet169_reg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = denseNet169_reg(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        predict_peak = predict_peak[0,0,:]
        predict_peak.resize([256,2]) #[256,2]
        predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/denseNet169_reg', dice)
    np.save(args.result + '/MSELoss/denseNet169_reg', mse)
    np.save(args.result + '/L1Loss/denseNet169_reg', l1)
    np.save(args.result + '/psnr/denseNet169_reg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        denseNet169_seg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = denseNet169_seg(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        predict_peak = predict_peak[0,0,:]
        # predict_peak.resize([256,2]) #[256,2]
        # predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        # predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/denseNet169_seg', dice)
    np.save(args.result + '/MSELoss/denseNet169_seg', mse)
    np.save(args.result + '/L1Loss/denseNet169_seg', l1)
    np.save(args.result + '/psnr/denseNet169_seg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        finalNet_reg_mean.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = finalNet_reg_mean(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        predict_peak = predict_peak[0,0,:]
        predict_peak.resize([256,2]) #[256,2]
        predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/finalNet_reg_mean', dice)
    np.save(args.result + '/MSELoss/finalNet_reg_mean', mse)
    np.save(args.result + '/L1Loss/finalNet_reg_mean', l1)
    np.save(args.result + '/psnr/finalNet_reg_mean', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        stRegNet_mean.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = stRegNet_mean(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        predict_peak = predict_peak[0,0,:]
        predict_peak.resize([256,2]) #[256,2]
        predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/stRegNet_mean', dice)
    np.save(args.result + '/MSELoss/stRegNet_mean', mse)
    np.save(args.result + '/L1Loss/stRegNet_mean', l1)
    np.save(args.result + '/psnr/stRegNet_mean', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        resnet18_reg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
        #input = input.repeat(2, axis=0) # [2,3,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = resnet18_reg(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        # predict_peak = predict_peak[0,0,:]
        predict_peak.resize([256,2]) #[256,2]
        predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/resnet18_reg', dice)
    np.save(args.result + '/MSELoss/resnet18_reg', mse)
    np.save(args.result + '/L1Loss/resnet18_reg', l1)
    np.save(args.result + '/psnr/resnet18_reg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        resnet18_seg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
        #input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = resnet18_seg(input)[0]
        predict_peak = F.softmax(predict_peak.permute(1,2,0), dim=-1).cpu().detach().numpy()
        predict_peak = predict_peak.argmax(axis=-1)
        peak = np.round(peak).astype(int)
        # predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/resnet18_seg', dice)
    np.save(args.result + '/MSELoss/resnet18_seg', mse)
    np.save(args.result + '/L1Loss/resnet18_seg', l1)
    np.save(args.result + '/psnr/resnet18_seg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        resnet34_reg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
        # input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = resnet34_reg(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        #predict_peak = predict_peak[0,0,:]
        predict_peak.resize([256,2]) #[256,2]
        predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/resnet34_reg', dice)
    np.save(args.result + '/MSELoss/resnet34_reg', mse)
    np.save(args.result + '/L1Loss/resnet34_reg', l1)
    np.save(args.result + '/psnr/resnet34_reg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        resnet34_seg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
        #input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = resnet34_seg(input)[0]
        predict_peak = F.softmax(predict_peak.permute(1,2,0), dim=-1).cpu().detach().numpy()
        predict_peak = predict_peak.argmax(axis=-1)
        peak = np.round(peak).astype(int)
        # predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/resnet34_seg', dice)
    np.save(args.result + '/MSELoss/resnet34_seg', mse)
    np.save(args.result + '/L1Loss/resnet34_seg', l1)
    np.save(args.result + '/psnr/resnet34_seg', psnr)
    
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        unet_reg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = unet_reg(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        #predict_peak = predict_peak[0,0,:]
        predict_peak.resize([256,2]) #[256,2]
        predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/unet_reg', dice)
    np.save(args.result + '/MSELoss/unet_reg', mse)
    np.save(args.result + '/L1Loss/unet_reg', l1)
    np.save(args.result + '/psnr/unet_reg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        unet_seg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
        #input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = unet_seg(input)[0]
        predict_peak = F.softmax(predict_peak.permute(1,2,0), dim=-1).cpu().detach().numpy()
        predict_peak = predict_peak.argmax(axis=-1)
        peak = np.round(peak).astype(int)
        # predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/unet_seg', dice)
    np.save(args.result + '/MSELoss/unet_seg', mse)
    np.save(args.result + '/L1Loss/unet_seg', l1)
    np.save(args.result + '/psnr/unet_seg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        wzl_reg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = wzl_reg(input)
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
        #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
        #print('loss:',loss.item()  )
        predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
        #predict_peak = predict_peak[0,0,:]
        predict_peak.resize([256,2]) #[256,2]
        predict_peak = np.round(predict_peak).astype(int)
        peak = np.round(peak).astype(int)
        predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/wzl_reg', dice)
    np.save(args.result + '/MSELoss/wzl_reg', mse)
    np.save(args.result + '/L1Loss/wzl_reg', l1)
    np.save(args.result + '/psnr/wzl_reg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        deeplabv3_resnet18_seg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = deeplabv3_resnet18_seg(input)[0]
        predict_peak = F.softmax(predict_peak.permute(1,2,0), dim=-1).cpu().detach().numpy()
        predict_peak = predict_peak.argmax(axis=-1)
        peak = np.round(peak).astype(int)
        # predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/deeplabv3_resnet18_seg', dice)
    np.save(args.result + '/MSELoss/deeplabv3_resnet18_seg', mse)
    np.save(args.result + '/L1Loss/deeplabv3_resnet18_seg', l1)
    np.save(args.result + '/psnr/deeplabv3_resnet18_seg', psnr)
    
    
    dice = []
    mse = []
    l1 = []
    psnr = []
    for idx in range(len(luts)):
        deeplabv3_resnet34_seg.train()
        lut = luts[idx]
        tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
        input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
        input = input.repeat(2, axis=0) # [2,1,256,256]
        peak = peaks[idx] # [256,2] 
        tmp_peak = peak
        input = torch.from_numpy(input)
        input = input.to(torch.float32).to(device)
        predict_peak = deeplabv3_resnet34_seg(input)[0]
        predict_peak = F.softmax(predict_peak.permute(1,2,0), dim=-1).cpu().detach().numpy()
        predict_peak = predict_peak.argmax(axis=-1)
        peak = np.round(peak).astype(int)
        # predict_map = heatmap2.heatmap(predict_peak)
        true_map = heatmap2.heatmap(peak)
        # tensor_predict_map = torch.from_numpy(predict_map)
        # tensor_true_map = torch.from_numpy(true_map)
        dice.append(2*np.sum(true_map*predict_map)/(np.sum(true_map)+np.sum(predict_map)))
        mse.append(np.mean((true_map-predict_map)**2))
        l1.append(np.mean(np.abs(predict_map - true_map)))
        psnr.append(exp_loss.psnr(predict_map, true_map))
    dice = np.array(dice)
    mse = np.array(mse)
    l1 = np.array(l1)
    psnr = np.array(psnr)
    np.save(args.result + '/Dice/deeplabv3_resnet34_seg', dice)
    np.save(args.result + '/MSELoss/deeplabv3_resnet34_seg', mse)
    np.save(args.result + '/L1Loss/deeplabv3_resnet34_seg', l1)
    np.save(args.result + '/psnr/deeplabv3_resnet34_seg', psnr)
    
    return 0


if __name__ == '__main__':
    predict()
        