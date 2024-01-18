# coding = utf-8
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import numpy as np
from config import args
import torch 
import models.satten_RegNet
import models.denseRegNet
import models.finalnet
import models.denseNet_reg
import models.denseNet_seg
import models.resnet_reg
import models.unet_reg
import models.UNet_3plus
import models.LadderNet
import models.ConvNext
import models.wnetl
import models.cbw_models.deeplabv3_resnet.nets.deeplabv3_plus as deeplabv3_plus
from models.cbw_models.resnet.nets.resnet import resnet18, resnet34
from models.cbw_models.unet.nets.unet import Unet
import isolate


class slot_functions():
    def __init__(self, ui):
        super(slot_functions, self).__init__()
        self.ui = ui

    def import_file_slot(self): 
        tmp_file_name, _ = QFileDialog.getOpenFileName(self.ui, "select a file", "C:/", "numpy Files (*.npy);")
        self.ui.working_dir = os.path.dirname(tmp_file_name)
        self.ui.filemodel.setRootPath(self.ui.working_dir)
        self.ui.working_tree.setRootIndex(self.ui.filemodel.index(self.ui.working_dir))
        messages = '--import file:' + tmp_file_name
        self.info_slot(messages)
        if tmp_file_name.endswith('.npy'):
            lut = np.load(tmp_file_name)
            if lut.shape[0] != 256 or lut.shape[1] != 256:
                # QMessageBox
                return
            self.ui.lut = lut
            self.ui.peak = None
            self.show(lut=lut)
            self.ui.file_name = tmp_file_name
            self.ui.vis = 1
            return
                                   
    def import_folder_slot(self):
        self.ui.working_dir = QFileDialog.getExistingDirectory(None, 'select a folder', os.getcwd())
        self.ui.filemodel.setRootPath(self.ui.working_dir)
        self.ui.working_tree.setRootIndex(self.ui.filemodel.index(self.ui.working_dir))
        messages = '--import folder:' + self.ui.working_dir
        self.info_slot(messages)
    
    def save_slot(self):
        print("save slot function")
        
    def quit_slot(self):
        app = QApplication.instance()
        app.quit()

    def predict_slot(self):
        if self.ui.lut is None:
            self.info_slot('--Lacking input lut!')
            return
        if self.ui.net_name == 'ResNet18':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak.resize([256, 2])
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'ResNet34':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak.resize([256, 2])
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'DenseNet121':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'DenseNet121_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'UNet':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'UNet_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'UNet3+':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'UNet3+_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Laddernet':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Laddernet_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Convnext':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Convnext_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'finalNet_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,3,256,256]
            input = input.repeat(2, axis=0) # [2,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'stRegNet_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Groundtruth':
            predict_peak = self.ui.gt
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2


    
    def visual_slot(self):
        index = 1
        if self.ui.lut is not None:
            index = index + 1
            if self.ui.peak is not None:
                index = index + 1
                if self.ui.binary_border is not None:
                    index = index + 1
                if self.ui.cluster_lut is not None:
                    index = index + 1
        self.ui.vis = (self.ui.vis+1)%index
        if self.ui.vis == 0:
            self.ui.canvas.axes.clear()
            self.show(None)
        elif self.ui.vis == 1:
            self.show(lut=self.ui.lut, peaks=None, boudnary=None, clusters=None)
        elif self.ui.vis == 2:
            self.show(lut=self.ui.lut, peaks=self.ui.peak, boudnary=None, clusters=None)
        elif self.ui.vis == 3:
            self.show(lut=self.ui.lut, peaks=self.ui.peak, boudnary=self.ui.binary_border, clusters=None)
        elif self.ui.vis == 4:
            self.show(lut=self.ui.lut, peaks=self.ui.peak, boudnary=self.ui.binary_border, clusters=self.ui.cluster_lut)
        
            
    def isolate_slot(self):
        if self.ui.peak is None or self.ui.lut is None:
            self.info_slot('--Lacking input lut!')
            return
        self.ui.cluster_lut, self.ui.binary_border, = isolate.clusterAndBoundary(self.ui.lut, self.ui.peak)
        self.show(lut=self.ui.lut, peaks=self.ui.peak, boudnary=self.ui.binary_border, clusters=self.ui.cluster_lut)
        self.ui.vis = 4

    
    def info_slot(self, messages):
        self.ui.plainTextEdit.appendPlainText(messages + '\n')
        
    def get_file(self, Qmodelidx):
        # print(self.ui.model.filePath(Qmodelidx))  # 输出文件的地址。
        # print(self.ui.model.fileName(Qmodelidx))  # 输出文件名
        tmp_file_name = self.ui.filemodel.filePath(Qmodelidx)
        messages = '--Current direction:' + tmp_file_name
        self.info_slot(messages)
        if tmp_file_name.endswith('.npy'):
            lut = np.load(tmp_file_name)
            # print(lut.shape)
            if lut.shape[0] == 256 and lut.shape[1] == 256:
                self.ui.file_name = tmp_file_name
                self.ui.lut = lut
                self.show(lut=lut)
                self.ui.vis = 1
            if lut.shape[0] == 256 and lut.shape[1] == 2:
                self.ui.file_name = tmp_file_name
                self.ui.gt = lut
    
    def show(self, width=None, height=None, lut=None, peaks=None, boudnary=None, clusters=None):
        rectItem = self.ui.main_widget.scene().itemsBoundingRect()
        rectView = self.ui.main_widget.rect()
        ratioView = rectView.height() / rectView.width()
        ratioItem = rectItem.height() / rectItem.width()
        if ratioView > ratioItem:
            rectItem.moveTop(rectItem.width()*ratioView - rectItem.height())
            rectItem.setHeight(rectItem.width()*ratioView)
            rectItem.setWidth(rectItem.width())
            rectItem.setHeight(rectItem.height())
        else:
            rectItem.moveLeft(rectItem.height()/ratioView - rectItem.width())
            rectItem.setWidth(rectItem.height()/ratioView)
            rectItem.setWidth(rectItem.width())
            rectItem.setHeight(rectItem.height())

        self.ui.main_widget.fitInView(rectItem, Qt.KeepAspectRatio)
        if width is not None and height is not None:
            self.ui.canvas.fig.set_figheight(height/self.ui.canvas.fig.dpi)
            self.ui.canvas.fig.set_figwidth(width/self.ui.canvas.fig.dpi)
        self.ui.canvas.axes.clear()
        if lut is not None:
            # self.ui.canvas.fig.savefig("./new_result.png", bbox_inches='tight')
            self.ui.canvas.axes.imshow(lut, cmap='inferno')
            if peaks is not None:
                self.ui.canvas.axes.plot(peaks[:, 1], peaks[:, 0], 'go', markersize=20)
                if boudnary is not None:
                    self.ui.canvas.axes.imshow(boudnary, cmap='gray')
                if clusters is not None:
                    self.ui.canvas.axes.imshow(clusters, cmap='jet')
        if peaks is not None or lut is not None:
            self.ui.canvas.fig.canvas.draw()
            
    def model_select_slot(self):
        if self.ui.model_select.currentText() == 'ResNet18':
            torch.cuda.empty_cache()
            self.ui.net_name = 'ResNet18'
            self.ui.net = models.resnet_reg.resnet18_reg().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.ResNet18, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'ResNet34':
            self.ui.net_name = 'ResNet34'
            torch.cuda.empty_cache()
            self.ui.net = models.resnet_reg.resnet34_reg().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.ResNet34, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'DenseNet121':
            self.ui.net_name = 'DenseNet121'
            torch.cuda.empty_cache()
            self.ui.net = models.denseNet_reg.denseNet121_reg().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.DenseNet121, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'DenseNet121_mean':
            self.ui.net_name = 'DenseNet121_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.denseNet_reg.denseNet121_reg().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.DenseNet121_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet':
            self.ui.net_name = 'UNet'
            torch.cuda.empty_cache()
            self.ui.net = models.unet_reg.Unet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet_mean':
            self.ui.net_name = 'UNet_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.unet_reg.Unet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet':
            self.ui.net_name = 'UNet'
            torch.cuda.empty_cache()
            self.ui.net = models.unet_reg.Unet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet3+':
            self.ui.net_name = 'UNet3+'
            torch.cuda.empty_cache()
            self.ui.net = models.UNet_3plus.UNet3Plus().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet3plus, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet3+_mean':
            self.ui.net_name = 'UNet3+_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.UNet_3plus.UNet3Plus().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet3plus_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'Laddernet':
            self.ui.net_name = 'Laddernet'
            torch.cuda.empty_cache()
            self.ui.net = models.LadderNet.LadderNetv6().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.Laddernet, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'Laddernet_mean':
            self.ui.net_name = 'Laddernet_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.LadderNet.LadderNetv6().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.Laddernet_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'Convnext':
            self.ui.net_name = 'Convnext'
            torch.cuda.empty_cache()
            self.ui.net = models.ConvNext.ConvNeXtV2().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.Convnext, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'Convnext_mean':
            self.ui.net_name = 'Convnext_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.ConvNext.ConvNeXtV2().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.Convnext_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'finalNet_mean':
            torch.cuda.empty_cache()
            self.ui.net_name = 'finalNet_mean'
            self.ui.net = models.finalnet.finalnet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.finalNet_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'stRegNet_mean':
            torch.cuda.empty_cache()
            self.ui.net_name = 'stRegNet_mean'
            self.ui.net = models.satten_RegNet.st_RegNet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.stRegNet_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'stRegNet_mean':
            torch.cuda.empty_cache()
            self.ui.net_name = 'stRegNet_mean'
            self.ui.net = models.satten_RegNet.st_RegNet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.stRegNet_mean, map_location=torch.device('cpu')), strict=False)
            
            
