import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget
from Ui_lut_calibration import Ui_MainWindow
from slot_functions import slot_functions
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from config import args
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pylab
import visualization
import torch
import models.finalnet
import numpy as np


class calibration_software(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(calibration_software, self).__init__()
        self.setupUi(self)
        self.setupButton()
        self.lut = None
        self.peak = None
        self.gt = None
        self.binary_border = None
        self.cluster_lut = None
        self.mean_cor = np.load('./meancordinate.npy')
        self.vis = 0
        self.cache = './cache/tmp_img.jpg'
        self.setupCache(os.path.dirname(self.cache))
        self.net = models.finalnet.finalnet().to(args.device)
        self.net.load_state_dict(torch.load(args.finalNet_mean, map_location=torch.device('cpu')), strict=False)
        self.net_name = 'finalNet_mean'
        
        self.working_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
        self.filemodel = QFileSystemModel()
        self.filemodel.setRootPath(self.working_dir)
        self.working_tree.setModel(self.filemodel)
        # self.working_tree.allColumnsShowFocus()
        
        self.slot_fs = slot_functions(self)
        self.verticalSplitter = QSplitter(Qt.Vertical)
        self.horizonSplitter = QSplitter(Qt.Horizontal)        
        
        self.zoom_factor = 3
        self.dpi = 60
        self.canvas = visualization.my_figure(width=self.main_widget.height(), height=self.main_widget.height(), dpi=self.dpi)
        # self.canvas.resize(self.main_widget.width()/self.dpi, self.main_widget.height()/self.dpi)
        # self.canvas.show(width=self.main_widget.width(), height=self.main_widget.height())
        self.graphicscene = QGraphicsScene()  
        self.graphicscene.addWidget(self.canvas)
        self.main_widget.setScene(self.graphicscene)
        #self.main_widget.centerOn(0, 0)
        self.main_widget.setAlignment(Qt.AlignCenter| Qt.AlignCenter)
        self.main_widget.setDragMode(QGraphicsView.ScrollHandDrag) 
        self.main_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn) 
        self.main_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.main_widget.show()
        
        self.setupConnections()
        self.setupSplitters()

    
    def setupButton(self):
        # setStyleSheet(
        #     "QPushButton{color: black}"
        #     "QPushButton{font: 22pt \"Agency FB\";\n}"
        #     "QPushButton{border-image: url(image/blue_label.png)}"
        #     "QPushButton:hover{border-image: url(image/blue_label_3.png)}"
        #     "QPushButton:pressed{border-image: url(image/blue_label_2.png)}"
        #     "QPushButton{border-radius:20px;}")
        self.visual.setStyleSheet("QPushButton{border-radius:20px;background-color: rgb(80, 80, 80);}")
        self.isolate.setStyleSheet("QPushButton{border-radius:20px;background-color: rgb(80, 80, 80);}")
        self.predict.setStyleSheet("QPushButton{border-radius:20px;background-color: rgb(80, 80, 80);}")
        self.model_select.setEditable(True)
        self.model_select.lineEdit().setReadOnly(True)
        self.model_select.lineEdit().setAlignment(Qt.AlignCenter)
        self.modelIdx = 0
        font = QFont()
        items = ['finalNet_mean','ResNet18','ResNet34','DenseNet121','DenseNet121_mean', 'UNet','UNet_mean',
                 'UNet3+', 'UNet3+_mean','Laddernet', "Laddernet_mean","Convnext","Convnext_mean","stRegNet_mean","Groundtruth"]
        self.model_select.addItems(items)
        self.model_select.setStyleSheet("background-color: rgb(80, 80, 80);\n""font: 75 16pt \"Times New Roman\";\n""font-weight: bold;")
        # self.model_select.setFont(font)
        self.model_select.setItemData(self.model_select.currentIndex(), font, role=Qt.FontRole)
        #self.model_select.setStyleSheet("QCombobox{background-color: rgb(80, 80, 80);}")
    
    def setupConnections(self):
        self.QFile.triggered.connect(self.slot_fs.import_file_slot)
        self.QFolder.triggered.connect(self.slot_fs.import_folder_slot)
        self.QSave.triggered.connect(self.slot_fs.save_slot)
        self.QQuit.triggered.connect(self.slot_fs.quit_slot)
        self.predict.clicked.connect(self.slot_fs.predict_slot)
        # self.update.clicked.connect(self.slot_fs.update_slot)
        self.visual.clicked.connect(self.slot_fs.visual_slot)
        self.isolate.clicked.connect(self.slot_fs.isolate_slot)
        self.working_tree.doubleClicked.connect(self.slot_fs.get_file)
        self.model_select.currentIndexChanged.connect((self.slot_fs.model_select_slot)) 
        pass
    
    def setupSplitters(self):
        self.horizonSplitter.setStyleSheet("background-color:rgb(180, 180, 180);border:0px solid green")
        self.verticalSplitter.setStyleSheet("background-color:rgb(180, 180, 180);border:0px solid green")
        self.verticalSplitter.setHandleWidth(0)
        self.verticalSplitter.setChildrenCollapsible(1)
        self.horizonSplitter.setHandleWidth(0)
        self.horizonSplitter.setChildrenCollapsible(1)
        self.verticalSplitter.addWidget(self.working_tree) 
        self.verticalSplitter.addWidget(self.label_4)
        self.verticalSplitter.addWidget(self.groupBox_4)
        self.verticalSplitter.addWidget(self.groupBox)
        self.verticalSplitter.addWidget(self.groupBox_3)
        self.working_tree.setColumnWidth(0, 800)
        self.groupBox_2.hide()
        self.horizonSplitter.addWidget(self.verticalSplitter)
        self.horizonSplitter.addWidget(self.main_widget)
        self.widget.layout().replaceWidget(self.groupBox_5, self.horizonSplitter)
        self.groupBox_5.hide()
        self.horizonSplitter.setMouseTracking(True)
        self.horizonSplitter.setStretchFactor(0, 6)
        self.horizonSplitter.setStretchFactor(1, 4)
        
    def setupCache(self, direc):
        if not os.path.exists(direc):  
            os.makedirs(direc)
        
 
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = calibration_software()
    window.show()
    sys.exit(app.exec_())
