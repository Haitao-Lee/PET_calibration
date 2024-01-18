#coding = utf-8
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

 
# pyplot.rcParams['font.sans-serif'] = ['SimHei']
# pyplot.rcParams['axes.unicode_minus'] = False

class my_figure(FigureCanvasQTAgg):
    def __init__(self, width, height, dpi, parent=None):
        self.fig = pyplot.figure(figsize=(width, height), facecolor='#000000', dpi=dpi, edgecolor='#0000FF')
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#000000')

    # def show(self, width=None, height=None, lut=None, peaks=None):
    #     if width is not None and height is not None:
    #         self.fig.set_figheight(height/self.fig.dpi)
    #         self.fig.set_figwidth(width/self.fig.dpi)
    #     if lut is not None:
    #         print('show lut')
    #         self.axes.clear()
    #         self.axes.matshow(lut, cmap='inferno')
    #     if peaks is not None:
    #         self.axes.plot(peaks[:, 1], peaks[:, 0], 'bo')
    #     if peaks is not None or lut is not None:
    #         self.fig.canvas.draw()

