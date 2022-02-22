import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.uic import loadUi
from numpy import ndarray
import numpy as np,  cv2
from Common.interpolation import bilinear_value
from Common.utils import contain









class RGBForm(QWidget):
    def __init__(self, parent=None, thread=None):
        super(RGBForm, self).__init__(parent)
        loadUi('mainRGB+.ui', self)


        self.parent = parent
        self.thread = thread
        self.thread.changePixmap.connect(self.setImage)

        self.thread.changeB.connect(self.setB)
        self.thread.changeG.connect(self.setG)
        self.thread.changeR.connect(self.setR)
        self.thread.changeRB.connect(self.setRB)
        self.thread.changeGB.connect(self.setGB)

        # self.lbR.setPixmap(QPixmap.fromImage(Rimg).scaled(self.lbR.size(), Qt.KeepAspectRatio))


    def setImage(self, image):
        self.label_ORG.setPixmap(QPixmap.fromImage(image))


    def setB(self, image):
        self.label_B.setPixmap(QPixmap.fromImage(image))

    def setG(self, image):
        self.label_G.setPixmap(QPixmap.fromImage(image))

    def setR(self, image):
        self.label_R.setPixmap(QPixmap.fromImage(image))

    def setRB(self, image):
        self.label_RB.setPixmap(QPixmap.fromImage(image))

    def setGB(self, image):
        self.label_GB.setPixmap(QPixmap.fromImage(image))





































































