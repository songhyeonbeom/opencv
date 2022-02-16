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

# 7장 04.edge_prewitt pyqt5 로 변환해보기

class EDGEForm(QWidget):
    def __init__(self, parent=None):
        super(EDGEForm, self).__init__(parent)
        loadUi('edge.ui', self)



    def RobertsBtn(self):
        pass

    def SobelBtn(self):
        pass

    def ch01MB(self):
        pass

    def ch02MB(self):
        pass





