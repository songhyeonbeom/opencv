import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from time import sleep
import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.uic import loadUi



class GTCForm(QWidget):
    def __init__(self, parent=None):
        super(GTCForm, self).__init__(parent)
        loadUi('GTC.ui', self)

    def display_image(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '.',"Image Files (*.jpg *.gif *.bmp *.png)")
        pixmap = QPixmap(fname[0])
        self.label01.setPixmap(pixmap)
        self.label01.setScaledContents(True)
        self.label02.setPixmap(pixmap)
        self.label02.setScaledContents(True)



    def onThreshold(value):
        pass

    #     th[0] = cv2.getTrackbarPos("Hue_th1", "result")
    #     th[1] = cv2.getTrackbarPos("Hue_th2", "result")
    #
    #     _, result = cv2.threshold(hue, th[1], 255, cv2.THRESH_TOZERO_INV)
    #     cv2.threshold(result, th[0], 255, cv2.THRESH_BINARY, result)
    #     cv2.imshow("result", result)
    #
    # BGR_img = cv2.imread("images/color_space.jpg", cv2.IMREAD_COLOR)
    # if BGR_img is None: raise Exception("영상 파일 읽기 오류")
    #
    # HSV_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2HSV)
    # hue = np.copy(HSV_img[:, :, 0])
    #
    # th = [50, 100]
    # cv2.namedWindow("result")
    # cv2.createTrackbar("Hue_th1", "result", th[0], 255, onThreshold)
    # cv2.createTrackbar("Hue_th2", "result", th[1], 255, onThreshold)
    # onThreshold(th[0])
    # cv2.imshow("BGR_img", BGR_img)
    # cv2.waitKey(0)