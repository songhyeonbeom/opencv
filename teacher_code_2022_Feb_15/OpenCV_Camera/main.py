from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.uic import loadUi
from time import sleep
import os
import random
import sys
import time
import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.image as plt_image
from image_min_max import ImageMinMaxForm
from SecondForm import SecondForm
from CameraHist import CameraHist

class Thread(QThread):
    changePixmap = pyqtSignal(QImage) #사용장가 정의한 시그널
    changePixmapGray = pyqtSignal(QImage)
    changeHist = pyqtSignal(QImage)

    def run(self):
        global cap
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                b,g,r = cv2.split(frame)
                rgbImageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w

                cvc = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                ctg = cvc.convertToFormat(QImage.Format_Grayscale8)

                p = cvc.scaled(640, 480, Qt.KeepAspectRatio)
                p2 = ctg.scaled(640, 480, Qt.KeepAspectRatio)

                self.changePixmap.emit(p)
                self.changePixmapGray.emit(p2)
                self.changeHist.emit(cvc)

    def test(self):
        pass

class WindowClass(QMainWindow) :
    def __init__(self) :
        super(WindowClass, self).__init__()
        loadUi('main.ui', self)

        self.th = Thread(self)

        self.action5_1.triggered.connect(self.openFirstForm)
        self.action5_2.triggered.connect(self.openSecondForm)
        self.action5_11.triggered.connect(self.openImageMinMaxForm)

        self.action6_1.triggered.connect(self.openCameraHistForm)

        self.stackedWidget.insertWidget(1,SecondForm(self))
        self.stackedWidget.insertWidget(2,ImageMinMaxForm(self))
        self.stackedWidget.insertWidget(3,CameraHist(self, self.th))

        canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.matvbox.addWidget(canvas)
        self.ax = canvas.figure.subplots()
        self.ax.plot([0, 1, 2], [1, 5, 3], '-')

        dynamic_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.matvbox.addWidget(dynamic_canvas)

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots()
        self.ax.set_axis_off()

        image = plt_image.imread("images/mine2.jpg")
        self.ax.imshow(image)

    def openFirstForm(self):
        self.stackedWidget.setCurrentIndex(0)

    def openSecondForm(self):
        self.stackedWidget.setCurrentIndex(1)

    def openImageMinMaxForm(self):
        self.stackedWidget.setCurrentIndex(2)

    def openCameraHistForm(self):
        self.stackedWidget.setCurrentIndex(3)

    def display_image(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file',
           '.',"Image Files (*.jpg *.gif *.bmp *.png)")
        pixmap = QPixmap(fname[0])
        self.imageshow.setPixmap(pixmap)
        self.imageshow.setScaledContents(True)

    def convert_to_gray(self):
        grayimage = cv2.imread(fname[0], cv2.IMREAD_GRAYSCALE)
        h, w = grayimage.shape[:2]
        img = QImage(grayimage, w, h, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(img)
        self.graychange.setPixmap(pixmap)
        self.graychange.setScaledContents(True)

    def setImage(self, image):
        self.originvideo.setPixmap(QPixmap.fromImage(image))

    def setImageGray(self, image):
        self.grayvideo.setPixmap(QPixmap.fromImage(image))

    def camera_open(self):
        self.th.changePixmap.connect(self.setImage)
        self.th.changePixmapGray.connect(self.setImageGray)
        self.th.start()
        pass

    def camera_close(self):
        pass



if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()