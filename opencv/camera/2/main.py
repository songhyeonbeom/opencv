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

class Thread(QThread):
    changePixmap = pyqtSignal(QImage) #사용장가 정의한 시그널
    changePixmapGray = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(2)
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

class WindowClass(QMainWindow) :
    def __init__(self) :
        super(WindowClass, self).__init__()
        loadUi('main.ui', self)
        self.action5_1.triggered.connect(self.openFirstForm)
        self.action5_2.triggered.connect(self.openSecondForm)

        otherview = SecondForm(self)

        self.stackedWidget.insertWidget(1,otherview)

        canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.vbox.addWidget(canvas)
        self.ax = canvas.figure.subplots()
        self.ax.plot([0, 1, 2], [1, 5, 3], '-')

        dynamic_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.vbox.addWidget(dynamic_canvas)

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots()
        self.ax.set_axis_off()

        image = plt_image.imread("mine2.jpg")
        self.ax.imshow(image)

    def openFirstForm(self):
        self.stackedWidget.setCurrentIndex(0)

    def openSecondForm(self):
        self.stackedWidget.setCurrentIndex(1)


    def display_image(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file', '.',"Image Files (*.jpg *.gif *.bmp *.png)")
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
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.changePixmapGray.connect(self.setImageGray)
        th.start()
        pass

    def camera_close(self):
        pass


class SecondForm(QWidget):

    def __init__(self, parent=None):
        super(SecondForm, self).__init__(parent)
        loadUi('secondform.ui', self)
        self.parent=parent
        self.closeButton.clicked.connect(self.goBackToOtherForm)

    def yax_flip(self):
        gray_img = cv2.imread(fname[0], cv2.IMREAD_COLOR)
        y_axis = cv2.flip(gray_img, 1)
        height, width = y_axis.shape[:2]

        img = QImage(y_axis, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(img)
        self.yax_image.setPixmap(pixmap)
        self.yax_image.setScaledContents(True);


    def xax_flip(self):
        color_img = cv2.imread(fname[0], cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        axis = cv2.flip(rgb_image, 0)
        height, width = axis.shape[:2]
        # self.xax_image.setFixedWidth(width)
        # self.xax_image.setFixedHeight(height)

        img = QImage(axis, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.xax_image.setPixmap(pixmap)
        self.xax_image.setScaledContents(True);

    def tax_flip(self):
        pass

    def image_read(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file',
           'd:\\',"Image Files (*.jpg *.gif *.bmp *.png)")
        pixmap = QPixmap(fname[0])
        self.orign_image.setPixmap(pixmap)
        self.orign_image.setScaledContents(True);


    def goBackToOtherForm(self):
        self.parent.stackedWidget.setCurrentIndex(0)

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()