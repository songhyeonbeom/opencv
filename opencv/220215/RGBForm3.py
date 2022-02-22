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


    def edgeImage(self, image):
        ndarry = self.qimg2nparr(image)
        canny = cv2.Canny(ndarry, 100, 150)
        h, w = canny.shape
        qimg = QImage(canny.data, w, h, QImage.Format_Grayscale8)
        self.label_CN.setPixmap(QPixmap.fromImage(qimg).scaled(self.label_CN.size(), Qt.KeepAspectRatio))

    def cannyEdge(self):
        self.thread.changePixmap.connect(self.edgeImage)



    def qimg2nparr(self, qimg):
        ''' convert rgb qimg -> cv2 bgr image '''
        # NOTE: it would be changed or extended to input image shape
        # Now it just used for canvas stroke.. but in the future... I don't know :(

        # qimg = qimg.convertToFormat(QImage.Format_RGB32)
        # qimg = qimg.convertToFormat(QImage.Format_RGB888)
        h, w = qimg.height(), qimg.width()
        print(h,w)
        ptr = qimg.constBits()
        ptr.setsize(h * w * 3)
        print(h, w, ptr)
        return np.frombuffer(ptr, np.uint8).reshape(h, w, 3)  # Copies the data
        #return np.array(ptr).reshape(h, w, 3).astype(np.uint8)  #  Copies the data




    def moUP(self):
        move = self.label_ORG.geometry()
        move.moveTop(move.y() - 10)
        self.label_ORG.setGeometry(move)

    def moDN(self):
        move = self.label_ORG.geometry()
        move.moveTop(move.y() + 10)
        self.label_ORG.setGeometry(move)

    def moLF(self):
        move = self.label_ORG.geometry()
        move.moveLeft(move.x() - 10)
        self.label_ORG.setGeometry(move)

    def moRT(self):
        move = self.label_ORG.geometry()
        move.moveLeft(move.x() + 10)
        self.label_ORG.setGeometry(move)

    def reSET(self):
        self.label_ORG.setGeometry(290, 230, 161, 161)

























































