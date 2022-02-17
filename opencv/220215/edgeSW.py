import sys
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.uic import loadUi
from comm.filters import filter

class Edge(QWidget):
    def __init__(self, parent=None, thread=None):
        super(Edge, self).__init__(parent)
        loadUi('edge.ui', self)
        self.parent = parent
        self.thread = thread
        self.thread.changePixmap.connect(self.setImage)

    def setImage(self, image):
        self.originCam.setPixmap(QPixmap.fromImage(image))

    def edgeImage(self, image):
        ndarry = self.qimg2nparr(image)
        canny = cv2.Canny(ndarry, 100, 150)
        h, w = canny.shape
        qimg = QImage(canny.data, w, h, QImage.Format_Grayscale8)
        self.edgeCam.setPixmap(QPixmap.fromImage(qimg).scaled(self.edgeCam.size(), Qt.KeepAspectRatio))

    def cannyEdgeSlot(self):
        self.thread.changePixmap.connect(self.edgeImage)

    def sobelEdgeSlot(self):
        pass








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