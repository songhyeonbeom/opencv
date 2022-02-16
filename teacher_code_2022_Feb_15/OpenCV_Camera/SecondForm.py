from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.uic import loadUi
import cv2

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