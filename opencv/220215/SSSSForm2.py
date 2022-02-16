from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import cv2
from PyQt5.uic import loadUi




class SSSSForm(QWidget):
    def __init__(self, parent=None):
        super(SSSSForm, self).__init__(parent)
        loadUi('main+ab.ui', self)

    def gobBackToOtherForm(self):
        self.parent.stackedWidget.setCurrentIndex(0)
        # self.parent().show()
        # self.close()

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

        img = QImage(axis, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(img)
        self.xax_image.setPixmap(pixmap)
        self.xax_image.setScaledContents(True);

    def xyax_flip(self):
        colorbk_img = cv2.imread(fname[0], cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(colorbk_img, cv2.COLOR_BGR2RGB)
        xy_axis = cv2.flip(colorbk_img, -1)
        height, width = xy_axis.shape[:2]

        img = QImage(xy_axis, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(img)
        self.xyax_image.setPixmap(pixmap)
        self.xyax_image.setScaledContents(True);

    def display_image(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file',
           '.',"Image Files (*.jpg *.gif *.bmp *.png)")
        pixmap = QPixmap(fname[0])
        self.origin_image.setPixmap(pixmap)
        self.origin_image.setScaledContents(True)

