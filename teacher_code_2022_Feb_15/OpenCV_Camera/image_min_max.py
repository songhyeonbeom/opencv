from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.uic import loadUi
import cv2
import numpy as np

class ImageMinMaxForm(QWidget):
    changePixmapsignal = pyqtSignal(str) #사용장가 정의한 시그널

    def __init__(self, parent=None):
        super(ImageMinMaxForm, self).__init__(parent)
        loadUi('image_min_max.ui', self)
        self.parent=parent
        self.changePixmapsignal.connect(self.cvt_image2)

    def image_open(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file',
           'd:\\',"Image Files (*.jpg *.gif *.bmp *.png)")
        pixmap = QPixmap(fname[0])
        self.origin_image.setPixmap(pixmap)
        self.origin_image.setScaledContents(True);

        self.changePixmapsignal.emit(str(fname[0]))


    def cvt_image2(self, image):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None: raise Exception("영상 파일 읽기 오류 발생")
        (min_val, max_val, _, _) = cv2.minMaxLoc(image)  # 최솟값과 최댓값 가져오기

        ratio = 255 / (max_val - min_val)
        dst = np.round((image - min_val) * ratio).astype('uint8')
        (min_dst, max_dst, _, _) = cv2.minMaxLoc(dst)

        print("원본 영상 최솟값= %d , 최댓값= %d" % (min_val, max_val))
        print("수정 영상 최솟값= %d , 최댓값= %d" % (min_dst, max_dst))
        height, width = dst.shape[:2]

        img = QImage(dst, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(img)
        self.cvt_image.setPixmap(pixmap)
        self.cvt_image.setScaledContents(True);
        self.cvt_min.setText(str(min_val))
        self.cvt_max.setText(str(max_val))

