import sys
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.uic import loadUi
import cv2
import numpy as np
import qimage2ndarray


# 7장 04.edge_prewitt pyqt5 로 변환해보기
from Common.filters import differential


class EDGEForm(QWidget):
    def __init__(self, parent=None, thread=None):
        super(EDGEForm, self).__init__(parent)
        loadUi('edge.ui', self)
        self.parent = parent
        self.thread = thread
        self.thread.changePixmap.connect(self.setImage)

    def setImage(self, image):
        self.cameraview.setPixmap(QPixmap.fromImage(image))
        

    def differential(image, data1, data2):
        mask1 = np.array(data1, np.float32).reshape(3, 3)
        mask2 = np.array(data2, np.float32).reshape(3, 3)

        dst1 = filter(image, mask1)  # 사용자 정의 회선 함수
        dst2 = filter(image, mask2)
        dst = cv2.magnitude(dst1, dst2)  # 회선 결과 두 행렬의 크기 계산

        dst = cv2.convertScaleAbs(dst)  # 윈도우 표시 위해 OpenCV 함수로 형변환 및 saturation 수행
        dst1 = cv2.convertScaleAbs(dst1)
        dst2 = cv2.convertScaleAbs(dst2)
        return dst, dst1, dst2



    def RobertsBtn(self):
        pass



    def SobelBtn(self):
        pass





    def ch01MB(self):
        pass

    def ch02MB(self):
        pass





# cv2.imshow("image", image)
# cv2.imshow("prewitt edge", dst)
# cv2.imshow("dst1 - vertical mask", dst1)
# cv2.imshow("dst2 - horizontal mask", dst2)
# cv2.waitKey(0)







