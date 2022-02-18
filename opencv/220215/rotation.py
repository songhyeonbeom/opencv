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



class RotationForm(QWidget):
    def __init__(self, parent=None, thread=None):
        super(RotationForm, self).__init__(parent)
        loadUi('rotation.ui', self)


        self.parent = parent
        self.thread = thread
        self.thread.changePixmap.connect(self.setImage)

    def setImage(self, image):
        self.Cam01.setPixmap(QPixmap.fromImage(image))


    def Rotation01(self):
        self.thread.changePixmap.connect(self.YSrotation_01)

    def YSrotation_01(self):
        pass









    # def rotate(img, degree):
    #     dst = np.zeros(img.shape[:2], img.dtype)  # 목적 영상 생성
    #     radian = (degree / 180) * np.pi  # 회전 각도 - 라디언
    #     sin, cos = np.sin(radian), np.cos(radian)  # 사인, 코사인 값 미리 계산
    #
    #     for i in range(img.shape[0]):  # 목적 영상 순회 - 역방향 사상
    #         for j in range(img.shape[1]):
    #             y = -j * sin + i * cos
    #             x = j * cos + i * sin  # 회선 변환 수식
    #             if contain((y, x), img.shape):  # 입력 영상의 범위 확인
    #                 dst[i, j] = bilinear_value(img, [x, y])  # 화소값 양선형 보간
    #
    #     img.Cam02.setPixmap()
    #
    #
    # def bilinear_value(img, pt):
    #     x, y = np.int32(pt)
    #     if y >= img.shape[0] - 1: y = y - 1
    #     if x >= img.shape[1] - 1: x = x - 1
    #
    #     P1, P2, P3, P4 = np.float32(img[y:y + 2, x:x + 2].flatten())
    #     alpha, beta = pt[1] - y, pt[0] - x  # 거리 비율
    #
    #     M1 = P1 + alpha * (P3 - P1)  # 1차 보간
    #     M2 = P2 + alpha * (P4 - P2)
    #     P = M1 + beta * (M2 - M1)  # 2차 보간
    #     return np.clip(P, 0, 255)  # 화소값 saturation후 반환
    #
    # def contain(p, shape):  # 좌표(y,x)가 범위내 인지 검사
    #     return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
    #
    # def qimg2nparr(self, qimg):
    #     ''' convert rgb qimg -> cv2 bgr image '''
    #     # NOTE: it would be changed or extended to input image shape
    #     # Now it just used for canvas stroke.. but in the future... I don't know :(
    #
    #     # qimg = qimg.convertToFormat(QImage.Format_RGB32)
    #     # qimg = qimg.convertToFormat(QImage.Format_RGB888)
    #     h, w = qimg.height(), qimg.width()
    #     print(h,w)
    #     ptr = qimg.constBits()
    #     ptr.setsize(h * w * 3)
    #     print(h, w, ptr)
    #     return np.frombuffer(ptr, np.uint8).reshape(h, w, 3)  # Copies the data
    #     #return np.array(ptr).reshape(h, w, 3).astype(np.uint8)  #  Copies the data


# def rotate(img, degree):
#     dst = np.zeros(img.shape[:2], img.dtype)                     # 목적 영상 생성
#     radian = (degree/180) * np.pi                               # 회전 각도 - 라디언
#     sin, cos = np.sin(radian), np.cos(radian)   # 사인, 코사인 값 미리 계산
#
#     for i in range(img.shape[0]):                                       # 목적 영상 순회 - 역방향 사상
#         for j in range(img.shape[1]):
#             y = -j * sin + i * cos
#             x =  j * cos + i * sin                  # 회선 변환 수식
#             if contain((y, x), img.shape):             # 입력 영상의 범위 확인
#                 dst[i, j] = bilinear_value(img, [x, y])           # 화소값 양선형 보간
#     return dst
#
# def rotate_pt(img, degree, pt):
#     dst = np.zeros(img.shape[:2], img.dtype)                     # 목적 영상 생성
#     radian = (degree/180) * np.pi                               # 회전 각도 - 라디언
#     sin, cos = np.sin(radian), np.cos(radian)   # 사인, 코사인 값 미리 계산
#
#     for i in range(img.shape[0]):                              # 목적 영상 순회 - 역방향 사상
#         for j in range(img.shape[1]):
#             jj, ii = np.subtract((j, i), pt)                # 중심좌표 평행이동,
#             y = -jj * sin + ii * cos               # 회선 변환 수식
#             x =  jj * cos + ii * sin
#             x, y = np.add((x, y), pt)
#             if contain((y, x), img.shape):                      # 입력 영상의 범위 확인
#                 dst[i, j] = bilinear_value(img, [x, y])           # 화소값 양선형 보간
#     return dst

