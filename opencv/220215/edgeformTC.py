from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.uic import loadUi
import cv2
import numpy as np
import qimage2ndarray
from numpy import ndarray


class EdgeForm(QWidget):

    edgeChageSignal = pyqtSignal(ndarray)

    def __init__(self, parent=None, thread=None):
        super(EdgeForm, self).__init__(parent)
        loadUi('edgeformTC.ui', self)
        self.parent=parent
        self.thread=thread
        self.thread.changeEdge.connect(self.camera_connect)


    def camera_connect(self,frame): #type ndarray
        self.edgeChageSignal.emit(frame)
        h, w = frame.shape
        cvc = QImage(frame.data, w, h, QImage.Format_Grayscale8)
        self.grayvideo.setPixmap(QPixmap.fromImage(cvc))

        pass

    def RobertsEdge(self, frame):
        data1 = [-1, 0, 0,
                 0, 1, 0,
                 0, 0, 0]
        data2 = [0, 0, -1,
                 0, 1, 0,
                 0, 0, 0]

#        gray_image = cv2.cvtColor(numpy_arr, cv2.COLOR_BGR2GRAY)
        dst, dst1, dst2 = self.differential(frame, data1, data2)
        print(dst.shape)
        h, w = dst.shape

        cvc = QImage(dst.data, w, h, QImage.Format_Grayscale8)
        self.grayvideo.setPixmap(QPixmap.fromImage(cvc))

        # cvc = QImage(dst.data, w, h, QImage.Format_Grayscale8)
        # self.maskvideo.setPixmap(QPixmap.fromImage(cvc))
        pass

    def PrewittEdge(self):

        pass


    def SobelEdgeSlot(self):
        self.edgeChageSignal.connect(self.SobelEdge)

    def SobelEdge(self, image):
        dst3 = cv2.Sobel(np.float32(image), cv2.CV_32F, 1, 0, 3)  # x방향 미분 - 수직 마스크
        dst4 = cv2.Sobel(np.float32(image), cv2.CV_32F, 0, 1, 3)  # y방향 미분 - 수평 마스크
        dst3 = cv2.convertScaleAbs(dst3)  # 절댓값 및 uint8 형변환
        dst4 = cv2.convertScaleAbs(dst4)
        h3, w3 = dst3.shape
        h4, w4 = dst4.shape
        edge3 = QImage(dst3.data, w3, h3, QImage.Format_Grayscale8)
        edge4 = QImage(dst4.data, w4, h4, QImage.Format_Grayscale8)
        self.maskvideo3.setPixmap(QPixmap.fromImage(edge3))
        self.maskvideo4.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.maskvideo4.setPixmap(QPixmap.fromImage(edge4))
        pass

    def DifferentialEdge(self):

        pass

    def TwoDifferentialEdge(self):

        pass

    def LoGEdgeSlot(self):
        self.edgeChageSignal.connect(self.LoGEdge)


    def LoGEdge(self, image):



        pass

    def DoGEdgeSlot(self):
        self.edgeChageSignal.connect(self.DoGEdge)

    def DoGEdge(self, image):

        pass


    def CannyEdgeSlot(self):
        print("CannyEdgeSlot Start")
        self.edgeChageSignal.connect(self.CannyEdge)

    def CannyEdge(self, image):
        canny1= np.uint8(image)
        canny2 = cv2.Canny(canny1.astype( np.uint8 ), 0, 150)  # OpenCV 캐니 에지
        h3, w3 = canny2.shape
        edge3 = QImage(canny2.data, w3, h3, QImage.Format_Grayscale8)
        self.maskvideo3.setPixmap(QPixmap.fromImage(edge3))


    def differential(self, image, data1, data2):
        mask1 = np.array(data1, np.float32).reshape(3, 3)
        mask2 = np.array(data2, np.float32).reshape(3, 3)

        dst1 = self.filter(image, mask1)  # 사용자 정의 회선 함수
        dst2 = self.filter(image, mask2)
        dst = cv2.magnitude(dst1, dst2)  # 회선 결과 두 행렬의 크기 계산
        dst1, dst2 = np.abs(dst1), np.abs(dst2)  # 회선 결과 행렬 양수 변경
        print("111111111113")

        dst = np.clip(dst, 0, 255).astype("uint8")
        dst1 = np.clip(dst1, 0, 255).astype("uint8")
        dst2 = np.clip(dst2, 0, 255).astype("uint8")
        print("111111111114")
        return dst, dst1, dst2

    # 회선 수행 함수 - 행렬 처리 방식(속도 면에서 유리)
    def filter(self, image, mask):
        rows, cols = image.shape[:2]
        dst = np.zeros((rows, cols), np.float32)  # 회선 결과 저장 행렬
        xcenter, ycenter = mask.shape[1] // 2, mask.shape[0] // 2  # 마스크 중심 좌표

        for i in range(ycenter, rows - ycenter):  # 입력 행렬 반복 순회
            for j in range(xcenter, cols - xcenter):
                y1, y2 = i - ycenter, i + ycenter + 1  # 관심영역 높이 범위
                x1, x2 = j - xcenter, j + xcenter + 1  # 관심영역 너비 범위
                roi = image[y1:y2, x1:x2].astype("float32")  # 관심영역 형변환

                tmp = cv2.multiply(roi, mask)  # 회선 적용 - OpenCV 곱셈
                dst[i, j] = cv2.sumElems(tmp)[0]  # 출력화소 저장
        return dst  # 자료형 변환하여 반환

    def qimg2nparr(self, qimg):
        ''' convert rgb qimg -> cv2 bgr image '''
        # NOTE: it would be changed or extended to input image shape
        # Now it just used for canvas stroke.. but in the future... I don't know :(

        # qimg = qimg.convertToFormat(QImage.Format_RGB32)
        # qimg = qimg.convertToFormat(QImage.Format_RGB888)
        h, w = qimg.height(), qimg.width()
#        print(h,w)
        ptr = qimg.constBits()
        ptr.setsize(h * w * 4)
#        print(h, w, ptr)
        return np.frombuffer(ptr, np.uint8).reshape(h, w, 4)  # Copies the data
        #return np.array(ptr).reshape(h, w, 3).astype(np.uint8)  #  Copies the data