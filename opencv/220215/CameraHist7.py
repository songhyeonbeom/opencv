from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
import cv2
import numpy as np
from PyQt5 import uic
import qimage2ndarray

# 6장 25페이지 예제 6.3.3 히스토그램 그래프 그리기 - 09.draw_histogram.py
# 심화예제 27페이지 6.3.4 색상 히스토그램 그리기 - 10.hue_histogram.py


class CameraHist(QWidget):
    changePixmapsignal = pyqtSignal(str) #사용장가 정의한 시그널

    def __init__(self, parent=None, thread=None):
        super(CameraHist, self).__init__(parent)
        loadUi('camera_hist.ui', self)
        self.parent = parent
        self.thread = thread
        self.thread.changeHist.connect(self.camera_connect)


    def camera_connect(self,frame):
        import qimage2ndarray
        self.camera_view.setPixmap(QPixmap.fromImage(frame))
        numpy_arr = self.qimg2nparr(frame)

        hist = cv2.calcHist([numpy_arr], [3], None, [32], [0, 256])
        # hist_img = self.draw_histo(hist)
        hue_hist_img= self.draw_histo_hue(hist)
        h, w, ch = hue_hist_img.shape
        bytesPerLine = ch * w
        cvc = QImage(hue_hist_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.hist_red.setPixmap(QPixmap.fromImage(cvc))


    def nparr2qimg(self, cvimg):
        ''' convert cv2 bgr image -> rgb qimg '''
        h, w, c = cvimg.shape
        byte_per_line = w * c  # cvimg.step() #* step # NOTE:when image format problem..
        return QImage(cvimg.data, w, h, byte_per_line,
                      QImage.Format_RGB888).rgbSwapped()

    def qimg2nparr(self, qimg):
        ''' convert rgb qimg -> cv2 bgr image '''
        # NOTE: it would be changed or extended to input image shape
        # Now it just used for canvas stroke.. but in the future... I don't know :(

        # qimg = qimg.convertToFormat(QImage.Format_RGB32)
        # qimg = qimg.convertToFormat(QImage.Format_RGB888)
        h, w = qimg.height(), qimg.width()
        # print(h,w)   #밑에 계속 뜨는거 잠시 안나오게 했음.
        ptr = qimg.constBits()
        ptr.setsize(h * w * 4)
        # print(h, w, ptr)
        return np.frombuffer(ptr, np.uint8).reshape(h, w, 4)  # Copies the data
        #return np.array(ptr).reshape(h, w, 3).astype(np.uint8)  #  Copies the data

    def draw_histo(hist, shape=(200, 256,3)):

        hist_img = np.full(shape, 255, np.uint8)
        cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)
        gap = hist_img.shape[1] / hist.shape[0]  # 한 계급 너비 256/32 = 8

        for i, h in enumerate(hist):
            x = int(round(i * gap))  # 막대 사각형 시작 x 좌표
            w = int(round(gap))
            roi = (x, 0, w, int(h))
            cv2.rectangle(hist_img, roi, 0, cv2.FILLED)

        return cv2.flip(hist_img, 0)  # 영상 상하 뒤집기 후 반환

    def make_palette(self, rows):
        # 리스트 생성 방식
        hue = [round(i * 180 / rows) for i in range(rows)]  # hue 값 리스트 계산
        hsv = [[(h, 255, 255)] for h in hue]  # (hue, 255,255) 화소값 계산
        hsv = np.array(hsv, np.uint8)  # numpy 행렬의 uint8형 변환
        # # 반복문 방식
        # hsv = np.full((rows, 1, 3), (255,255,255), np.uint8)
        # for i in range(0, rows):                                # 행수만큼 반복
        #     hue = round(i / rows * 180 )                        # 색상 계산
        #     hsv[i] = (hue, 255, 255)                            # HSV 컬러 지정

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # HSV 컬러 -> BGR 컬러

    def draw_histo_hue(self, hist, shape=(200, 256, 3)):
        hsv_palate = self.make_palette(hist.shape[0])  # 색상 팔레트 생성
        hist_img = np.full(shape, 255, np.uint8)
        cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)  # 정규화

        gap = hist_img.shape[1] / hist.shape[0]  # 한 계급 크기
        for i, h in enumerate(hist):
            x, w = int(round(i * gap)), int(round(gap))
            color = tuple(map(int, hsv_palate[i][0]))  # 정수형 튜플로 변환
            cv2.rectangle(hist_img, (x, 0, w, int(h)), color, cv2.FILLED)  # 팔레트 색으로 그리기

        return cv2.flip(hist_img, 0)
