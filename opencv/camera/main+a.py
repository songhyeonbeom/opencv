import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from time import sleep
import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.uic import loadUi



class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    changePixmapGray = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(2)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgbImageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w

                cvc = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                ctg = cvc.convertToFormat(QImage.Format_Grayscale8)

                p = cvc.scaled(240, 480, Qt.KeepAspectRatio)
                p2 = ctg.scaled(240, 480, Qt.KeepAspectRatio)

                self.changePixmap.emit(p)
                self.changePixmapGray.emit(p2)





class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi('main+a.ui', self)
        self.action5_1.triggered.connect(self.openSecondForm)
        self.action_main.triggered.connect(self.openMainForm)
        self.action_SSSS.triggered.connect(self.openSSSSForm)
        self.action_RGB.triggered.connect(self.openRGBForm)
        self.action_BW.triggered.connect(self.openBWForm)

        otherview = SecondForm(self)
        self.stackedWidget.insertWidget(1, otherview)

        sssotherview = SSSSForm(self)
        self.stackedWidget.insertWidget(2, sssotherview)

        rgbotherview = RGBForm(self)
        self.stackedWidget.insertWidget(3, rgbotherview)

        blackwhiteview = BWForm(self)
        self.stackedWidget.insertWidget(4, blackwhiteview)



        canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.vbox.addWidget(canvas)

        self.ax = canvas.figure.subplots()
        self.ax.plot([0, 1, 2], [1, 5, 3], '-')

        dynamic_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.vbox.addWidget(dynamic_canvas)

        self.dynamic_ax = dynamic_canvas.figure.subplots()
        self.timer = dynamic_canvas.new_timer(100, [(self.update_canvas, (), {})])
        self.timer.start()


        # image = cv2.imread("images/mine2.jpg", cv2.IMREAD_COLOR)
        # image_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        # self.vbox.addWidget(dynamic_canvas)
        # image_canvas.imshow()





    def update_canvas(self):
        self.dynamic_ax.clear()
        t = np.linspace(0, 2 * np.pi, 101)
        self.dynamic_ax.plot(t, np.sin(t + time.time()), color='deeppink')
        self.dynamic_ax.figure.canvas.draw()



    def setImage(self, image):
        self.originvideo.setPixmap(QPixmap.fromImage(image))

    def setImageGray(self, image):
        self.originvideo_2.setPixmap(QPixmap.fromImage(image))

    def Camera_Open(self):
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.changePixmapGray.connect(self.setImageGray)
        th.start()

    def Camera_close(self):
        pass


    def display_image(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '.',"Image Files (*.jpg *.gif *.bmp *.png)")
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



    def openMainForm(self):
        self.stackedWidget.setCurrentIndex(0)

    def openSecondForm(self):
        self.stackedWidget.setCurrentIndex(1)

    def openSSSSForm(self):
        self.stackedWidget.setCurrentIndex(2)

    def openRGBForm(self):
        self.stackedWidget.setCurrentIndex(3)

    def openBWForm(self):
        self.stackedWidget.setCurrentIndex(4)

    def imageread(self):
        pass





class BWForm(QWidget):
    def __init__(self, parent=None):
        super(BWForm, self).__init__(parent)
        loadUi('blackwhite.ui', self)
        self.button3.clicked.connect(self.gobBackToOtherForm)

    def gobBackToOtherForm(self):
        self.parent.stackedWidget.setCurrentIndex(0)

    def display_image(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file',
           '.',"Image Files (*.jpg *.gif *.bmp *.png)")
        pixmap = QPixmap(fname[0])
        self.mttyy01.setPixmap(pixmap)
        self.mttyy01.setScaledContents(True)





class RGBForm(QWidget):
    def __init__(self, parent=None):
        super(RGBForm, self).__init__(parent)
        loadUi('mainRGB+.ui', self)
        self.button5.clicked.connect(self.gobBackToOtherForm)

    def gobBackToOtherForm(self):
        self.parent.stackedWidget.setCurrentIndex(0)


class SecondForm(QWidget):
    def __init__(self, parent=None):
        super(SecondForm, self).__init__(parent)
        loadUi('secondform.ui', self)
        self.button2.clicked.connect(self.gobBackToOtherForm)

    def gobBackToOtherForm(self):
        self.parent.stackedWidget.setCurrentIndex(0)



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


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()