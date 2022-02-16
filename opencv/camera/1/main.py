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

form_class = uic.loadUiType("untitled.ui")[0]


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





class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
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







if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()