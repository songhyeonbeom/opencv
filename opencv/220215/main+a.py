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
# from FormPK import SecondForm1, SSSSForm2, RGBForm3, BWForm4, GTCForm5
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.uic import loadUi
from PyQt5 import QtGui


from SecondForm1 import SecondForm
from SSSSForm2 import SSSSForm
from RGBForm3 import RGBForm
from BWForm4 import BWForm
from GTCForm5 import GTCForm
from EDGEForm6 import EDGEForm
from CameraHist7 import CameraHist
from edgeSW import EdgeSW
from rotation import RotationForm


# ch5-3 RGB 확인하기


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    changePixmapGray = pyqtSignal(QImage)
    changeHist = pyqtSignal(QImage)
    changeEdge = pyqtSignal(ndarray)

    changeB = pyqtSignal(QImage)
    changeG = pyqtSignal(QImage)
    changeR = pyqtSignal(QImage)
    changeRG = pyqtSignal(QImage)
    changeRB = pyqtSignal(QImage)
    changeGB = pyqtSignal(QImage)

    recordFrame = pyqtSignal(np.ndarray)



    def run(self):
        global cap
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            self.recordFrame.emit(frame)

            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = frame.shape

                zeros = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
                B, G, R = cv2.split(frame)

                BH_R = cv2.merge((R, zeros, zeros))
                BH_G = cv2.merge((zeros, G, zeros))
                BH_B = cv2.merge((zeros, zeros, B))
                BH_RG = cv2.merge((R, G, zeros))
                BH_RB = cv2.merge((R, zeros, B))
                BH_GB = cv2.merge((zeros, G, B))

                R = QtGui.QImage(BH_R.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(161, 150, Qt.KeepAspectRatio)
                G = QtGui.QImage(BH_G.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(161, 150, Qt.KeepAspectRatio)
                B = QtGui.QImage(BH_B.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(161, 150, Qt.KeepAspectRatio)
                RG = QtGui.QImage(BH_RG.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(161, 150, Qt.KeepAspectRatio)
                RB = QtGui.QImage(BH_RB.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(161, 150, Qt.KeepAspectRatio)
                GB = QtGui.QImage(BH_GB.data, w, h, w * c, QtGui.QImage.Format_RGB888).scaled(161, 150, Qt.KeepAspectRatio)


                rgbImageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w

                cvc = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                ctg = cvc.convertToFormat(QImage.Format_Grayscale8)

                p = cvc.scaled(240, 480, Qt.KeepAspectRatio)
                p2 = ctg.scaled(240, 480, Qt.KeepAspectRatio)

                self.changeR.emit(R)
                self.changeG.emit(G)
                self.changeB.emit(B)
                self.changeRG.emit(RG)
                self.changeRB.emit(RB)
                self.changeGB.emit(GB)


                self.changePixmap.emit(p)
                self.changePixmapGray.emit(p2)
                self.changeHist.emit(cvc)
                self.changeEdge.emit(rgbImageGray)



    def test(self):
        pass


class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi('main+a.ui', self)

        self.th = Thread(self)

        self.action5_1.triggered.connect(self.openSecondForm)
        self.action_main.triggered.connect(self.openMainForm)
        self.action_SSSS.triggered.connect(self.openSSSSForm)
        self.action_RGB.triggered.connect(self.openRGBForm)
        self.action_BW.triggered.connect(self.openBWForm)
        self.action_color.triggered.connect(self.openGTCForm)
        self.action_Edge.triggered.connect(self.openEDGEForm)
        self.action_cameraYD.triggered.connect(self.openCameraHist)
        self.action_edgeSW.triggered.connect(self.openEdgeSW)
        self.action_rotation.triggered.connect(self.openRotation)


        otherview = SecondForm(self)
        self.stackedWidget.insertWidget(1, otherview)
        # self.stackedWidget.insertWidget(1, SecondForm(self))

        sssotherview = SSSSForm(self)
        self.stackedWidget.insertWidget(2, sssotherview)
        # self.atackedWidget.insertWidget(2, SSSSForm(self))

        # rgbotherview = RGBForm(self)
        # self.stackedWidget.insertWidget(3, rgbotherview)
        self.stackedWidget.insertWidget(3, RGBForm(self, self.th))

        blackwhiteview = BWForm(self)
        self.stackedWidget.insertWidget(4, blackwhiteview)
        # self.stackedWidget.insertWidget(4, BWForm(self))

        gitacolorview = GTCForm(self)
        self.stackedWidget.insertWidget(5, gitacolorview)
        # self.stackedWidget.insertWidget(5, GTCForm(self))

        # edgeview = EDGEForm(self)
        # self.stackedWidget.insertWidget(6, edgeview)
        self.stackedWidget.insertWidget(6, EDGEForm(self, self.th))

        # camerahistview = CameraHist(self, self.th)
        # self.stackedWidget.insertWidget(7, camerahistview)
        self.stackedWidget.insertWidget(7, CameraHist(self, self.th))

        self.stackedWidget.insertWidget(8, EdgeSW(self, self.th))

        self.stackedWidget.insertWidget(9, RotationForm(self, self.th))



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


        self.pushbutton_SAVE.setCheckable(True)
        self.pushbutton_SAVE.clicked.connect(self.recstart)



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

    def openGTCForm(self):
        self.stackedWidget.setCurrentIndex(5)

    def openEDGEForm(self):
        self.stackedWidget.setCurrentIndex(6)

    def openCameraHist(self):
        self.stackedWidget.setCurrentIndex(7)

    def openEdgeSW(self):
        self.stackedWidget.setCurrentIndex(8)

    def openRotation(self):
        self.stackedWidget.setCurrentIndex(9)

    def imageread(self):
        pass



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
        self.th.changePixmap.connect(self.setImage)
        self.th.changePixmapGray.connect(self.setImageGray)
        self.th.start()

    def recstart(self, state):
        global writer
        if state:
            fps = 29.97
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            delay = round(1000 / fps)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter('test.avi', fourcc, fps, size)
            self.th.recordFrame.connect(self.recording)
        else:
            writer.release()
            self.th.recordFrame.disconnect(self.recording)
            print('Recording Stop')

    def recording(self, frame):
        writer.write(frame)
        print('Recording.')
        print('Recording..')
        print('Recording...')

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