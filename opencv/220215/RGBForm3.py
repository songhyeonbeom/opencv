from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *













class RGBForm(QWidget):
    def __init__(self, parent=None, thread=None):
        super(RGBForm, self).__init__(parent)
        loadUi('mainRGB+.ui', self)


        self.parent = parent
        self.thread = thread
        self.thread.changePixmap.connect(self.setImage)

    def setImage(self, image):
        self.label_ORG.setPixmap(QPixmap.fromImage(image))













































































