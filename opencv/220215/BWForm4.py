from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi






class BWForm(QWidget):
    def __init__(self, parent=None):
        super(BWForm, self).__init__(parent)
        loadUi('blackwhite.ui', self)
        self.button3.clicked.connect(self.display_image)

    # def gobBackToOtherForm(self):
    #     self.parent.stackedWidget.setCurrentIndex(0)

    def display_image(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file',
           '.',"Image Files (*.jpg *.gif *.bmp *.png)")
        pixmap = QPixmap(fname[0])
        self.mttyy01.setPixmap(pixmap)
        self.mttyy01.setScaledContents(True)
