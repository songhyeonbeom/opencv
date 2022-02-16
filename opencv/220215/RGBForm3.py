from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi








class RGBForm(QWidget):
    def __init__(self, parent=None):
        super(RGBForm, self).__init__(parent)
        loadUi('mainRGB+.ui', self)
        self.button5.clicked.connect(self.gobBackToOtherForm)

    def gobBackToOtherForm(self):
        self.parent.stackedWidget.setCurrentIndex(0)


