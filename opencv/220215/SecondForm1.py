from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi




class SecondForm(QWidget):
    def __init__(self, parent=None):
        super(SecondForm, self).__init__(parent)
        loadUi('secondform.ui', self)
        self.button2.clicked.connect(self.gobBackToOtherForm)

    def gobBackToOtherForm(self):
        self.parent.stackedWidget.setCurrentIndex(0)

