import sys

from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QPushButton,
    QWidget,
    QMainWindow,
    QSizePolicy,
    QLabel,
)

class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
  
        # set the title
        self.setWindowTitle("Main Menu")
  
        width = 800
        height = 800

        button_style = "QPushButton {color: black; font-size: 20px; font-weight: bold;}"

        # setting  the fixed width of window
        self.setFixedWidth(width)
        self.setFixedHeight(height)

        widget = QWidget()
        layout = QGridLayout()
        
        button_left = QPushButton("Left")
        button_left.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        button_left.setStyleSheet(button_style)
        layout.addWidget(button_left, 0, 0)

        button_right = QPushButton("Right")
        button_right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        button_right.setStyleSheet(button_style)
        layout.addWidget(button_right, 0, 1)
        
        widget.setLayout(layout)

        self.setCentralWidget(widget)
  
        # show all the widgets
        self.show()