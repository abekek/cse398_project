from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QPushButton,
    QWidget,
    QMainWindow,
    QSizePolicy,
    QLabel,
)

class LeftKeyboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Left Keyboard")
        
        button_style = "QPushButton {color: black; font-size: 20px; font-weight: bold;}"

        width = 800
        height = 800

        # setting  the fixed width of window
        self.setFixedWidth(width)
        self.setFixedHeight(height)

        widget = QWidget()
        layout = QGridLayout()
 
        widget.setLayout(layout)
 
        self.setCentralWidget(widget)
 
        # show all the widgets
        self.show()