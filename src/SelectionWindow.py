from MenuWindow import MenuWindow
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QPushButton,
    QWidget,
    QMainWindow,
    QSizePolicy,
    QLabel,
    QVBoxLayout,
    QLineEdit,
    QHBoxLayout,
)

class SelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Selection Window")
        width = 1200
        height = 800

        self.button_style = "QPushButton {color: black; font-size: 20px; font-weight: bold;}"

        # setting  the fixed width of window
        self.setFixedWidth(width)
        self.setFixedHeight(height)

        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(10, 250, 10, 10)
        self.main_layout.setSpacing(0)

        self.left_button = QPushButton("Calibration-Free\n(lower precision)")
        self.left_button.setStyleSheet(self.button_style)
        self.left_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.left_button.clicked.connect(self.left_button_clicked)

        self.right_button = QPushButton("Calibration\n(higher precision)")
        self.right_button.setStyleSheet(self.button_style)
        self.right_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.right_button.clicked.connect(self.right_button_clicked)

        self.main_layout.addWidget(self.left_button)
        self.main_layout.addWidget(self.right_button)

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        self.show()

    def left_button_clicked(self):
        # open the menu window
        self.menu_window = MenuWindow()
        self.menu_window.show()

    def right_button_clicked(self):
        print("Right button clicked")
        
