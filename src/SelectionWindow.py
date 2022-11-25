from FaceMeshMethod import FaceMeshMethod
from CustomLandmarkDetection import CustomLandmarkDetection
from NNMethod import NNMethod

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

        # setting the main layout
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(10, 250, 10, 10)
        self.main_layout.setSpacing(0)

        # setting the left button
        self.left_button = QPushButton("Google Face Mesh\n(higher precision)")
        self.left_button.setStyleSheet(self.button_style)
        self.left_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.left_button.clicked.connect(self.left_button_clicked)

        # setting the center button
        self.center_button = QPushButton("NN Method\n(higher stability)")
        self.center_button.setStyleSheet(self.button_style)
        self.center_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.center_button.clicked.connect(self.center_button_clicked)

        # setting the right button
        self.right_button = QPushButton("Custom Landmark Detection\n(lower precision)")
        self.right_button.setStyleSheet(self.button_style)
        self.right_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.right_button.clicked.connect(self.right_button_clicked)

        # adding the buttons to the main layout
        self.main_layout.addWidget(self.left_button)
        self.main_layout.addWidget(self.center_button)
        self.main_layout.addWidget(self.right_button)

        # setting the main widget
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        self.show()

    # if left button is clicked
    def left_button_clicked(self):
        self.menu_window = FaceMeshMethod()
        self.menu_window.show()

    # if center button is clicked
    def center_button_clicked(self):
        self.menu_window = NNMethod()
        self.menu_window.show()

    # if right button is clicked
    def right_button_clicked(self):
        self.menu_window = CustomLandmarkDetection()
        self.menu_window.show()
        
