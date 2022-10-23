import sys

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

from LeftKeyboard import LeftKeyboard

class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
  
        # set the title
        self.setWindowTitle("Main Menu")
  
        width = 1200
        height = 800

        self.button_style = "QPushButton {color: black; font-size: 20px; font-weight: bold;}"

        # setting  the fixed width of window
        self.setFixedWidth(width)
        self.setFixedHeight(height)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 250, 10, 10)
        main_layout.setSpacing(0)

        input_field = QLineEdit()
        input_field.setStyleSheet("QLineEdit {color: black; font-size: 20px; font-weight: bold; margin-bottom: 50px; padding: 10px;}")

        main_layout.addWidget(input_field)

        widget = QWidget()
        # layout = QGridLayout()

        keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
                "A", "S", "D", "F", "G", "H", "J", "K", "L",
                "Z", "X", "C", "V", "B", "N", "M",
                "Enter", "Space", "Delete"]

        # buttonList = []
        # for k in range(len(keys)):
        #     buttonList.append(QPushButton(keys[k]))
        #     buttonList[k].setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        #     buttonList[k].setStyleSheet(button_style)
        #     layout.addWidget(buttonList[k], k // 10, k % 10)

        layoutKeyboard = self.constructKeyboard(keys)

        main_layout.addLayout(layoutKeyboard)
        
        widget.setLayout(main_layout)

        self.setCentralWidget(widget)
  
        # show all the widgets
        self.show()

    def constructKeyboard(self, keys):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        leftKeyboard = QGridLayout()
        leftKeyboard.setContentsMargins(0, 150, 20, 150)

        buttons = []
        for k in range(len(keys) // 2 + 1):
            buttons.append(QPushButton(keys[k]))
            buttons[k].setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            buttons[k].setStyleSheet(self.button_style)
            leftKeyboard.addWidget(buttons[k], k // 10, k % 10)

        layout.addLayout(leftKeyboard)

        rightKeyboard = QGridLayout()
        rightKeyboard.setContentsMargins(20, 150, 0, 150)

        for k in range(len(keys) // 2 + 1, len(keys)):
            buttons.append(QPushButton(keys[k]))
            buttons[k].setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            buttons[k].setStyleSheet(self.button_style)
            rightKeyboard.addWidget(buttons[k], k // 10, k % 10)

        layout.addLayout(rightKeyboard)

        return layout


    def show_left_keyboard(self):
        self.left_keyboard = LeftKeyboard()
        self.left_keyboard.show()