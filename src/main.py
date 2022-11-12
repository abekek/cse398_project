import sys

from PyQt6.QtWidgets import (
    QApplication,
)

from SelectionWindow import SelectionWindow

# create pyqt5 app
app = QApplication([])
  
# create the instance of our Window
window = SelectionWindow()
  
# start the app
sys.exit(app.exec())