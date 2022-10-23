import sys

from PyQt6.QtWidgets import (
    QApplication,
)

from MenuWindow import MenuWindow

# create pyqt5 app
app = QApplication([])
  
# create the instance of our Window
window = MenuWindow()
  
# start the app
sys.exit(app.exec())