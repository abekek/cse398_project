from asyncio import sleep
import math
import sys
import cv2
import numpy as np
import dlib
from math import hypot

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

import pyttsx3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.interactive(True)
import mediapipe as mp


LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.8
        ) as face_mesh:

            self.font = cv2.FONT_HERSHEY_PLAIN

            engine = pyttsx3.init()

            # to open webcab to capture the image
            cap = cv2.VideoCapture(0)

            # self.frames = 0
            # self.right_keyboard_selection_frames = 0
            # self.left_keyboard_selection_frames = 0
            self.keyboard_selected = "none"
    
            # set the title
            self.setWindowTitle("Main Menu")

            self.input_text = ""
    
            width = 1200
            height = 800

            self.button_style = "QPushButton {color: black; font-size: 20px; font-weight: bold;}"

            # setting  the fixed width of window
            self.setFixedWidth(width)
            self.setFixedHeight(height)

            self.main_layout = QVBoxLayout()
            self.main_layout.setContentsMargins(10, 250, 10, 10)
            self.main_layout.setSpacing(0)

            self.input_field = QLineEdit()
            self.input_field.setStyleSheet("QLineEdit {color: black; font-size: 20px; font-weight: bold; margin-bottom: 50px; padding: 10px;}")

            self.main_layout.addWidget(self.input_field)

            self.widget = QWidget()

            self.keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                    "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
                    "A", "S", "D", "F", "G", "H", "J", "K", "L",
                    "Z", "X", "C", "V", "B", "N", "M",
                    "Enter", "Space", "Delete"]

            self.layoutKeyboard, leftKeyboard, rightKeyboard = self.constructKeyboard(self.keys)

            leftKeyboard.setStyleSheet("QLabel {color: black; font-size: 20px; font-weight: bold; margin-bottom: 50px; padding: 10px;}")
            rightKeyboard.setStyleSheet("QLabel {color: black; font-size: 20px; font-weight: bold; margin-bottom: 50px; padding: 10px;}")

            self.main_layout.addLayout(self.layoutKeyboard)
            
            self.widget.setLayout(self.main_layout)

            self.setCentralWidget(self.widget)

            keys_copy = self.keys.copy()

            self.previous_direction = "none"
            amount_straight_events = 0

            while True:
                _, self.frame = cap.read()
                self.frame = cv2.flip(self.frame, 1)
                self.rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                # self.frames += 1

                results = face_mesh.process(self.rgb_frame)
                
                #getting width and height of frame
                img_h, img_w = self.frame.shape[:2]

                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                
                # turn center points into np array 
                center_left = np.array((l_cx, l_cy), dtype=np.int32)
                center_right = np.array((r_cx, r_cy), dtype=np.int32)

                cv2.circle(self.frame, tuple(center_left), int(l_radius), (255,0,255), 2, cv2.LINE_AA)
                cv2.circle(self.frame, tuple(center_right), int(r_radius), (255,0,255), 2, cv2.LINE_AA)

                face_bottom_left = mesh_points[149] # bottom left corner of face
                face_bottom_right = mesh_points[378]
                face_top_left = mesh_points[67]
                face_top_right = mesh_points[297] # top right corner of face
                
                face_bottom_left_x = img_w - face_bottom_right[0]
                face_bottom_right_x = img_w - face_bottom_left[0] # bottom right corner of face
                face_top_right_x = img_w - face_top_left[0]
                face_top_left_x = img_w - face_top_right[0]

                position_left_iris_x = img_w - l_cx
                position_right_iris_x = img_w - r_cx

                print('Left iris: ' + str(position_left_iris_x))
                print('Right iris: ' + str(position_right_iris_x))

                print('Top right X: ' +  str(face_top_right_x))
                print('Bottom left X: ' +  str(face_bottom_left_x))

                normalized_position_left_iris_x = self.normalize(position_left_iris_x, face_top_right_x, face_bottom_left_x)
                normalized_position_right_iris_x = self.normalize(position_right_iris_x, face_top_left_x, face_bottom_right_x)

                normalized_position_iris_x = (normalized_position_left_iris_x + normalized_position_right_iris_x) / 2.0

                print(normalized_position_iris_x)

                if len(keys_copy) == 1:
                    engine.say(keys_copy[0])
                    engine.runAndWait()
                    
                    if keys_copy[0] == "Delete":
                        self.input_text = self.input_text[:-1]
                    elif keys_copy[0] == 'Space':
                        self.input_text += " "
                    elif keys_copy[0] == 'Enter':
                        self.input_text += "\n"
                    else:
                        self.input_text += keys_copy[0]
                    
                    self.input_field.setText(self.input_text)
                    
                    keys_copy = self.keys.copy()
                    
                    self.keyboard_selected = "none"
                    
                    text = ""
                    for i in range(len(keys_copy) // 2):
                        text += keys_copy[i] + " "
                    leftKeyboard.setText(text)
                    
                    text = ""
                    for i in range(len(keys_copy) // 2, len(keys_copy)):
                        text += keys_copy[i] + " "
                    rightKeyboard.setText(text)

                
                if normalized_position_iris_x > 0.09:
                    self.keyboard_selected = "right"
                elif normalized_position_iris_x < 0.01:
                    self.keyboard_selected = "left"
                else:
                    amount_straight_events += 1
                    if amount_straight_events > 8:
                        self.keyboard_selected = "none"
                        amount_straight_events = 0
                    print("none")

                if self.keyboard_selected == "right":
                    if self.previous_direction != self.keyboard_selected:
                        self.keyboard_selected = "right"
                        print("right")
                        keys_copy = keys_copy[len(keys_copy) // 2:]
                        print(keys_copy)
                        
                        text = ""
                        for i in range(len(keys_copy) // 2):
                            text += keys_copy[i] + " "
                        leftKeyboard.setText(text)
                        
                        text = ""
                        for i in range(len(keys_copy) // 2, len(keys_copy)):
                            text += keys_copy[i] + " "
                        rightKeyboard.setText(text)

                        self.previous_direction = self.keyboard_selected

                elif self.keyboard_selected == "left":
                    if self.previous_direction != self.keyboard_selected:
                        self.keyboard_selected = "left"
                        print("left")
                        keys_copy = keys_copy[:len(keys_copy) // 2]
                        print(keys_copy)
                        
                        text = ""
                        for i in range(len(keys_copy) // 2):
                            text += keys_copy[i] + " "
                        leftKeyboard.setText(text)
                        
                        text = ""
                        for i in range(len(keys_copy) // 2, len(keys_copy)):
                            text += keys_copy[i] + " "
                        rightKeyboard.setText(text)

                        self.previous_direction = self.keyboard_selected

                elif self.keyboard_selected == "none":
                    self.previous_direction = self.keyboard_selected
                
                self.show()
                cv2.imshow("Frame", self.frame)

                key = cv2.waitKey(1)
                #close the webcam when escape key is pressed
                if key == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

    def normalize(self, val, maxVal, minVal):
        return max(0, min(1, (val - minVal) / (maxVal - minVal)))

    def constructKeyboard(self, keys):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # leftKeyboard = QGridLayout()
        leftKeyboard = QLabel()
        leftKeyboard.setContentsMargins(0, 150, 20, 150)

        buttons = []
        leftText = ""
        for k in range(len(keys) // 2 + 1):
            # buttons.append(QPushButton(keys[k]))
            # buttons[k].setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            # buttons[k].setStyleSheet(self.button_style)
            # leftKeyboard.addWidget(buttons[k], k // 10, k % 10)
            leftText += keys[k] + " "
            leftKeyboard.setText(leftText)

        layout.addWidget(leftKeyboard)

        rightKeyboard = QLabel()
        rightKeyboard.setContentsMargins(20, 150, 0, 150)

        rightText = ""
        for k in range(len(keys) // 2 + 1, len(keys)):
            # buttons.append(QPushButton(keys[k]))
            # buttons[k].setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            # buttons[k].setStyleSheet(self.button_style)
            # rightKeyboard.addWidget(buttons[k], k // 10, k % 10)
            rightText += keys[k] + " "
            rightKeyboard.setText(rightText)

        layout.addWidget(rightKeyboard)

        return layout, leftKeyboard, rightKeyboard