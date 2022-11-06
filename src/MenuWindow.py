from asyncio import sleep
import math
import sys
import cv2
import numpy as np
import dlib
from math import hypot
# from gaze_tracking import GazeTracking

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

            # we used the detector to detect the frontal face
            # self.detector = dlib.get_frontal_face_detector()

            # it will dectect the facial landwark points 
            # self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

            self.font = cv2.FONT_HERSHEY_PLAIN

            engine = pyttsx3.init()

            # to open webcab to capture the image
            cap = cv2.VideoCapture(0)

            self.frames = 0
            self.right_keyboard_selection_frames = 0
            self.left_keyboard_selection_frames = 0
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
    
            # show all the widgets
            # self.show()

            self.previous_direction = "none"
            amount_straight_events = 0

            while True:
                _, self.frame = cap.read()
                self.frame = cv2.flip(self.frame, 1)
                self.rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                # rows, cols, _ = self.frame.shape
                self.frames += 1

                #change the color of the frame captured by webcam to grey
                # self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                # to detect faces from grey color frame
                # faces = self.detector(self.gray)

                results = face_mesh.process(self.rgb_frame)
                
                #getting width and height of frame
                img_h, img_w = self.frame.shape[:2]

                # print(np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark]))
                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                # print('Test')

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
                # face_bottom_left_y = face_bottom_right[1]

                face_bottom_right_x = img_w - face_bottom_left[0] # bottom right corner of face
                # face_bottom_right_y = face_bottom_left[1] # bottom right corner of face

                face_top_right_x = img_w - face_top_left[0]
                # face_top_right_y = face_top_left[1]

                face_top_left_x = img_w - face_top_right[0]
                # face_top_left_y = face_top_right[1]

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
                    else:
                        self.input_text += keys_copy[0]
                    self.input_field.setText(self.input_text)
                    keys_copy = self.keys.copy()
                    self.keyboard_selected = "none"
                    self.right_keyboard_selection_frames = 0
                    self.left_keyboard_selection_frames = 0
                    
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

                        # sleep(0.5) # To avoid multiple selection of same key
                                
                elif normalized_position_iris_x < 0.01:
                    self.keyboard_selected = "left"

                        # sleep(0.5) # To avoid multiple selection of same key

                else:
                    amount_straight_events += 1
                    if amount_straight_events > 8:
                        # self.previous_direction = "none"
                        self.keyboard_selected = "none"
                        # self.left_keyboard_selection_frames = 0
                        # self.right_keyboard_selection_frames = 0
                        amount_straight_events = 0
                    print("none")

                if self.keyboard_selected == "right":
                    if self.previous_direction != self.keyboard_selected:
                        self.keyboard_selected = "right"
                        self.right_keyboard_selection_frames += 1
                        # If Kept gaze on one side more than 15 frames, move to keyboard
                        # if self.right_keyboard_selection_frames == 1:
                        print("right")
                        keys_copy = keys_copy[len(keys_copy) // 2:]
                        print(keys_copy)
                        self.frames = 0
                        self.right_keyboard_selection_frames = 0
                        self.left_keyboard_selection_frames = 0
                        
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
                        self.left_keyboard_selection_frames += 1
                        # If Kept gaze on one side more than 15 frames, move to keyboard
                        # if self.left_keyboard_selection_frames == 1:
                        print("left")
                        keys_copy = keys_copy[:len(keys_copy) // 2]
                        print(keys_copy)
                        self.frames = 0
                        self.left_keyboard_selection_frames = 0
                        self.right_keyboard_selection_frames = 0
                        
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
                # cv2.imshow("RGB Frame", self.rgb_frame)

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

        # layout.addLayout(leftKeyboard)
        layout.addWidget(leftKeyboard)

        # rightKeyboard = QGridLayout()
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

        # layout.addLayout(rightKeyboard)
        layout.addWidget(rightKeyboard)

        return layout, leftKeyboard, rightKeyboard

    def get_gaze_ratio(self, eye_points, facial_landmarks):
        # Gaze detection
        #getting the area from the frame of the left eye only
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                            (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                            (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                            (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                            (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                            (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
        
        #cv2.polylines(frame, [left_eye_region], True, 255, 2)
        height, width, _ = self.frame.shape
    
        #create the mask to extract xactly the inside of the left eye and exclude all the sorroundings.
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(self.gray, self.gray,mask = mask)
        
        #We now extract the eye from the face and we put it on his own window.Onlyt we need to keep in mind that wecan only cut
        #out rectangular shapes from the image, so we take all the extremes points of the eyes to get the rectangle
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = eye[min_y: max_y, min_x: max_x]
        
        #threshold to seperate iris and pupil from the white part of the eye.
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        
        #dividing the eye into 2 parts .left_side and right_side.
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)
        
        if left_side_white == 0:
            gaze_ratio = 1
            
        elif right_side_white == 0:
            gaze_ratio = 5
            
        else:
            gaze_ratio = left_side_white / right_side_white
        return(gaze_ratio)

    def eyes_contour_points(self, facial_landmarks):
        left_eye = []
        right_eye = []
        for n in range(36, 42):
            x = facial_landmarks.part(n).x
            y = facial_landmarks.part(n).y
            left_eye.append([x, y])
        for n in range(42, 48):
            x = facial_landmarks.part(n).x
            y = facial_landmarks.part(n).y
            right_eye.append([x, y])
        left_eye = np.array(left_eye, np.int32)
        right_eye = np.array(right_eye, np.int32)
        return left_eye, right_eye

    def midpoint(self, p1 ,p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
