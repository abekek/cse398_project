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

from LeftKeyboard import LeftKeyboard

class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # we used the detector to detect the frontal face
        self.detector = dlib.get_frontal_face_detector()

        # it will dectect the facial landwark points 
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.font = cv2.FONT_HERSHEY_PLAIN

        # to open webcab to capture the image
        cap = cv2.VideoCapture(0)

        self.frames = 0
        self.keyboard_selection_frames = 0
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

        self.layoutKeyboard = self.constructKeyboard(self.keys)

        self.main_layout.addLayout(self.layoutKeyboard)
        
        self.widget.setLayout(self.main_layout)

        self.setCentralWidget(self.widget)

        keys_copy = self.keys.copy()
  
        # show all the widgets
        # self.show()

        while True:
            _, self.frame = cap.read()
            # rows, cols, _ = self.frame.shape
            self.frames += 1

            #change the color of the frame captured by webcam to grey
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # to detect faces from grey color frame
            faces = self.detector(self.gray)
            for face in faces:
                
                #to detect the landmarks of a face
                landmarks = self.predictor(self.gray, face)
                
                left_eye, right_eye = self.eyes_contour_points(landmarks)
                
                # Eyes color
                # right now colo red around eyes cause we are not blinking them
                cv2.polylines(self.frame, [left_eye], True, (0, 0, 255), 2)
                cv2.polylines(self.frame, [right_eye], True, (0, 0, 255), 2)
                
                #Detecting gaze to select left or right keybaord.
                gaze_ratio_left_eye = self.get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
                gaze_ratio_right_eye = self.get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
                gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

                print(gaze_ratio)

                if len(keys_copy) == 1:
                    self.input_text += keys_copy[0]
                    self.input_field.setText(self.input_text)
                    keys_copy = self.keys.copy()
                    self.keyboard_selected = "none"
                    self.keyboard_selection_frames = 0
                    self.layoutKeyboard = self.constructKeyboard(keys_copy)
                    self.main_layout.itemAt(1).setParent(None)
                    # self.main_layout.addLayout(self.layoutKeyboard)
                
                if gaze_ratio <= 0.5:
                    self.keyboard_selected = "right"
                    self.keyboard_selection_frames += 1
                    # If Kept gaze on one side more than 15 frames, move to keyboard
                    if self.keyboard_selection_frames == 20:
                        print("left")
                        keys_copy = keys_copy[:len(self.keys) // 2]
                        self.layoutKeyboard = self.constructKeyboard(keys_copy)
                        self.frames = 0
                        self.keyboard_selection_frames = 0
                        self.main_layout.itemAt(1).setParent(None)
                        self.main_layout.addLayout(self.layoutKeyboard)
                                
                elif gaze_ratio >= 1.0:
                    self.keyboard_selected = "left"
                    self.keyboard_selection_frames += 1
                    # If Kept gaze on one side more than 15 frames, move to keyboard
                    if self.keyboard_selection_frames == 20:
                        print("right")
                        keys_copy = keys_copy[len(keys_copy) // 2:]
                        self.layoutKeyboard = self.constructKeyboard(keys_copy)
                        self.frames = 0
                        self.keyboard_selection_frames = 0
                        self.main_layout.itemAt(1).setParent(None)
                        self.main_layout.addLayout(self.layoutKeyboard)

                else:
                    self.keyboard_selected = "none"
                    self.keyboard_selection_frames = 0
                    print("none")
            
            self.show()
            cv2.imshow("Frame", self.frame)

            key = cv2.waitKey(1)
            #close the webcam when escape key is pressed
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

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


    def show_left_keyboard(self):
        self.left_keyboard = LeftKeyboard()
        self.left_keyboard.show()