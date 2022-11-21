from asyncio import sleep
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


LEFT_IRIS = [36, 37, 38, 39, 40, 41]
RIGHT_IRIS = [42, 43, 44, 45, 46, 47]

class CustomLandmarkDetection(QMainWindow):
    def __init__(self):
        super().__init__()

        # net =  cv2.dnn.readNetFromONNX('./saved_models/resnet18.onnx')
        # net =  cv2.dnn.readNetFromONNX('./saved_models/model.onnx')
        net =  cv2.dnn.readNetFromONNX('./saved_models/model_64.onnx')
        # net =  cv2.dnn.readNetFromONNX('./saved_models/model_64 (1).onnx')

        self.font = cv2.FONT_HERSHEY_PLAIN

        engine = pyttsx3.init()

        # to open webcab to capture the image
        cap = cv2.VideoCapture(0)

        self.keyboard_selected = "none"

        # set the title
        self.setWindowTitle("Handcrafted Method")

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
        self.input_field.setStyleSheet("QLineEdit {color: black; font-size: 30px; font-weight: bold; margin-bottom: 50px; padding: 10px;}")

        self.main_layout.addWidget(self.input_field)

        self.widget = QWidget()

        self.keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
                "A", "S", "D", "F", "G", "H", "J", "K", "L",
                "Z", "X", "C", "V", "B", "N", "M",
                "Enter", "Space", "Delete"]

        self.layoutKeyboard, leftKeyboard, rightKeyboard = self.constructKeyboard(self.keys)

        leftKeyboard.setStyleSheet("QLabel {color: black; font-size: 30px; border: 1px solid black; font-weight: bold; margin-bottom: 50px; padding: 10px;}")
        rightKeyboard.setStyleSheet("QLabel {color: black; font-size: 30px; border: 1px solid black; font-weight: bold; margin-bottom: 50px; padding: 10px;}")

        self.main_layout.addLayout(self.layoutKeyboard)
        
        self.widget.setLayout(self.main_layout)

        self.setCentralWidget(self.widget)

        keys_copy = self.keys.copy()

        self.previous_direction = "-1"
        amount_straight_events = 0

        count = 0

        detector = dlib.get_frontal_face_detector()

        while True:
            _, self.frame = cap.read()
            self.frame = cv2.flip(self.frame, 1)

            self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            rects = detector(self.image, 1)

            if len(rects) == 0:
                continue
        
            (x, y, self.w, self.h) = self.rect_to_bb(rects[0])
            # cv2.rectangle(self.frame, (x, y), (x + self.w, y + self.h), (0, 255, 0), 2)
            crop_img = self.image[y: y + self.h, x: x + self.w]

            blob = cv2.dnn.blobFromImage(crop_img, 1.0 / 255, (224, 224), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)

            results = net.forward()

            results = np.reshape(results, (68, 2))

            results += 0.5
            
            #getting width and height of frame
            # img_h, img_w = self.frame.shape[:2]

            left_results= results[LEFT_IRIS]
            right_results = results[RIGHT_IRIS]

            # for i, (x1, y1) in enumerate(results, 1):
            #     try:
            #         cv2.circle(self.frame, (int((x1 * self.w) + x), int((y1 * self.h) + y)), 2, [40, 117, 255], -1)
            #     except:
            #         pass

            points_left = np.array([np.multiply([p[0], p[1]], [self.w, self.h]).astype(int) for p in left_results])
            points_right = np.array([np.multiply([p[0], p[1]], [self.w, self.h]).astype(int) for p in right_results])

            points_left += [x, y]
            points_right += [x, y]

            cv2.polylines(self.frame, [points_left], True, (0, 0, 255), 2)
            cv2.polylines(self.frame, [points_right], True, (0, 0, 255), 2)

            # Detecting gaze to select left or right keybaord.
            gaze_ratio_left_eye = self.get_gaze_ratio(points_left)
            gaze_ratio_right_eye = self.get_gaze_ratio(points_right)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

            print(gaze_ratio)

            if count < 10:
                count += 1
                continue

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

            
            if gaze_ratio > 1.50:
                self.keyboard_selected = "right"
            elif gaze_ratio < 0.7:
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

            count += 1

        cap.release()
        cv2.destroyAllWindows()

    def get_gaze_ratio(self, eye_region):
        height, width, _ = self.frame.shape
    
        #create the mask to extract xactly the inside of the left eye and exclude all the sorroundings.
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_and(self.image, self.image, mask = mask)

        # cv2.imshow("eye", eye)
        
        #We now extract the eye from the face and we put it on his own window.Onlyt we need to keep in mind that wecan only cut
        #out rectangular shapes from the image, so we take all the extremes points of the eyes to get the rectangle
        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])
        gray_eye = eye[min_y: max_y, min_x: max_x]

        # cv2.imshow("gray_eye", gray_eye)
        
        #threshold to seperate iris and pupil from the white part of the eye.
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        cv2.imshow("threshold_eye", threshold_eye)
        
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


    def rect_to_bb(self, rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

    def normalize(self, val, maxVal, minVal):
        return max(0, min(1, (val - minVal) / (maxVal - minVal)))

    def constructKeyboard(self, keys):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        leftKeyboard = QLabel()
        leftKeyboard.setContentsMargins(0, 150, 20, 150)

        leftText = ""
        for k in range(len(keys) // 2 + 1):
            leftText += keys[k] + " "
            leftKeyboard.setText(leftText)

        layout.addWidget(leftKeyboard)

        rightKeyboard = QLabel()
        rightKeyboard.setContentsMargins(20, 150, 0, 150)

        rightText = ""
        for k in range(len(keys) // 2 + 1, len(keys)):
            rightText += keys[k] + " "
            rightKeyboard.setText(rightText)

        layout.addWidget(rightKeyboard)

        return layout, leftKeyboard, rightKeyboard