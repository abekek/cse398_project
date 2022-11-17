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

import os
import torch
from torch.nn import DataParallel
from models.eyenet import EyeNet
import imutils
import util.gaze
from imutils import face_utils

from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample

import pyttsx3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.interactive(True)

torch.backends.cudnn.enabled = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class RightWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Network")
        width = 1200
        height = 800

        # setting  the fixed width of window
        self.setFixedWidth(width)
        self.setFixedHeight(height)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(0)

        self.input_field = QLineEdit()
        self.input_field.setStyleSheet("QLineEdit {color: black; font-size: 30px; font-weight: bold; margin-bottom: 50px; padding: 10px;}")

        self.main_layout.addWidget(self.input_field)

        self.widget = QWidget()

        engine = pyttsx3.init()

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

        webcam = cv2.VideoCapture(0)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        webcam.set(cv2.CAP_PROP_FPS, 60)

        dirname = os.path.dirname(__file__)
        face_cascade = cv2.CascadeClassifier(os.path.join(dirname, './saved_models/lbpcascade_frontalface_improved.xml'))
        self.landmarks_detector = dlib.shape_predictor(os.path.join(dirname, './saved_models/shape_predictor_5_face_landmarks.dat'))

        checkpoint = torch.load('./saved_models/checkpoint.pt', map_location=device)
        nstack = checkpoint['nstack']
        nfeatures = checkpoint['nfeatures']
        nlandmarks = checkpoint['nlandmarks']
        self.eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
        self.eyenet.load_state_dict(checkpoint['model_state_dict'])

        current_face = None
        landmarks = None
        alpha = 0.95
        left_eye = None
        right_eye = None

        self.previous_direction = "-1"
        amount_straight_events = 0
        self.keyboard_selected = "-1"

        self.input_text = ""
        self.counter = 0

        while True:
            sleep(0.5)
            _, frame_bgr = webcam.read()
            orig_frame = frame_bgr.copy()
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)

            if len(faces):
                next_face = faces[0]
                if current_face is not None:
                    current_face = alpha * next_face + (1 - alpha) * current_face
                else:
                    current_face = next_face

            if current_face is not None:
                next_landmarks = self.detect_landmarks(current_face, gray)

                if landmarks is not None:
                    landmarks = next_landmarks * alpha + (1 - alpha) * landmarks
                else:
                    landmarks = next_landmarks


            if landmarks is not None:
                eye_samples = self.segment_eyes(gray, landmarks)

                eye_preds = self.run_eyenet(eye_samples)
                left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
                right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

                if left_eyes:
                    left_eye = self.smooth_eye_landmarks(left_eyes[0], left_eye, smoothing=0.1)
                if right_eyes:
                    right_eye = self.smooth_eye_landmarks(right_eyes[0], right_eye, smoothing=0.1)

                for ep in [left_eye, right_eye]:
                    for (x, y) in ep.landmarks[16:33]:
                        color = (0, 255, 0)
                        if ep.eye_sample.is_left:
                            color = (255, 0, 0)
                        cv2.circle(orig_frame,
                                (int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)

                    gaze = ep.gaze.copy()
                    if ep.eye_sample.is_left:
                        gaze[1] = -gaze[1]
                    util.gaze.draw_gaze(orig_frame, ep.landmarks[-2], gaze, length=60.0, thickness=2)

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

                    print(ep.eye_sample.is_left)

                    if gaze[0] < 0 and gaze[1] < 0:
                        self.keyboard_selected = "left"
                        # self.previous_direction = "left"
                        # amount_straight_events = 0
                        print("left" + str(gaze))
                    elif gaze[0] > 0.2 and gaze[1] > 0:
                        self.keyboard_selected = "right"
                        # self.previous_direction = "right"
                        # amount_straight_events = 0
                        print("right" + str(gaze))
                    else:
                        amount_straight_events += 1
                        if amount_straight_events > 5:
                            self.keyboard_selected = "none"
                            # self.previous_direction = "none"
                            print("none" + str(gaze))

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


            self.counter += 1
            self.show()
            cv2.imshow("Webcam", orig_frame)
            key = cv2.waitKey(1)

            if key == 27:
                break
        
        webcam.release()
        cv2.destroyAllWindows()


    def detect_landmarks(self, face, frame, scale_x=0, scale_y=0):
        (x, y, w, h) = (int(e) for e in face)
        rectangle = dlib.rectangle(x, y, x + w, y + h)
        face_landmarks = self.landmarks_detector(frame, rectangle)
        return face_utils.shape_to_np(face_landmarks)


    def draw_cascade_face(self, face, frame):
        (x, y, w, h) = (int(e) for e in face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


    def draw_landmarks(self, landmarks, frame):
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)


    def segment_eyes(self, frame, landmarks, ow=160, oh=96):
        eyes = []

        # Segment eyes
        for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
            x1, y1 = landmarks[corner1, :]
            x2, y2 = landmarks[corner2, :]
            eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
            if eye_width == 0.0:
                return eyes

            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

            # center image on middle of eye
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            inv_translate_mat = np.asmatrix(np.eye(3))
            inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

            # Scale
            scale = ow / eye_width
            scale_mat = np.asmatrix(np.eye(3))
            scale_mat[0, 0] = scale_mat[1, 1] = scale
            inv_scale = 1.0 / scale
            inv_scale_mat = np.asmatrix(np.eye(3))
            inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

            estimated_radius = 0.5 * eye_width * scale

            # center image
            center_mat = np.asmatrix(np.eye(3))
            center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
            inv_center_mat = np.asmatrix(np.eye(3))
            inv_center_mat[:2, 2] = -center_mat[:2, 2]

            # Get rotated and scaled, and segmented image
            transform_mat = center_mat * scale_mat * translate_mat
            inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

            eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
            eye_image = cv2.equalizeHist(eye_image)

            if is_left:
                eye_image = np.fliplr(eye_image)
                cv2.imshow('left eye image', eye_image)
            else:
                cv2.imshow('right eye image', eye_image)
            eyes.append(EyeSample(orig_img=frame.copy(),
                                img=eye_image,
                                transform_inv=inv_transform_mat,
                                is_left=is_left,
                                estimated_radius=estimated_radius))
        return eyes


    def smooth_eye_landmarks(self, eye, prev_eye, smoothing=0.2, gaze_smoothing=0.4):
        if prev_eye is None:
            return eye
        return EyePrediction(
            eye_sample=eye.eye_sample,
            landmarks=smoothing * prev_eye.landmarks + (1 - smoothing) * eye.landmarks,
            gaze=gaze_smoothing * prev_eye.gaze + (1 - gaze_smoothing) * eye.gaze)


    def run_eyenet(self, eyes, ow=160, oh=96):
        result = []
        for eye in eyes:
            with torch.no_grad():
                x = torch.tensor([eye.img], dtype=torch.float32).to(device)
                _, landmarks, gaze = self.eyenet.forward(x)
                landmarks = np.asarray(landmarks.cpu().numpy()[0])
                gaze = np.asarray(gaze.cpu().numpy()[0])
                assert gaze.shape == (2,)
                assert landmarks.shape == (34, 2)

                landmarks = landmarks * np.array([oh/48, ow/80])

                temp = np.zeros((34, 3))
                if eye.is_left:
                    temp[:, 0] = ow - landmarks[:, 1]
                else:
                    temp[:, 0] = landmarks[:, 1]
                temp[:, 1] = landmarks[:, 0]
                temp[:, 2] = 1.0
                landmarks = temp
                assert landmarks.shape == (34, 3)
                landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
                assert landmarks.shape == (34, 2)
                result.append(EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze))
        return result


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