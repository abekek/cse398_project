# kEyeBoard - an eye-gaze controlled keyboard for people with Motor Neuron Diseases

## Introduction

This project is a keyboard that can be controlled by eye-gaze. It is designed for people with Motor Neuron Diseases (MND) who have lost the ability to use their hands.

## Training/Inference Pipeline

![Training/Inference Pipeline](./public/img/kEyeBoard%20Steps.drawio.svg)

## Testing and Demo

The program was tested on Windows 11 laptop with an 7-10750H CPU @ 2.60GHz and an Nvidia GeForce GTX 1650 GPU.

To test the program, first install the dependencies:

```
conda env create -f environment.yml
```

And then run the the demo:

```
python main.py
```