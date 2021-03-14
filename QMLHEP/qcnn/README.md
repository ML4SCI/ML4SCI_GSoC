# Quantum Convolutional Neural Network Task

## Overview
Your task is to setup and apply a quantum convolutional neural network (QCNN) on particle physics data to perform binary classification on two types of objects (electrons and photons). You should use TFQ for this task.

## Dataset
This dataset [download link](./electron-photon.npz) contains 100 samples for training and another 100 for testing, laid out as follows:
- `data["x_train"]`: Training dataset of 100 32x32 images containing the particles' energy (100, 32, 32) 
- `data["y_train"]`:" Training labels, 0 = "photon", 1 = "electron" (100,)
- `data["x_test"]`: Test dataset of 100 32x32 images containing the particles' energy (100, 32, 32) 
- `data["y_test"]`:" Test labels, 0 = "photon", 1 = "electron" (100,)

The dataset labels are labelled 0 for photons and 1 for electrons.

## Task
Your task is to implement a QCNN model in Tensorflow Quantum that uses this dataset’s input and performs binary classification. Please feel free to experiment with different ways of encoding the classical data inputs into the qubits. 
Specifically, show that the model fits the dataset and that your training loss decreases over time. _(Given the small dataset size, we will not be focusing on the accuracy of your model)._
