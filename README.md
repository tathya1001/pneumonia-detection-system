# Pneumonia Detection System

Pneumonia is a serious illness that affects millions of people worldwide every year. Early diagnosis plays a crucial role in preventing severe complications. This repository contains a model that predicts the presence of pneumonia by analyzing chest X-ray images. The model uses a Convolutional Neural Network (CNN) architecture, which has been trained on a publicly available Chest X-ray Pneumonia dataset. The goal is to classify images into two categories: "Pneumonia" or "Normal."

## Dataset and Kaggle Code

The dataset used in this project is the [Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which contains labeled images of X-ray scans. The dataset is divided into three main subsets: training, validation, and testing.

- [Kaggle Code](https://www.kaggle.com/code/tathya1001/pneumonia-detection)
- [Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

## Model Architecture

The model is built using a Convolutional Neural Network (CNN) with the following architecture:

1. **Convolutional Layers**:
   - **Conv1**: 3x3 convolution, 8 output channels, followed by Batch Normalization, ReLU activation, and Max-Pooling with a 4x4 kernel.
   - **Conv2**: 3x3 convolution, 16 output channels, followed by Batch Normalization, ReLU activation, and Max-Pooling with a 4x4 kernel.
   - **Conv3**: 3x3 convolution, 32 output channels, followed by Batch Normalization, ReLU activation, and Max-Pooling with a 2x2 kernel.
   - **Conv4**: 3x3 convolution, 64 output channels, followed by Batch Normalization, ReLU activation, and Max-Pooling with a 2x2 kernel.

2. **Fully Connected Layers**:
   - **Lin1**: Fully connected layer with 128 output units, followed by Batch Normalization and ReLU activation.
   - **Lin2**: Fully connected layer with 64 output units, followed by Batch Normalization and ReLU activation.
   - **Lin3**: Fully connected layer with 2 output units (corresponding to the two classes: Pneumonia and Normal).

3. **Activation Function**:
   - The model uses the **ReLU** activation function after each convolutional and fully connected layer, except for the final output layer, which uses **softmax** for multi-class classification.

The model is trained using the **Adam optimizer** with a learning rate of 0.001, and the loss function used is **CrossEntropyLoss**, which is ideal for multi-class classification tasks.

## Training Configuration

- **Batch Size**: 64
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: CrossEntropyLoss

## Model Performance

The model achieved an accuracy of **78%** on the test dataset, demonstrating its ability to generalize to unseen data. On the training dataset, the model achieved a high accuracy of **99%**, showing that it has effectively learned the features of the chest X-rays in the training set.

## Sample Images from a Batch

Below is an image showing a batch of 64 sample images loaded for training:

![Sample Images from a Batch](image-link-to-sample-images.jpg)

## Predicted Labels for a Batch

Here are the predicted labels for the samples in a batch. The model outputs probabilities for each class, which are used to predict the labels:

![Predicted Labels](image-link-to-predicted-labels.jpg)
