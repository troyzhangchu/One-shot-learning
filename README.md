# slml2018
This repository includes codes for my course Statistical Learning and Machine Learning in Fudan University.These codes are used for my final project, which is about one shot learning.


# About My Paper
My paper of the course is called 'Embarrassingly Simple One Shot Learning and Siamese Neural Networks Based on Deep Features', which contains some of my trails on deep features extractors and one shot learning tasks. The methods used in my paper including ESOSL, which is a lazy learning way developed from embarrassingly simple zero shot learning (ESZSL) raised by Bernardino Romera-Paredes and Philip H S Torr in 2015, and Siamese NN, which is a very popular neural network used in transfer learning.


## 'models'
In the directory 'models', there are codes for several models used in my paper, including:

Image features extractors: ResNet50, DenseNet121, AlexNet and VGG16;

One-shot learning models: ESOSL, SiameseNN;

Linear classification models: SVR, kNN and Logistic Regression.


## 'utils'
These utils are used in most models, including funtions to get data, compute cross entropy loss and so on.


## Appreciation
Most of these models are completed by myself, including image features extractors, linear models and ESOSL.

And part of these codes are developed based on open source codes offered on github. Thanks to gidariss for Cosine-basedNN (https://github.com/gidariss/FewShotWithoutForgetting), yijiuzai for Matching Network (https://github.com/yijiuzai/Matching-Networks-for-One-Shot-Learning) and Lectures2codes for SiameseNN and some utils(https://github.com/Lectures2Code/slml/tree/master/models).
