# Mask-and-no-Mask-Detection

![image](https://user-images.githubusercontent.com/66860602/122380586-a7ef2600-cf85-11eb-9c95-810605956e36.png)

# Introduction :

We are given two datasets which contain images of people either wearing a mask
or not wearing a mask.
We have assigned labels to the classes of masked and unmasked and on the basis
of that we predicted the labels by applying the classifiers

# Requirements :

We are building CNN model for masked image detection and also use of open cv to
read and preprocess those images and hence some libraries are required to be pre
installed in your system or you can install by
1. pip install tensorflow
2. pip install keras
3. pip install opencv-python
4. pip install --user opencv-contrib-python

#Libraries required for Data Processing and Building the model:

1. Keras.models - Sequential
2. Keras.layers - Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D
3. Sklearn.metrics - accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix,roc_curve
4. Importing cv2 and os

# Classifiers Used :

1. Convolutional Neural Network (CNN)
2. SVM (linear and gaussian)
3. Logistic Regression
4. KNN

# Process of Execution :

1. Reading the directory having the images using os.
2. Converting images to the numpy array using cv.imread for reading and
converting into grayscale images.
3. Then we are creating data and target numpy arrays by resizing the image
using cv.resize.
4. Then we scale the images and on target we used to_categorical function from
np.utils.
5. We applied the CNN model, SVM model, KNN and Logistic Regression.


  ![](https://cdn-images-1.medium.com/max/1600/1*SKlPuk4vscYs3bl1bFdT5g.gif)
