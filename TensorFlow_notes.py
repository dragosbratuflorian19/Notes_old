#######################################################################################################
# Importing tensorflow
import tensorflow as tf
#######################################################################################################
# Creating a model
model = tf.keras.models.Sequential() # most common model
model.add(Dense(32, input_shape=(10,), activation='relu')) # The hidden layer: Dense is the most common one
model.add(Dense(2, activation='softmax')) # the output layer
#######################################################################################################
# Layers:
# Dense - fully connected Layers
# Convolutional - image processing
# Pooling layers
#######################################################################################################
# Activation functions
# Sigmoid
#               ~
# relu
#                         /
#                        /
#                       /
#                      /
#                     /
#                    /
# __________________/
#######################################################################################################
# Training the model: Optimizing the weights
# Stochastic Gradient Descent (SGD):
# - the most widely known optimizer
# - Objective: minimize the loss function (like mean squared error)
# cat 0.75, dog 0.25: error = 0 - 0.25 = 0.25
# Compiling the model
model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# or
model.loss = 'sparse_categorical_crossentropy'
model.optimizer.lr = 0.0001
# the optimizer = Adam
# loss function: mse mean squared error
# lr = learning rate between 0.01 and 0.0001
# metrics: what's printed out when the model it's trained
model.fit(train_samples, train_labels, batch_size=10, epochs=20, shuffle=True, verbose=2)
# batch_size = how many piceces of data to be sent to the model at once
# how much output we want to see when we train the model
#######################################################################################################
# The datasets used:
# - Training sets
# - Validation sets
# - Testing sets
# The input keras is expecting is a numpy array or a list of numpy arrays : array([[1,2,3], [1,2,3]])
# to chose the validation set:
model.fit(train_set, train_labels, validation_split=0.2, batch_size=10, epochs=20, shuffle=True, verbose=1)
# or we have explicitly the validation set:
model.fit(train_set, train_labels, validation_data=valid_set, batch_size=10, epochs=20, shuffle=True, verbose=1)
# and the valid_set has to be this format:
valid_set = [(sample, label), (sample, label) ... (sample, label)]
#######################################################################################################
# Making a prediction
predictions = model.predict(test_samples, batch_size=10, verbose=0)
#######################################################################################################
# Overfitting: good at classifying the train data, but not good at classifying the test data
# How do we know: when validation << training
# Fighting again overfitting: more data and data augmentation (crop, zoom, rotating, flipping)
# and reducing the complexity of the model (reducing layers or neurons from layers)
#######################################################################################################
# Underfitting: when training acc is low or loss is high
# Fighting again overfitting: increase the complexity of the model ( higher number of layers or neurons),
# adding more features to the data,
#######################################################################################################
# Supervised learning
train_samples = [[123, 21], [312, 43], [231, 12]]
train_labels = [1, 0, 1]
model.fit(x=train_samples, y=train_labels, batch_size=3, epochs=10, shuffle=True, verbose=2)
#######################################################################################################
# Supervised learning
# Labeled data
#######################################################################################################
# Unsupervised learning
# clustering
#######################################################################################################
# Semi-supervised learning
# When only part of the data is labeled
# Pseudo-labeling: Train with the labeled data -> label using the model the unlabeled data -> train the whole model with the new 100% labeled data
#######################################################################################################
# Encoding
# Cat, Dog, Lizard - > [x, x, x]
# Cat - > [1, 0, 0]
#######################################################################################################
# Convolutinal NN : for analyzing images
from tensorflow.keras.layers import Activation
from tensorflow.keras.convolutional import *
from tensorflow.keras.core import Dense, Flatten
# use of filters: Edge detector, squares, corners, circles
# Sliding through each 3x3 block of pixels: convolving through each block ( using a filter)
# Type of filters:
# -1: black, 1: white, 0: grey
# -1 -1 -1
#  1  1  1
#  0  0  0
#######################################################################################################
# Zero padding
# Initial    Padding
#
#            0 0 0 0 0
# 1 1 3      0 1 1 3 0
# 4 2 4      0 4 2 4 0
# 2 4 9      0 2 4 9 0
#            0 0 0 0 0

model = tf.keras.models.Sequential([
    Dense(16, activation='relu', input_shape(20, 20, 3)),
    Conv2D(32, kernel_size(3, 3), activation='relu', padding='valid'), # no padding ( default)
    Conv2D(64, kernel_size(5, 5), activation='relu', padding='same'), # padding
    Dense(2, activation='softmax')
])
#######################################################################################################
# Max pooling
from tensorflow.keras.pooling import *
# filter size = how much we pool
# stride = how much we move after we pool
model = tf.keras.models.Sequential([
    Dense(16, activation='relu', input_shape(20, 20, 3)),
    Conv2D(32, kernel_size(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size(2, 2), strides=2, padding='valid'),
    Conv2D(64, kernel_size(5, 5), activation='relu', padding='same'), # padding
    Dense(2, activation='softmax')
])
#######################################################################################################
# Back propagation



















