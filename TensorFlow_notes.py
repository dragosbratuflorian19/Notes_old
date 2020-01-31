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
