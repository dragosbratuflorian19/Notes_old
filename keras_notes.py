#######################################################################################################
# For large datasets, instalation of cuDNN (NVIDIAs Deep Neural Network library) is needed.
# Keras supports 3 different types of backends:
# - TensorFlow
# - Theano
# - CNTK
# For saving keras models on disk: HDF5 and h5py
# The samples for keras has to be a numpy array or a list of numpy arrays
# The labesl has to be a numpy array
#######################################################################################################
# To scale data (from 0 to 1):
import numpy as np
from sklearn.preprocessing import MinMaxScaler
data = [23, 40, 12, 1, 0]
n_data = np.array(data)
scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform((n_data).reshape(-1,1))
print(scaled_data)
#######################################################################################################
To visualize the model:
model.summary()
#######################################################################################################
# Creating a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_shape=(1,), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
#######################################################################################################
# Compiling a model
model.compile(tf.keras.optimizer.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#######################################################################################################
# Fitting a model
model.fit(scaled_data, train_labels, batch_size=10, shuffle=True, verbose=2)
#######################################################################################################
# Create a validation set
# 1st way
valid_set = [(sample, label), ... , (sample, label)]
model.fit(scaled_data, train_labels, validation_data=valid_set, batch_size=10, shuffle=True, verbose=2)
model.fit(scaled_data, train_labels, validation_split=0.1, batch_size=10, shuffle=True, verbose=2)
#######################################################################################################
# Make a prediction
# Classic prediction:
predictions = model.predict(test_data, batch_size=10, verbose=0)
# Rounded prediction:
rounded_prediction = model.predict_classes(test_data, batch_size=10, verbose=0)
# Confusion matrix to see the prediction accuracy
confusion_matrix from sklearn.metrics
#######################################################################################################
# Save a model classic:
model.save('my_model.h5')
# it saves:
# the architecture of the model
# the weights
# the training configuration(compile): loss, optimizer
# the state of the optimizer (resume training)
# Save a model as json string:
model.to_json()
# it saves only the architecture
#######################################################################################################
# Load the model:
new_model = tf.keras.models.load_model('my_model.h5')
or
new_model = tf.keras.models.model_from_json(json_string)
#######################################################################################################
# See the weights:
model.get_weights()
#######################################################################################################
# See the tree in CLI: $ tree
#######################################################################################################
# Prepare a CNN data(images)
import tensorflow as tf

train_path = 'cats and dogs/train'
valid_path = 'cats and dogs/valid'
test_path = 'cats and dogs/test'

train_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(train_path, target_size=(244, 244), classes=['dog', 'cat'], batch_size=10)
valid_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(valid_path, target_size=(244, 244), classes=['dog', 'cat'], batch_size=5)
test_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(test_path, target_size=(244, 244), classes=['dog', 'cat'], batch_size=5)
test_batches.class_indices # to see the indices : {'cat':0, 'dog': 1, 'lizard': 2} : [1. 0. 0.] -> cat
#######################################################################################################
# create and train the model
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)), # number of output filter, the kernel size ( convo window), hight/width/channel(RGB)
	tf.keras.layers.Flatten(), # used to flat the output of the convo layer into a 1D tensor --> then fed into the dense layer
	tf.keras.layers.Dense(2, activation='softmax'),
	])
model.compile(tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=5, validation_data=validation_batches, validation_steps=4, epochs=5, verbose=2)
# fit_generator - to fit the model batch by batch ( because of the ImageDataGenerator
# steps_per_epoch - total number of batches until a epoch is finished ( 50 / 10 = 5)
# validation steps - the same as for steps_per_epoch
#######################################################################################################
# Make a prediction
predictions = model.predict_generator(test_batches, steps=1, vervose=0)
#######################################################################################################
# Importing an already trained model
vgg16_model = tf.keras.applications.vgg16.VGG16()
# Because VGG is not a sequential model, we will take each layer and createa sequential model
model = tf.keras.models.Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
model.layers.pop() # Delete that last 1000 outputs layer
for layer in model.layers:
    layer.trainable = False
model.add(tf.keras.layers.Dense(2, activation='softmax'))
#######################################################################################################
# Data augmentation
from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, # 10 radians
                                                            width_shift_range=0.1, # 0.1 fraction of the entire width of the image
                                                            height_shift_range=0.1,
                                                            shear_range=0.15,
                                                            zoom_range=0.1,
                                                            channel_shift_range=10.,
                                                            horizontal_flip=True)
image_path = 'man.png'
image = np.expand_dims(ndimage.imread(image_path), 0) # expand_dims to be compatible later on
aug_iter = gen.flow(image) # generate batches of augmented images: takes the numpy data and generates back augmented data
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
#######################################################################################################
#