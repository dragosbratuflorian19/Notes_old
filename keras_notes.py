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