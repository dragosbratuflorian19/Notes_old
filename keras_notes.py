# For large datasets, instalation of cuDNN (NVIDIAs Deep Neural Network library) is needed.
# Keras supports 3 different types of backends:
# - TensorFlow
# - Theano
# - CNTK
# For saving keras models on disk: HDF5 and h5py
# The samples for keras has to be a numpy array or a list of numpy arrays
# The labesl has to be a numpy array
# To scale data (from 0 to 1):
import numpy as np
from sklearn.preprocessing import MinMaxScaler
data = [23, 40, 12, 1, 0]
n_data = np.array(data)
scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform((n_data).reshape(-1,1))
print(scaled_data)
