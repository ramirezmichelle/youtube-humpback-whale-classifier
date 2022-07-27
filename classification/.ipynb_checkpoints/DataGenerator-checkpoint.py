'''Generating a sequence of frames in batches involves 2 steps:
    1. Structure the data in sequence
    2. Feed the data to the generator
    
   Source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''

import numpy as np
import keras
from keras.utils import np_utils

#frame reading function
import sys
sys.path.insert(0, '/workspace/youtube-humpback-whale-classifier/video-download')
from hdf5_data_loading import read_frames_hdf5

class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, n_classes=2, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 461, 224, 224, 3))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            frames, frame_labels = read_frames_hdf5(ID)
            
            X[i, ...] = frames

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.np_utils.to_categorical(y, num_classes=self.n_classes)