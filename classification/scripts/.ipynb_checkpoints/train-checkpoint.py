from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import pickle
import glob
import cv2
import sys
import os
import time

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

# from cnn import FeatureExtractor

import h5py
from pathlib import Path

NGC_WORKSPACE = '/mount/data/'
HDF5_DIR = Path(NGC_WORKSPACE + "frames_hdf5/")


def limit_gpu_memory_growth():
    """Function to limit gpu memory growth. Prevents TensorFlow 
    from taking up all GPU memory available.
    """

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth is now the same across all {len(gpus)} GPUs.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def load_dataframes(dataset_path):
    """Read in dataframe from .csv file and separate labels from actual data
    """
    
    #load dataset in
    data = pd.read_csv(dataset_path)
    y = data.pop('relevant')
    X = data
    
    return X, y

def get_video_data(video_clip_title):
    """ Reads video frames in from HDF5 file.
        Yields numpy array of shape (461, 224, 224, 3)
    """
    file = h5py.File(HDF5_DIR / f"{video_clip_title}.h5", "r+")
    
    # the video's overall label is stored with each frame, 
    # so we can just grab the first one since all labels are the same
    video_frames = np.array(file["/images"]).astype("uint8")
    video_label = np.array(file["/meta"]).astype("uint8")[0]
    
    return video_frames, video_label


def load_frames_and_labels(video_names):

    """Gets video frames and video labels ready. 
    Returns
    ----------------------------------------------
       frames    list of N generator objects representing each video's frames to avoid taking up memory
                 and speed up retrieval/loading
        
       labels    (N, 1) numpy array of each video's label (0 for `irrelevant` and 1 for `relevant`)
    """
    # convert names from video_xyz to video_clip_xyz
    clip_names = [video.replace("_", "_clip_").replace(".mp4", "") for video in video_names]
    
    # preallocate arrays to store video frames and labels
    N = len(video_names)
    videos = np.empty((N, 461, 224, 224, 3), dtype=np.uint8)
    labels = np.empty(N, dtype = np.uint8)
    
    # begin reading data in
    for i, clip_name in enumerate(clip_names):
        if i % 50 == 0:
            print(f"Video {i} ...")
        frames, label = get_video_data(clip_name) 
        videos[i, ...] = frames
        labels[i] = label 

    return videos, labels

# def get_feature_representations(cnn, frames):
#     """Uses a CNN to extract feature representations from video frames dataset."""
   
#     #get the size of features outputted at last layer of cnn
#     feature_dim = cnn.layers[-1].output_shape[1] 
#     num_videos = frames.shape[0]
#     frames_per_video = 461
    
#     #init empty array to store features
#     features = np.empty((num_videos, frames_per_video, feature_dim), dtype=np.float32)
    
#     #get feature representations from each video's set of frames (fed as a batch to cnn)
#     for i, frame_batch in enumerate(frames):
#         features[i, ...] = cnn.predict_on_batch(next(iter(frame_batch)))
    
#     return features

# def get_feature_extractor(cnn_model, frame_dim=(224, 224)): #fix to allow frame_dim tuple (H, W)
#     """Returns keras CNN architecture to use as feature extractor"""

#     base_models = FeatureExtractor(frame_dim)

#     if cnn_model == "vgg16":
#         return base_models.VGG16()
    
#     elif cnn_model == "vgg19":
#         return base_models.VGG19()
    
#     elif cnn_model == "resnet50":
#         return base_models.ResNet50()
    
#     elif cnn_model == "resnet101":
#         return base_models.ResNet101()

#     elif cnn_model == "inception":
#         return base_models.InceptionV3()


# def get_data_splits(X, y, features, labels):
#     """Splits features and labels into train and test sets"""
#     X_train, X_test, _ , _ = train_test_split(X, y, test_size = 0.20, random_state = 42)

#     train_index, test_index = list(X_train.index), list(X_test.index)

#     train_features, train_labels = features[train_index], labels[train_index]
#     test_features, test_labels = features[test_index], labels[test_index]

#     train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
#     test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

#     return (train_features, train_labels), (test_features, test_labels)

# def train(X_train, y_train): #Consider separating model architecture + placing in rnn file
#     """Train the RNN model using the video frame feature representations"""
#     features_input       = keras.Input((461, 512))
#     x                    = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(features_input)
#     x                    = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x)
#     x                    = attention(return_sequences=False)(x)
#     x                    = Dropout(0.2)(x)
#     output               = keras.layers.Dense(2, activation="softmax")(x) #2 bc 2 class categories (0,1)
#     model                = keras.Model(features_input, output)

#     model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#     my_callbacks    = [keras.callbacks.EarlyStopping(monitor="val_accuracy", 
#                                                     patience=3,
#                                                     mode="max",
#                                                     min_delta = 0.01,
#                                                     restore_best_weights=True)]
#     history = model.fit(X_train, 
#                         y_train,
#                         validation_split = 0.2,
#                         epochs = 15,
#                         callbacks = my_callbacks,
#                         verbose= 1)

#     print('Done training.')

# def test(X_test, y_test):
#     """Test our model on our test set"""
#     loss, accuracy = model.evaluate(X_test, y_test)
#     print(f"Test Metrics - Loss: {loss}, Accuracy: {accuracy}")

if __name__ == "__main__":
    #run commands here
    
    # don't let TF takeup all the gpu memory
#     limit_gpu_memory_growth()
    
    #read in dataframes
    X, y = load_dataframes(NGC_WORKSPACE + "downloaded_videos.csv")

    # get our data ready
    print('loading data...')
    video_names = list(X.renamed_title)
    videos, labels = load_frames_and_labels(video_names) 
    print(videos.shape)
    print(labels.shape)

#     # get video frame feature representations with CNN
#     start = time.time()
#     feature_extractor = get_feature_extractor("resnet101")
#     features = get_feature_representations(feature_extractor, videos)
#     stop = time.time()
#     print(f'Done getting video frame feature representations in {stop-start} seconds.')

#     # split data
#     (X_train, y_train), (X_test, y_test) = get_data_splits(X, y, features, labels)

#     #train RNN
#     train(X_train, y_train)

#     #test RNN
#     test(X_test, y_test)