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

from cnn import FeatureExtractor

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

def get_feature_extractor(cnn_model, frame_dim=(224, 224)): #fix to allow frame_dim tuple (H, W)
    """Returns keras CNN architecture to use as feature extractor"""

    base_models = FeatureExtractor(frame_dim[0], frame_dim[1])

    if cnn_model == "vgg16":
        return base_models.VGG16()
    
    elif cnn_model == "vgg19":
        return base_models.VGG19()
    
    elif cnn_model == "resnet50":
        return base_models.ResNet50()
    
    elif cnn_model == "resnet101":
        return base_models.ResNet101()

    elif cnn_model == "inception":
        return base_models.InceptionV3()

    
# def get_feature_representations(cnn, frames):
#     """Uses a CNN to extract feature representations from video frames dataset."""
   
#     #get the size of features outputted at last layer of cnn
#     num_videos = frames.shape[0]
#     frames_per_video = frames.shape[1]
#     feature_dim = cnn.layers[-1].output_shape[1] 
    
#     #init empty array to store features
#     features = np.empty((num_videos, frames_per_video, feature_dim), dtype=np.float32)
    
#     #get feature representations from each video's set of frames (fed as a batch to cnn)
#     for i, frame_batch in enumerate(frames):
#         features[i, ...] = cnn.predict_on_batch(next(iter(frame_batch)))
    
#     return features

def create_tf_dataset(videos, labels, batch_size):
    """Store video frames and labels in a tf.data.Dataset"""
    
    # we set CPU context to avoid memory alloc. errors
    # since tensorflow wants to copy dataset into GPU
    with tf.device("/device:CPU:0"):
        dataset = tf.data.Dataset.from_tensor_slices((videos, labels)).batch(batch_size)
        dataset = dataset.prefetch(2)
        
    return dataset


def feature_extraction_gpu(num_gpus, videos, video_labels, cnn_choice):
    
    #check for correct amoung of GPUs
    if num_gpus < 1:
        print(f"{num_gpus} GPUs is not enough to perform feature extraction in GPU mode. \
        Consider switching to CPU.")
        return
    
    # create TF strategy with selected num_gpus 
    print(f'Setting TensorFlow Mirrored Strategy with {num_gpus} GPUs...')
    active_gpus = select_gpus(num_gpus)
    strategy = tf.distribute.MirroredStrategy(active_gpus)
    print ('Number of devices in strategy: {}'.format(strategy.num_replicas_in_sync))
    
    # keep batch size at 1 to avoid memory alloc. issues since tensors are large
    BATCH_SIZE_PER_REPLICA = 1
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
    # store data in TF dataset with batch + prefetch
    print("Creating TF Dataset...")
    dataset = create_tf_dataset(videos, video_labels, GLOBAL_BATCH_SIZE)
    
    # creates a distributed dataset aligned with our TF strategy
    print("Creating TF DISTRIBUTED Dataset...")
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    
    # create a feature extractor on each active GPU in our TF strategy
    print("Creating CNN Model on Each Active Replica (GPU)")
    with strategy.scope():
        cnn_model = get_feature_extractor(cnn_choice)
        
    @tf.function
    def distributed_test_step(dataset_inputs):
        """ Distributes data inputs and invokes feature extraction on each GPU """
        return strategy.run(test_step, args=(dataset_inputs,))

    def test_step(inputs):
        """ Returns feature representations on inputs """
        images, label = inputs
        predictions = cnn_model(tf.squeeze(images), training=False)
        return predictions, label
        
    print("Beginning Feature Extraction in GPU Mode...")
    start = time.time()
    
    distributed_features = []
    distributed_labels = []
    for batch in dist_dataset:
        batch_features, batch_labels = distributed_test_step(batch)
        distributed_features.append(batch_features)
        distributed_labels.append(batch_labels)

    stop = time.time()
    print(f'Done getting video frame feature representations in {stop-start} seconds.')
    
    print("Formatting Results into Numpy Arrays...")
    features = replica_objects_to_numpy(distributed_features, num_gpus)
    labels = replica_objects_to_numpy(distributed_labels, num_gpus)
    
    return features, labels

def replica_objects_to_numpy(replica_results, num_gpus):
    """ Converts a list of TF Replica Objects into a Numpy ND array """
    
    if num_gpus > 1:
        # turn PerReplica objects from multi gpu's into list of tensors
        tensors = []
        for replica_obj in replica_results:
            tensors += list(replica_obj.values)

        #convert list of tensors to list of numpy arrays
        results = []
        for tensor in tensors:
            resuls.append(tensor.numpy())
            
    else:
        #convert list of tensors from single gpu to list of np arrays
        results = []
        for tensor in replica_results:
            results.append(tensor.numpy())
    
    return np.array(results)

def select_gpus(num_gpus):
    return [f'/GPU:{i}' for i in range(num_gpus)]


if __name__ == "__main__":
    #parse args here
    #[ADD CODE HERE...]
    
    # don't let TF takeup all the gpu memory
    limit_gpu_memory_growth()
    
    #read in dataframes
    X, y = load_dataframes(NGC_WORKSPACE + "downloaded_videos.csv")

    # get our data ready
    print('Loading data...')
    start = time.time()
    video_names = list(X.renamed_title)
    videos, video_labels = load_frames_and_labels(video_names) 
    stop = time.time()
    print(f"Done loading videos in {stop-start} seconds.")

    # get video frame feature representations with CNN    
    num_gpus = 1
    cnn_choice = "resnet101"
    features, labels = feature_extraction_gpu(num_gpus, videos, video_labels, cnn_choice)
    print(f"Back from feature Extraction.\nFeatures: {features.shape}\nLabels: {labels.shape}")


#     # split data
#     (X_train, y_train), (X_test, y_test) = get_data_splits(X, y, features, labels)

#     #train RNN
#     train(X_train, y_train)

#     #test RNN
#     test(X_test, y_test)