from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths
import argparse

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
from tabulate import tabulate

from cnn import FeatureExtractor
from rnn import attention
import wandb

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
            print(f"Memory growth is now the same across all {len(gpus)} available GPUs.")
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


def select_gpus(num_gpus):
    """ Create a list of GPU devices to use with TF Strategy """
    return [f'/GPU:{i}' for i in range(num_gpus)]


def create_tf_dataset(videos, labels, batch_size):
    """Store video frames and labels in a tf.data.Dataset"""
    
    # we set CPU context to avoid memory alloc. errors
    # since tensorflow wants to copy dataset into GPU
    with tf.device("/device:CPU:0"):
        dataset = tf.data.Dataset.from_tensor_slices((videos, labels)).batch(batch_size)
        dataset = dataset.prefetch(2)
        
    return dataset


def feature_extraction_cpu(videos, video_labels, cnn_choice):
    """Uses a CNN to extract feature representations from video frames dataset."""
   
    with tf.device("/device:CPU:0"):
        #create feature extractor
        cnn_model = get_feature_extractor(cnn_choice)

        #get the size of features outputted at last layer of cnn
        num_videos = videos.shape[0]
        frames_per_video = videos.shape[1]
        feature_dim = cnn_model.layers[-1].output_shape[1] 

        #init empty array to store features
        features = np.empty((num_videos, frames_per_video, feature_dim), dtype=np.uint8)

        #get feature representations from each video's set of frames (fed as a batch to cnn)
        start = time.time()
        for i, frame_batch in enumerate(videos):
            features[i, ...] = cnn_model.predict_on_batch(frame_batch)
        
        stop = time.time()
        duration = stop - start
        print(f'Done getting video frame feature representations in {stop-start} seconds (CPU mode).')
    
    return features, video_labels, duration


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

        #in the case that data does not divide evenly across GPUs,
        #TF creates a placeholder tensor so all GPUs have something to process
        #however this tensor will be empty and have shape (0, 461, 224, 224, 3)
        #so instead of running prediction, we return None values and filter these out later
        try:
            predictions = cnn_model(tf.squeeze(images), training=False)
            return predictions, label
        except ValueError:
            return None, None
        
    print("Beginning Feature Extraction in GPU Mode...")
    start = time.time()
    
    distributed_features = []
    distributed_labels = []
    for batch in dist_dataset:
        batch_features, batch_labels = distributed_test_step(batch)
        distributed_features.append(batch_features)
        distributed_labels.append(batch_labels)

    stop = time.time()
    duration = stop - start
    print(f'Done getting video frame feature representations in {stop-start} seconds.')
    
    print("Formatting Results into Numpy Arrays...")
    features = replica_objects_to_numpy(distributed_features, num_gpus)
    labels = replica_objects_to_numpy(distributed_labels, num_gpus)
    
    return features, labels, duration

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
            #filter out empty tensors generated during data distribution
            if tensor is not None: 
                results.append(tensor.numpy())
            
    else:
        #convert list of tensors from single gpu to list of np arrays
        results = []
        for tensor in replica_results:
            results.append(tensor.numpy())
    
    return np.array(results)


def get_data_splits(X, y, features, labels, batch_size=32):
    """ Create train, validation, and test TF datasets with batching + prefetching. 
    We use train_test_split to keep classes balanced in each split."""
    
    X_0, X_test, y_0, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_0, y_0, test_size = 0.20, random_state = 42)

    train_index = list(X_train.index)
    test_index = list(X_test.index)
    val_index = list(X_val.index)

    train_features, train_labels = features[train_index], labels[train_index]
    val_features, val_labels     = features[val_index], labels[val_index]
    test_features, test_labels   = features[test_index], labels[test_index]

    # reshape label arrays from (n_videos) to (n_videos, 1)
    train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
    val_labels = np.reshape(val_labels, (val_labels.shape[0], 1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))
    
    # Batch the input data
    buffer_size_train = train_features.shape[0]
    buffer_size_val = val_features.shape[0]
    buffer_size_test = test_features.shape[0]

    # Create Datasets from the batches
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).shuffle(buffer_size_train).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).shuffle(buffer_size_val).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).shuffle(buffer_size_test).batch(batch_size)

    train_dataset = train_dataset.prefetch(2)
    val_dataset = val_dataset.prefetch(2)
    test_dataset = test_dataset.prefetch(2)

    return train_dataset, val_dataset, test_dataset

def train_rnn(train_dataset, val_dataset, test_dataset, feature_dim):
    """ Create RNN model and run training and evaluation. """
    
    features_input       = keras.Input((461, feature_dim))
    x                    = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(features_input)
    x                    = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x)
    x                    = attention(return_sequences=False)(x)
    x                    = keras.layers.Dropout(0.2)(x)
    output               = keras.layers.Dense(2, activation="softmax")(x) #2 bc 2 class categories (0,1)
    model                = keras.Model(features_input, output)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    my_callbacks = [keras.callbacks.EarlyStopping(monitor="val_accuracy", 
                                                  patience=3,
                                                  mode="max",
                                                  min_delta = 0.01,
                                                  restore_best_weights=True)]

    # train model
    history = model.fit(train_dataset,
                        validation_data = val_dataset,
                        epochs = 15,
                        callbacks = my_callbacks,
                        verbose= 1)

    # evaluate trained model on test data
    loss, accuracy = model.evaluate(test_dataset)    
    f1 = get_F1_score(test_dataset, model)
    
    return history, loss, accuracy, f1

def get_F1_score(test_dataset, model):
    """ Get F1 Score for trained model """
    
    probabilities = model.predict(test_dataset)
    y_pred = np.array([np.argmax(p) for p in probabilities])
    y_true = np.concatenate([label for _, label in test_dataset], axis=0)

    return metrics.f1_score(y_true, y_pred)

def print_final_info(cnn, loss, accuracy, f1, duration):
    """ Print presentation info at process completion. """
    
    videos_per_sec = 364/duration
    frames_per_sec = (364*461)/duration
    row_data = [[cnn, accuracy, loss, f1, duration, videos_per_sec, frames_per_sec]]
    
    print(tabulate(row_data, headers = ["CNN","Accuracy (Test)", "Loss (Test)", "F1 Score", "Time to Extract Features (sec)", "Videos/Second (Feat. Ext.)", "Frames/Second (Feat. Ext.)"]))
    return

def main():
    #parse args here
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpu", default=0, type=int)
    parser.add_argument("--cnn_model", default="resnet101", type=str)
    parser.add_argument("--wandb_run", default="just another run", type=str)
    
    args = parser.parse_args()
    
    print(f"NUM GPUS: {args.num_gpu}")
    print(f"CNN: {args.cnn_model}")   
    print(f"wandb run name: {args.wandb_run}")   
    
    #start logging info
    os.environ["WANDB_API_KEY"] = "53f9fb8ccf6dd926fcfa46f72943e2d7c43494a9"
    wandb.login()
    wandb.init(project="whale-classification")
    wandb.run.name = args.wandb_run
    
    # don't let TF take up all the gpu memory
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
    if args.num_gpu >= 1:
        features, labels, feature_extraction_duration = feature_extraction_gpu(args.num_gpu, videos, video_labels, args.cnn_model)
    else:
        features, labels, feature_extraction_duration = feature_extraction_cpu(videos, video_labels, args.cnn_model)
        
    print(f"Back from feature Extraction.\nFeatures: {features.shape}\nLabels: {labels.shape}")

    # split data
    print("Splitting + batching features and labels for RNN ...")
    train_dataset, val_dataset, test_dataset = get_data_splits(X, y, features, labels)
    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    #train RNN
    print("Training RNN ...")
    history, loss, accuracy, f1 = train_rnn(train_dataset, val_dataset, test_dataset, features.shape[2])
    
    print_final_info(args.cnn_model, loss, accuracy, f1, feature_extraction_duration)
    wandb.finish()
    return

if __name__ == "__main__":
    main()
