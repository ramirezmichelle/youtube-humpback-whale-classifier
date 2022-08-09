import tensorflow as tf
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

NGC_WORKSPACE = '/mount/data/'
HDF5_DIR = Path(NGC_WORKSPACE + "frames_hdf5/")

from sklearn.model_selection import train_test_split

def load_dataframes(dataset_path = "/mount/data/downloaded_videos.csv"):
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


def split_video_dataset(X, y, videos, labels):
    """ Takes numpy video frames and labels and uses sklearn's train_test_split to generate
    a train, validation, and test dataset, each with balanced class proportions """
    X_0, X_test, y_0, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_0, y_0, test_size = 0.20, random_state = 42)

    train_index = list(X_train.index)
    test_index = list(X_test.index)
    val_index = list(X_val.index)

    train_videos, train_labels = videos[train_index], labels[train_index]
    val_videos, val_labels     = videos[val_index], labels[val_index]
    test_videos, test_labels   = videos[test_index], labels[test_index]
    
    with tf.device("/device:CPU:0"):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_videos, train_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_videos, val_labels))    
        test_dataset = tf.data.Dataset.from_tensor_slices((test_videos, test_labels))
        
    return train_dataset, val_dataset, test_dataset