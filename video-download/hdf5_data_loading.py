import cv2
import pandas as pd
import numpy as np
from decord import VideoReader
from decord import cpu, gpu
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

workspace_path = '/mount/data'
hdf5_dir = Path(workspace_path + "/frames_hdf5/")

def store_frames_hdf5(images, labels, video_clip_title):
    """
    Stores an array of images to HDF5.
    Parameters:
    ---------------
    images       images array, (N, 32, 32, 3) to be stored
    labels       labels array, (N, 1) to be stored
    """
    
    #create new hdf5 file
    file = h5py.File(hdf5_dir / f"{video_clip_title}.h5", "w")
    
    #create a dataset in the file
    dataset = file.create_dataset("images", np.shape(images), h5py.h5t.STD_U8BE, data=images)
    meta_set = file.create_dataset("meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels)
    
    file.close()
    
def read_frames_hdf5(video_clip_title):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{video_clip_title}.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

def get_video_frames(video_clip_title, max_frames, context=cpu(0), resize=(224,224)):
    ''' Get individual image frames from video '''
    
    #get clip number for frame naming
    clip_number = video_clip_title.split('_')[2].split('.')[0]
    
    #read video
    vr = VideoReader(workspace_path + '/video_clips/' + video_clip_title, ctx=context, width=resize[0], height=resize[1])
    
    #get batch of frames that matches amount needed
    frame_indices, frames = get_n_frames(vr, max_frames)

    #convert frames to numpy array and save in hdf5 file
    frames_numpy = frames.asnumpy()
    
    return frame_indices, frames_numpy
    

def get_n_frames(all_frames, max_frames):
    ''' Get the needed number of frames (max_frames)'''
    
    if len(all_frames) > max_frames:
        #only sample the needed amt of frames
        frame_indices = sample_n_frames(len(all_frames), max_frames)
        frames = all_frames.get_batch(frame_indices)
    
    elif len(all_frames) < max_frames:
        #pad frames
        frame_indices = pad_frames(len(all_frames), max_frames)
        frames = all_frames.get_batch(frame_indices)
   
    else:
        frame_indices = [i for i in range(len(all_frames))]
        frames = all_frames.get_batch(frame_indices)
        
    return frame_indices, frames
    

def sample_n_frames(num_available, n):
    ''' Function to pick out n frames from total frames available
    Inputs
        num_available (int): number of frames available
        n (int): number of frames we need
    
    Outputs
        frame_indices (int): indices of frames, including padded middle frame 
    '''
    
    if num_available <= n:
        print("Video does not have sufficient frames and does not need undersampling. Returning..")
        return False
    
    #get interval of how often to sample
    sampling_interval = int(np.floor(num_available/ n))
    
    #establish which indices we have to start out with
    frame_indices = [i for i in range(num_available)]
    
    #recursively get a list of evenly spread out frame indices across the video's duration
    undersampled_indices = recursive_undersample(frame_indices, n, sampling_interval)
            
    return undersampled_indices

def recursive_undersample(frame_indices, n, step):
    '''Recursively narrow down list of frame indices to n total frame indices'''
    
    if len(frame_indices) == n:
        return frame_indices
        
    #put together list of cut down (undersampled) frame indices
    num_available = len(frame_indices)

    if step == 1:
        undersampled_indices = [frame_indices[i] for i in range(0, n, step)]  
    else:
        undersampled_indices = [frame_indices[i] for i in range(0, num_available, step)]    

    new_step = int(np.floor(len(undersampled_indices)/n))
    
    return recursive_undersample(undersampled_indices, n, new_step)

    
def pad_frames(num_available, n):
    
    '''Function to pad videos that fall short of standardized frame count by replicating middle frame.
    Inputs
        num_available (int): number of frames available
        n (int): number of frames we need
    
    Outputs
        frame_indices (int): indices of frames, including padded middle frame 
    '''
    if num_available >= n:
        print("Video has sufficient frames and does not need padding. Returning..")
        return False
    
    #find how many times we need to replicate the middle frame
    num_frames_needed = n - num_available
    
    #find the middle frame index
    mid_frame_idx = num_available // 2
    
    #replicate mid_frame_idx as many times as needed 
    padded = [mid_frame_idx for i in range(num_frames_needed)]
    
    #include mid frame padding indices in list of final indices
    existing = [i for i in range(num_available)]
    final_frame_indices = existing + padded
    
    #sort index list so padded frame indices are in place
    final_frame_indices.sort()
    
    return final_frame_indices

if __name__ == '__main__':
    
    sequence_length = 461
    df = pd.read_csv(workspace_path + "/downloaded_videos.csv")
    
    print('<<<<START>>>>')

    for i, row in df.iterrows():
        
        if i % 50 == 0:
            print(f"Saving frames for video {i}...")
            
        #create array of relevance labels (one label per frame)
        frame_labels = [int(row['relevant']) for i in range(sequence_length)]
        
        #get name of video clip and filename for .h5 file
        video_clip = row['renamed_title'].replace("_", "_clip_")
        fn = video_clip.replace(".mp4", "")

        #get numpy frames and store in hdf5
        _, frames = get_video_frames(video_clip, max_frames=sequence_length)
        store_frames_hdf5(frames, frame_labels, fn)

    print('Done saving video frames to their HDF5 files.')
    
    
