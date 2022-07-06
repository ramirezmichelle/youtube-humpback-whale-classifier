import cv2
import numpy as np
from decord import VideoReader
from decord import cpu, gpu
import matplotlib.pyplot as plt

workspace_path = '/mount/data'

def get_video_frames(video_title, max_frames, context=cpu(0), resize=(224,224)):
    ''' Get individual image frames from video '''
    
    #get clip number for frame naming
    clip_number = video_title.split('_')[2].split('.')[0]
    
    #read video
    vr = VideoReader(workspace_path + '/video_clips/' + video_title, ctx=context)
    
    #get batch of frames that matches amount needed
    frames = get_n_frames(vr, len(vr))

    #save frames as jpg images 
    for i in range(len(frames)):
        frame = frames[i].asnumpy()
        frame = cv2.resize(frame, resize)
        
        #reorder color channels (will leave out for now)
        #frame = frame[:, :, [2, 1, 0]] 
        
        #save frame image in directory
        plt.imsave(workspace_path + "/frames/" + "/clip_%s_frame_%d.jpg" % (clip_number, i), frame)
    
    return len(frames), len(vr)
    

def get_n_frames(video_reader, max_frames):
    ''' Get the needed number of frames (max_frames)'''
    
    if len(video_reader) > max_frames:
        #only sample the needed amt of frames
        frame_indices = sample_n_frames(len(video_reader), max_frames)
        frames = video_reader.get_batch(frame_indices)
    
    elif len(video_reader) < max_frames:
        #pad frames
        frame_indices = pad_frames(len(video_reader), max_frames)
        frames = video_reader.get_batch(frame_indices)
   
    else:
        frames = video_reader
        
    return frames
    

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
    sample_interval = num_available // n
    
    #put together list of cut down (undersampled) frame indices
    undersampled_indices = [i for i in range(0, num_available, sample_interval)]
    
    #check if we fall short of n total frames (due to uneven division) -
    #if so, pad the middle frame
    if len(undersampled_indices) < n:
        num_needed = n - len(undersampled_indices)
        mid_idx = len(undersampled_indices) // 2
        
        padded = [undersampled_indices[mid_idx] for i in range(num_needed)]
        undersampled_indices += padded
        undersampled_indices.sort()
        
            
    return undersampled_indices
    
    
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

# def standardize_videos():
    
