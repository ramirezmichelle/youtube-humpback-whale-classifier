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
    return
    
    
def pad_frames(num_available, n):
    
    '''Function to pad videos that fall short of standardized frame count by replicating middle frame.
    Inputs
        num_available (int): number of frames available
        n (int): number of frames we need
    
    Outputs
        frame_indices (int): indices of frames, including padded middle frame 
    '''
    return
    
    
# def standardize_videos():
    
