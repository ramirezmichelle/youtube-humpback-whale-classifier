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
    all_video_frames = VideoReader(workspace_path + '/video_clips/' + video_title, ctx=context)
    
    #get batch of frames that matches amount needed
    frames = get_n_frames(all_video_frames, max_frames)

    #save frames as jpg images 
    for i in range(frames.shape[0]):
        frame = frames[i].asnumpy()
        frame = cv2.resize(frame, resize)
        
        #reorder color channels (will leave out for now)
        #frame = frame[:, :, [2, 1, 0]] 
        
        #save frame image in directory
        plt.imsave(workspace_path + "/frames/" + "/clip_%s_frame_%d.jpg" % (clip_number, i), frame)
    
    
    #return frame numbers to double check functionality
    num_frames_collected = frames.shape[0]
    num_total_frames = len(all_video_frames)
    
    return num_frames_collected, num_total_frames
    

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
        frames = all_frames
        
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

# def standardize_videos():
    
