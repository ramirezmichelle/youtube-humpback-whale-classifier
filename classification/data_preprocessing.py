import cv2
import numpy as np
from decord import VideoReader
from decord import cpu, gpu
import matplotlib.pyplot as plt

workspace_path = '/mount/data'

def get_video_frames_opencv(video, resize=(224, 224)):
    '''Function to get frames from .mp4 video files'''
        
    clip_number = video.split('_')[2].split('.')[0]
    cap = cv2.VideoCapture(workspace_path + "/video_clips/" + video)
    
    #store how many frames are reported to exist in video's metadata
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
    #keep track of frame count
    count = 0
    
    #check whether frames were extracted
    success = True
    
    #collect every frame available
    while success:
        
        success, frame = cap.read()
        
        if not success:
            break

        #resize frame to 224 x 224 and reorder color channels
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]  

        #save frame as jpeg file
        cv2.imwrite(workspace_path + "/frames/" + "/clip_%s_frame_%d.jpg" % (clip_number, count), frame)             
        count += 1

    cap.release()

    return count, total_frames


def get_video_frames(video_title, max_frames, context=cpu(0), resize=(224,224)):
    ''' Get individual image frames from video '''
    
    #get clip number for frame naming
    clip_number = video_title.split('_')[2].split('.')[0]
    
    #read video
    vr = VideoReader(workspace_path + '/video_clips/' + video_title, ctx=context)
    
    #get batch of frames that matches amount needed
    frames = get_n_frames(vr, len(vr))

#     #save frames as jpg images 
#     for i in range(len(frames)):
#         frame = frames[i].asnumpy()
#         frame = cv2.resize(frame, resize)
#         plt.imsave(workspace_path + "/frames/" + "/clip_%s_frame_%d.jpg" % (clip_number, i), frame)
    
    
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
    
