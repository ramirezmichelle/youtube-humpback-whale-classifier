import cv2
import numpy as np

workspace_path = '/mount/data'

def get_video_frames(video_title, max_frames=500, resize=(224, 224)):
    '''Function to get frames from .mp4 video files'''
        
    num_frames_extracted = 0
    frames = []

    cap = cv2.VideoCapture(workspace_path + '/video_clips/' + video_title)
    num_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    success = True
    next_frame_index = 0
    if num_total_frames <= max_frames:
        #collect every frame available
        while success:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  #reorders color channels from BGR [0,1,2] to RGB order [2,1,0]
            frames.append(frame)
            num_frames_extracted += 1 

            next_frame_index += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_index)

        cap.release()

    print(f'num frames extracted: {num_frames_extracted}')
    return np.array(frames)

# def pad_videos():
    
#     '''Function to pad videos that fall short of standardized frame count by replicating middle frame'''
    
# def standardize_videos():
    
