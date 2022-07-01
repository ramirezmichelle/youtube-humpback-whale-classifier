import cv2
import numpy as np

workspace_path = '/mount/data'

def get_video_frames(video_title, max_frames=500, resize=(224, 224)):
    '''Function to get frames from .mp4 video files'''
    
    print(max_frames) 
    
    count = 0
    num_frames = 0

    cap = cv2.VideoCapture(workspace_path + '/video_clips/' + video_title)
    frames = []
    
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)

    try:
        while True and num_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  #reorders color channels from BGR [0,1,2] to RGB order [2,1,0]
            frames.append(frame)

            #to skip over some redundant frames and get more diverse set of frames over span of our video
            #while still getting a fixed, uniform number of max_frames for each video
            count += 0.5 * fps
            cap.set(1, count)

            num_frames += 1

    finally:
        cap.release()

    print(f'num frames {num_frames}')
    return np.array(frames)

# def pad_videos():
    
#     '''Function to pad videos that fall short of standardized frame count by replicating middle frame'''
    
# def standardize_videos():
    
