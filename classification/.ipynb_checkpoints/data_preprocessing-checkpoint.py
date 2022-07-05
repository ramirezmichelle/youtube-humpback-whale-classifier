import cv2
import numpy as np

workspace_path = '/mount/data'

def get_video_frames(video, resize=(224, 224)):
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
#         cv2.imwrite(workspace_path + "/frames/" + "/clip_%s_frame_%d.jpg" % (clip_number, count), frame)             
        count += 1

    cap.release()

    return count, total_frames


def frame_capture(path):
    
    vidObj = cv2.VideoCapture(path)
    total_frames = vidObj.get(cv2.CAP_PROP_FRAME_COUNT)
    print(total_frames)
    
    #keep track of frame count
    count = 0
    
    #check whether frames were extracted
    success = 1
    
    while count != total_frames:
        success, image = vidObj.read()
        
        if success:
            cv2.imwrite("frames/frame%d.jpg" % count, image)
            count +=1
        else:
            print('frame not available')
            break
                
    return count
    
# def pad_videos():
    
#     '''Function to pad videos that fall short of standardized frame count by replicating middle frame'''
    
# def standardize_videos():
    
