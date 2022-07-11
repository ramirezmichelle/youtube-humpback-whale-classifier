import cv2
import numpy as np


workspace_path = '/mount/data'

def load_frames(video_title, max_frames):
    '''read all frames in for 1 video from workspace 'frames' directory'''

    #get number associated with clip to retrieve respective frames
    clip_number = video_title.split('_')[1].split('.')[0]

    #create list to store each frame
    frames = []

    for i in range(max_frames):

        #read in .jpg file as array for video clip 
        img = cv2.imread(workspace_path + f'/frames/clip_{clip_number}_frame_{i}.jpg')
        frames.append(img)

    #put list of frames in numpy format
    frames = np.array(frames)

    #return frames with an extra batch dimension
    return frames[None, ...]

def prepare_all_videos(X, y, max_frames, num_features, feature_extractor):

    num_samples = len(X)
    videos = list(X['renamed_title'].values)

    # `frame masks` and `frame_features are what we will feed to our sequence model
    frame_masks = np.zeros(shape=(num_samples, max_frames), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, max_frames, num_features) , dtype="float32")

    for index, video_title in enumerate(videos):

        if index % 100 == 0:
            print(video_title)
            
        #Gather all the video's frames and add a batch dimension (frames has shape frames[None, ...])
        frames = load_frames(video_title, max_frames)

        #initialize placeholders to store the masks and features of the current video
        temp_frame_mask = np.zeros(shape=(1, max_frames), dtype="bool")  
        temp_frame_features = np.zeros(shape=(1, max_frames, num_features), dtype="float32")

        for i, batch in enumerate(frames):
            #extract features from the frames of the current video
            for j in range(max_frames):
                current_frame = batch[None, j, :]
                temp_frame_features[i, j, :] = feature_extractor.predict(current_frame)

            #create mask for current video 
            #1 = not masked, 0 = masked
            temp_frame_mask[i, :max_frames] = 1 

        frame_features[index, ] = temp_frame_features.squeeze()
        frame_masks[index, ] = temp_frame_mask.squeeze()


    labels = y.astype(int).tolist()
    return (frame_features, frame_masks), labels