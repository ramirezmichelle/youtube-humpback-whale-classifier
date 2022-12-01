import cv2
import numpy as np
import tensorflow as tf


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
#     return frames[None, ...]

    return frames
@tf.function
def prepare_all_videos(string_video_names, labels, max_frames, num_features, feature_extractor):       
# def prepare_all_videos(replica_data, replica_id, max_frames, num_features, feature_extractor):       

    #decode byte video names into strings and bool labels into ints 
    #this process merges all batches contained in tf_dataset back into one entity

#     feature_tensor = replica_data[0].values[replica_id]
#     labels_tensor = replica_data[1].values[replica_id]
    
#     print(feature_tensor)
#     print(labels_tensor)
#     return
    
#     string_video_names = [video.decode("utf-8") for video in list(feature_tensor.numpy())]
#     labels = [bool_label.astype(int) for bool_label in list(labels_tensor.numpy())]

    num_samples = len(string_video_names)

    # `frame masks` and `frame_features are what we will feed to our sequence model
    frame_masks = np.zeros(shape=(num_samples, max_frames), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, max_frames, num_features) , dtype="float32")

    res = []
    for index, video_title in enumerate(string_video_names):

        if index % 50 == 0:
            print(video_title)

        #Gather all the video's frames and add a batch dimension (frames has shape frames[None, ...])
        frames = load_frames(video_title, max_frames)

        #initialize placeholders to store the masks and features of the current video
        temp_frame_mask = np.zeros(shape=(1, max_frames), dtype="bool")  
        temp_frame_features = np.zeros(shape=(1, max_frames, num_features), dtype="float32")

        video_batch_features = []
        for i, batch in enumerate(frames):

            #extract features from all (461) frames in batch at once
#             batch_features = feature_extractor.predict_on_batch(batch)
            batch_features = feature_extractor(batch, training=False)
    
#             print(type(batch_features))
            
            video_batch_features.append(batch_features)
        
        res.append(video_batch_features)
        
        return res

#             temp_frame_features[i, :, :] = batch_features

#             #create mask for current video: 1 = not masked, 0 = masked
#             temp_frame_mask[i, :max_frames] = 1 

#         frame_features[index, ] = temp_frame_features.squeeze()
#         frame_masks[index, ] = temp_frame_mask.squeeze()


#     return (frame_features, frame_masks), labels

#     def prepare_all_videos(X, y, max_frames, num_features):

#         num_samples = len(X)
#         videos = list(X['renamed_title'].values)

#         # `frame masks` and `frame_features are what we will feed to our sequence model
#         frame_masks = np.zeros(shape=(num_samples, max_frames), dtype="bool")
#         frame_features = np.zeros(shape=(num_samples, max_frames, num_features) , dtype="float32")

#         for index, video_title in enumerate(videos):

#             if index % 100 == 0:
#                 print(video_title)

#             #Gather all the video's frames and add a batch dimension (frames has shape frames[None, ...])
#             frames = load_frames(video_title, max_frames)

#             #initialize placeholders to store the masks and features of the current video
#             temp_frame_mask = np.zeros(shape=(1, max_frames), dtype="bool")  
#             temp_frame_features = np.zeros(shape=(1, max_frames, num_features), dtype="float32")

#             for i, batch in enumerate(frames):

#                 #extract features from all (461) frames in batch at once
#                 batch_features = self.feature_extractor.predict_on_batch(batch)

#                 temp_frame_features[i, :, :] = batch_features

#                 #create mask for current video: 1 = not masked, 0 = masked
#                 temp_frame_mask[i, :max_frames] = 1 

#             frame_features[index, ] = temp_frame_features.squeeze()
#             frame_masks[index, ] = temp_frame_mask.squeeze()


#         labels = y.astype(int).tolist()
#         return (frame_features, frame_masks), labels