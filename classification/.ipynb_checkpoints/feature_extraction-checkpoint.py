import cv2
import numpy as np


workspace_path = '/mount/data'

class Extraction:
    def __init__(self, cnn):
        self.feature_extractor = cnn
        
    def load_frames(self, video_title, max_frames):
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
    
    def prepare_all_videos(self, tf_dataset, max_frames, num_features):
            
#         videos = [video_name.numpy().decode("utf-8") for video_name, label in tf_dataset]
#         i=0
#         for features_tensor, target_tensor in tf_dataset:
#             print(i)
#             print(features_tensor.shape)
#             print(features_tensor)
#             print('\n')
#             i+=1

        videos = []
        for features_tensor, target_tensor in tf_dataset:

            videos += [video_name.decode("utf-8") for video_name in features_tensor.numpy()]
            print('\n\n')
                        
        print(videos)
        
        return
        
#         num_samples = len(videos)
        
#         # `frame masks` and `frame_features are what we will feed to our sequence model
#         frame_masks = np.zeros(shape=(num_samples, max_frames), dtype="bool")
#         frame_features = np.zeros(shape=(num_samples, max_frames, num_features) , dtype="float32")

#         for index, video_title in enumerate(videos):

#             if index % 50 == 0:
#                 print(video_title)

#             #Gather all the video's frames and add a batch dimension (frames has shape frames[None, ...])
#             frames = self.load_frames(video_title, max_frames)

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

#         labels = [label.numpy() for video_name, label in tf_dataset]
#         labels = y.astype(int).tolist()
#         return (frame_features, frame_masks), labels

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