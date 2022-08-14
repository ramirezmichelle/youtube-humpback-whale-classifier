import numpy as np
from sklearn import metrics
from tabulate import tabulate
import wandb


def get_predicted_labels(dataset, model):
    "Generate predicted probabilites and get predicted class (0 or 1). \
    Relevant class is denoted by label = 1, and Irrelevant class by label 0."
    
    probabilities = model.predict(dataset)
    y_pred = np.array([np.argmax(p) for p in probabilities])
    y_true = np.concatenate([label for _, label in dataset], axis=0)
    
    return y_pred, y_true


def get_F1_score(test_dataset, model):
    """ Get F1 Score for trained model """
    
    y_pred, y_true = get_predicted_labels(test_dataset, model)
    return metrics.f1_score(y_true, y_pred)


def get_test_results(test_dataset, test_video_names):
    """ Create a dataframe of all test videos and their true and predicted labels"""

    y_pred, y_true = get_predicted_labels(test_dataset)
    test_results = pd.DataFrame({'File': test_video_names, 
                                 'True Relevant Label': list(map(bool, y_true)),
                                 'Pred Relevant Label': list(map(bool, y_pred))})
    
    return test_results


# def get_false_positives(test_results):
#     # False Positives (True Label = 0 ; Pred Label = 1)
#     false_positives = test_results[(test_results["True Relevant Label"] == False) & (test_results["Pred Relevant Label"] == True)].copy(deep=True)

#     clip_folder = workspace_path + "/video_clips/"

#     #get wandb video objects
#     for i, row in false_positives.iterrows():
#         video_clip = clip_folder + row['File'].replace('_', '_clip_')
#         false_positives.at[i, ('video_clip')] = wandb.Video(video_clip, fps=4, format="gif")

#     #create wandb table with all information
#     table = wandb.Table(dataframe=false_positives)
#     wandb.log({"Videos Mistaken as Humpback Whale Encounters (False Positives)": table})

# def get_false_negatives():
#     # False Negatives (True Label = 1 ; Pred Label = 0)
#     false_negatives = test_results[(test_results["True Relevant Label"] == True) & (test_results["Pred Relevant Label"] == False)].copy(deep = True)

#     clip_folder = workspace_path + "/video_clips/"

#     #get wandb video objects
#     for i, row in false_negatives.iterrows():
#         video_clip = clip_folder + row['File'].replace('_', '_clip_')
#         false_negatives.at[i, ('video_clip')] = wandb.Video(video_clip, fps=4, format="gif")

#     #create wandb table with all information
#     table = wandb.Table(dataframe=false_negatives)
#     wandb.log({"Videos Mistaken as NOT Having Humpback Whales (False Negatives)": table})

# def display_model_train_wandb():
#     #log training and validation metrics on wandb
#     for epoch, train_loss in enumerate(history.history['loss']):
#         wandb.log({'training_loss': train_loss, "epoch": epoch})

#     for epoch, train_acc in enumerate(history.history['accuracy']):
#         wandb.log({'training_accuracy': train_acc, "epoch": epoch})

#     for epoch, val_loss in enumerate(history.history['val_loss']):
#         wandb.log({'val_loss': val_loss, "epoch": epoch})

#     for epoch, val_acc in enumerate(history.history['val_accuracy']):
#         wandb.log({'val_accuracy': val_acc, "epoch": epoch})

#     print('Done Logging WandB metrics.')


def display_results(cnn, loss, accuracy, f1, train_duration, val_duration):
    """ Print presentation info at process completion. """
    
    total_videos = 232 + 59
    total_frames = total_videos * 461
    duration = train_duration + val_duration
    
    videos_per_sec = total_videos/duration
    frames_per_sec = total_frames/duration
    row_data = [[cnn, accuracy, loss, f1, duration, videos_per_sec, frames_per_sec]]
    
    print(tabulate(row_data, headers = ["CNN","Accuracy (Test)", "Loss (Test)", "F1 Score", \
                                        "Time to Extract Features (sec)", "Videos/Second (Feat. Ext.)", \
                                        "Frames/Second (Feat. Ext.)"]))
    return