import tensorflow as tf
import numpy as np
import argparse
import time
import os

from load_data import *
from get_features import feature_extraction_cpu, feature_extraction_gpu
from rnn import train_rnn
from analysis import *
import wandb


def limit_gpu_memory_growth():
    """Function to limit gpu memory growth. Prevents TensorFlow 
    from taking up all GPU memory available.
    """

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth is now the same across all {len(gpus)} available GPUs.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

            
def main():
    #parse args here
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", default=0, type=int)
    parser.add_argument("--cnn_model", default="resnet101", type=str)
    parser.add_argument("--wandb_run", default="just another run", type=str)
    parser.add_argument("--wandb_api_key", default=None, type=str)
    
    args = parser.parse_args()
    
    print(f"NUM GPUS: {args.num_gpus}")
    print(f"CNN: {args.cnn_model}")   
    print(f"wandb run name: {args.wandb_run}")   
    
    #start logging info
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.login()
        wandb.init(project="whale-classification-ngc")
        wandb.run.name = args.wandb_run
    
    # don't let TF take up all the gpu memory
    limit_gpu_memory_growth()
    
    #read in dataframes
    X, y = load_dataframes()

    # get our data ready
    print('Loading data...')
    start = time.time()
    video_names = list(X.renamed_title)
    videos, video_labels = load_frames_and_labels(video_names) 
    stop = time.time()
    print(f"Done loading videos in {stop-start} seconds.")

    #split videos into train, val, test datasets
    train_dataset, val_dataset, test_dataset = split_video_dataset(X, y, videos, video_labels)
    
    # get video frame feature representations with CNN for each dataset split
    if args.num_gpus >= 1:
        train_features, train_labels, train_duration_cnn = feature_extraction_gpu(args.num_gpus, train_dataset, args.cnn_model, augment_data=True)
        val_features, val_labels, val_duration_cnn = feature_extraction_gpu(args.num_gpus, val_dataset, args.cnn_model, augment_data=True)
        test_features, test_labels, _ = feature_extraction_gpu(args.num_gpus, test_dataset, args.cnn_model)

    else:
        frames_per_video = videos.shape[1]
        train_features, train_labels, train_duration_cnn = feature_extraction_cpu(train_dataset, frames_per_video, args.cnn_model, augment_data=True)
        val_features, val_labels, val_duration_cnn = feature_extraction_cpu(val_dataset, frames_per_video, args.cnn_model, augment_data=True)
        test_features, test_labels, _ = feature_extraction_cpu(test_dataset, frames_per_video, args.cnn_model)
        
    print("Back from feature Extraction.")
    print(f"Train Features: {train_features.shape} || Train Labels: {train_labels.shape}")
    print(f"Val Features: {val_features.shape} || Val Labels: {val_labels.shape}")
    print(f"Test Features: {test_features.shape} || Test Labels: {test_labels.shape}")

    # split and batch data for RNN 
    train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
    val_labels = np.reshape(val_labels, (val_labels.shape[0], 1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))
    
    with tf.device("/device:CPU:0"):
        BUFFER_SIZE_TRAIN = train_features.shape[0]
        BUFFER_SIZE_VAL = val_features.shape[0]
        BUFFER_SIZE_TEST = test_features.shape[0]
        BATCH_SIZE = 32
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).shuffle(BUFFER_SIZE_TRAIN).batch(BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).batch(BATCH_SIZE)    
        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(BATCH_SIZE)

    # train RNN
    model = train_rnn(train_dataset, val_dataset, train_features.shape[2])
    
    # evaluate trained model on test data
    loss, accuracy = model.evaluate(test_dataset)    
    f1 = get_F1_score(test_dataset, model)
    
#     # visualize misclassifications on wandb
#     test_video_names = get_test_video_names(X)
#     test_classifications = get_test_results(test_dataset, test_video_names)
    
    # print metrics for model run on test data
    display_results(args.cnn_model, loss, accuracy, f1, train_duration_cnn, val_duration_cnn)
    
    if args.wandb_api_key:
        wandb.finish()
    
    return

if __name__ == "__main__":
    main()
