## Training ML Models to Classify Humpback Whales in YouTube Videos

This repository contains code to train and compare models following a CNN-RNN architecture (CNN is pretrained on imagenet weights) for classication of YouTube videos in search of videos that contain true humpback whale encounters.

To run classification, you will need to run the **main.py** script inside of the **classification/scripts** folder. 

    python main.py --num_gpus=<NUM GPUS> --cnn_model=<SPECIFY ONE OF 'resnet101', 'resnet50', 'vgg16', 'vgg19', or 'inception'>

To run the command and visualize outputs in Weights & Biases (wandb), add the following arguments as such:

    python main.py --num_gpus=<NUM GPUS> --cnn_model=<SPECIFY ONE OF 'resnet101', 'resnet50', 'vgg16', 'vgg19', or 'inception'> --wandb_run=<RUN NAME> --wandb_api_key=<API KEY TO CONNECT TO WANDB>

Dataset is currently not available remotely and publicly. Readme will be update accordinly once dataset is available. 