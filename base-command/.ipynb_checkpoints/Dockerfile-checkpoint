FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3

RUN git clone --branch nvidia-intern-project --single-branch https://github.com/ramirezmichelle/youtube-humpback-whale-classifier 

RUN apt update && \
    apt install -y \
        ffmpeg \
        libsm6 \
        libxext6

RUN pip install imageio \
    imutils \
    opencv-python \
    seaborn \
    git+https://github.com/tensorflow/docs \
    wandb \ 
    decord \ 
    moviepy \
    tabulate


