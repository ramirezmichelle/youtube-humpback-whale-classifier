FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3

RUN git clone --branch nvidia-intern-project --single-branch https://github.com/ramirezmichelle/youtube-humpback-whale-classifier 

RUN apt update && \
    apt install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        build-essential \
        python3-dev \
        python3-setuptools \
        make \ 
        cmake \
        libavcodec-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev

RUN git clone --recursive https://github.com/dmlc/decord

RUN cd decord && mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && make -j2 && cd ../python && python3 setup.py install

RUN pip install imageio \
    imutils \
    opencv-python \
    seaborn \
    git+https://github.com/tensorflow/docs \
    wandb \ 
    decord \ 
    moviepy


