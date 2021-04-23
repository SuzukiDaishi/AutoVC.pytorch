FROM pytorch/pytorch:latest
USER root

RUN apt-get update && \
    apt-get update && apt-get upgrade -y && apt-get install -y

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools

RUN apt install -y llvm-8 && \
    export LLVM_CONFIG=/usr/bin/llvm-config-8 && \
    pip install llvmlite

RUN apt-get install -y ffmpeg &&\
    apt-get install -y libsndfile1-dev

RUN pip3 install librosa==0.7.2 \
                 numba==0.48.0 \
                 pyworld \
                 matplotlib \
                 tqdm \
                 resemblyzer \
                 wavenet_vocoder


# COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt