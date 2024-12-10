#FROM pytorch/pytorch:latest
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y upgrade 
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update

RUN apt-get install -y python3.10 python3.10-distutils python3.10-venv
RUN python3.10 -m ensurepip
RUN python3.10 -m pip install --upgrade pip

RUN apt-get update && apt-get install -y git python3.10-tk
RUN apt-get -y install libgl1-mesa-glx libglib2.0-0 libgomp1

# 必要なPythonパッケージのインストール
RUN pip install matplotlib scipy scikit-image scikit-video h5py opencv-python cmapy tensorboard configargparse moviepy plyfile nd2reader pandas open3d PyOpenGL numpy tensorrt

# 特定バージョンのTensorFlowおよびtensorflow-io-gcs-filesystemをインストール
RUN pip install tensorflow==2.14.1 tensorflow-io-gcs-filesystem==0.34.0

# PyTorch関連パッケージのインストール
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
