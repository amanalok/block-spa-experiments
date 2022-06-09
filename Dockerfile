# for list of aws deep learning containers : https://github.com/aws/deep-learning-containers/blob/master/available_images.md

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu115-ubuntu20.04-e3

ENV TZ=Europe/Copenhagen
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

# Project setup

COPY requirements.txt .
COPY mlops/requirements.txt mlops/requirements.txt
COPY mlops/requirements-non-tensorflow.txt mlops/requirements-non-tensorflow.txt

RUN python -m pip install --upgrade pip
RUN pip list
RUN pip install -r requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN pip install sagemaker-training
RUN python -c 'import torch; print(torch.__version__)'
RUN nvcc --version