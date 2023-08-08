
From nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update

RUN \
    apt-get install -y git && \
    apt-get install -y python3-pip && \
    apt-get install python3-dev

RUN apt-get install -y libosmesa6-dev
RUN apt-get install -y xvfb

RUN \
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install torchrl && \
    pip install transformers && \
    pip install einops && \
    pip install sentencepiece && \
    pip install protobuf==3.20 && \
    pip install accelerate


RUN \
    pip install datasets && \
    pip install bitsandbytes && \
    pip install scipy


RUN pip install matplotlib



# Mujoco
RUN apt install libosmesa6-dev
RUN apt install libgl1-mesa-glx
RUN apt install libglfw3
RUN apt install patchelf

RUN pip install mujoco
RUN pip install pyvirtualdisplay

RUN \
    pip install -U 'mujoco-py<2.2,>=2.1' && \
    pip install "cython<3" && \
    pip install gymnasium==0.29.0
  
## add path for MUJOCO
# RUN export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210
# RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin

ENV MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin