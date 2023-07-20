
From nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN \
    apt-get update && \
    apt-get install -y git && \
    apt-get install -y python3-pip





RUN \
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install transformers && \
    pip install einops && \
    pip install sentencepiece && \
    pip install protobuf==3.20 && \
    pip install accelerate


RUN \
    pip install datasets
    pip install bitsandbytes
    pip install scipy


# Mujoco

RUN \
    apt install libosmesa6-dev libgl1-mesa-glx libglfw3
    apt install patchelf

RUN \
    pip install -U 'mujoco-py<2.2,>=2.1'
    pip install "cython<3"
    
# RUN \
#     export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210

