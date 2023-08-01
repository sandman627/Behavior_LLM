



# 환경에 대한 설명

Transformer 최신버전을 사용하기 위해서는 pytorch 2.0.0 이상을 사용해야는데, 이 버전은 CUDA 11.7을 요구한다.
이를 맞추기 위해 Ubuntu 22.04 버전을 사용하며, python3.10 을 사용한다.




## MUJOCO 환경변수 설정
이상하게도 Terminal을 한번 키고 닫을때마다 이렇게 export 해줘야함.


'''
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
'''