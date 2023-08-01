# Behavior_LLM














## Mujoco 설정
Mujoco210을 설치한 path를 추가해줘야한다. 원래 default 위치는 root/.mujoco/mujoco210 이다.
이 곳 외에 다른 곳에 mujoco210을 저장해서 사용하고 싶다면 아래 환경변수 추가를 해줘야 한다.

'''
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210
'''