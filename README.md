# Behavior_LLM







# Installation and settings

```
pip install -r requirements_gym.txt
```

## Mujoco 설정
Mujoco210을 설치한 path를 추가해줘야한다. 원래 default 위치는 root/.mujoco/mujoco210 이다.
이 곳 외에 다른 곳에 mujoco210을 저장해서 사용하고 싶다면 아래 환경변수 추가를 해줘야 한다.


```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210
```


## Metaworld와 Stable-baselines3 설치
2023년 8월 1일 현재 위 두  package 모두 gymnasium 0.29.0 와 호환성 패치가 진행된지 2주 혹은 며칠 밖에 되지 않았다.
그래서 github에서 직접 가장 최신 버전을 설치했다.


https://github.com/Farama-Foundation/Metaworld


https://github.com/DLR-RM/stable-baselines3

pip install git+{git repo 주소}로 설치하던가 아니면 git clone을 이용해 직접 설치하자