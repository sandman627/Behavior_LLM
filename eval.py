import gymnasium
from gymnasium.wrappers import TimeLimit

import metaworld

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy



## https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#id3



if __name__=="__main__":
    print("Evaluating Behavior Framework!!")
    
    ## Environment declare and setting
    print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
    ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks
    env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
    env = TimeLimit(env, max_episode_steps=1000)
    print("Task List : ", ml1.train_tasks[0])
    
    
    
    # task = random.choice(ml1.train_tasks)
    task = ml1.train_tasks[0]
    env.set_task(task)  # Set task
    obs, info = env.reset()
    print("Task : ", task)


    # Create the model and the training environment
    model = PPO(policy="MlpPolicy", env=env)

    # train the model
    model.learn(total_timesteps=100)

    # save the model
    model.save("ppo_pendulum")

    # the saved model does not contain the replay buffer
    loaded_model = PPO.load("ppo_pendulum")
    # print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

    # now save the replay buffer too
    # model.save_replay_buffer("ppo_replay_buffer")

    # load it into the loaded_model
    # loaded_model.load_replay_buffer("ppo_replay_buffer")

    # now the loaded replay is not empty anymore
    # print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

    # Save the policy independently from the model
    # Note: if you don't save the complete model with `model.save()`
    # you cannot continue training afterward
    policy = model.policy
    policy.save("sac_policy_pendulum")

    # Retrieve the environment
    env = model.get_env()

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=1, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    exit()
    # Load the policy independently from the model
    saved_policy = MlpPolicy.load("sac_policy_pendulum")

    # Evaluate the loaded policy
    mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=10, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")







    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(
        eval_env=env, 
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=500
    )

    model = PPO()
    model.learn(5000, callback=eval_callback)