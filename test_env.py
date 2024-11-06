import numpy as np
from make_env import make_env
import time

env = make_env('simple_adversary')
print(f"obs space {env.observation_space}")
print(f"action space {env.action_space}")
print(f"n_agents {env.n}")

obs = env.reset()
print(obs)

# Action Space is discrete
no_op = np.array([1, 0, 0, 0, 0])
action = [no_op, no_op, no_op]

obs, reward, done, info = env.step(action)
print(obs)
print(reward)
print(done)
print(info)

for _ in range(1000):
    env.render()
    action = [np.eye(5)[np.random.randint(5)] for i in range(env.n)]
    obs, reward, done, info = env.step(action)
    print(obs)
    print(reward)
    print(done)
    print(info)
    time.sleep(0.01)