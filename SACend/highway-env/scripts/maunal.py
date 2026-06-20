

import gym
import highway_env

env = gym.make("highway-v0")
env.configure({
    "action": {"type": "ContinuousAction"},
    "manual_control": True
})
env.reset()
done = False
while not done:
    _, _, done, _ = env.step(env.action_space.sample())  # with manual control, these actions are ignored
    env.render()
