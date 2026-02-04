import gym
env = gym.make("MontezumaRevengeNoFrameskip-v4")
print(env.observation_space)
env.reset()
