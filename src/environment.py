from environment import MicrogridEnv

env = MicrogridEnv()
obs, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    env.render()