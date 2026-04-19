from src.environment import MicrogridEnv

env = MicrogridEnv()
obs, _ = env.reset()
done = False

total_reward = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    env.render()
    total_reward += reward

print(f"\nTotal reward (random agent): {total_reward:.3f}")