import gym
import numpy as np

env = gym.make('FrozenLake-v0')

# matriz com estados e acoes que podemos tomar.
Q = np.zeros([env.observation_space.n, env.action_space.n])

# hyperparametros
learning_rate = 0.8
y = 0.95
num_episodes = 2000

# lista de rewards
rewards = []

for i in range(num_episodes):
    s = env.reset()
    rewards_episode = 0
    d = False
    j = 0
    while j < 99:
        j += 1
        # escolher uma acao
        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1.0/(i+1)))
        s1, r, d, _ = env.step(a)
        Q[s,a] = Q[s,a] + learning_rate*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rewards_episode += r
        s = s1
        if d == True:
            break
            rewards.append(rewards_episode)

print("Score over time: " +  str(sum(rewards)/num_episodes))
print("Final Q-Table Values")
print(Q)