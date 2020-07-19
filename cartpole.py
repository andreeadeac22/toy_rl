import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#env=gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")

class Policy(nn.Module):
  def __init__(self, input, hidden, actions):
    super(Policy, self).__init__()
    self.linear1 = nn.Linear(input, hidden)
    self.relu = nn.ReLU()
    self.actor = nn.Linear(hidden, actions)
    self.critic = nn.Linear(hidden, 1)

  def forward(self, obs):
    obs = self.relu(self.linear1(obs))
    return self.actor(obs), self.critic(obs)

print(env.observation_space.shape[0])
print(env.action_space.n)

model = Policy(input=env.observation_space.shape[0], hidden=32, actions=env.action_space.n)
opt = torch.optim.Adam(list(model.parameters()), lr=1e-2, weight_decay=5e-3)

gamma = 0.99

for _ in range(1000):
  #starting episode
  total_reward = 0
  pow_gamma = 1.
  traj_prob = []
  traj_actions = []
  traj_rewards = []
  traj_values = []

  obs = torch.Tensor(env.reset())
  while True:
    env.render()
    #action = env.action_space.sample()
    output, value = model(obs)
    categ = torch.distributions.categorical.Categorical(F.softmax(output))
    action = categ.sample()

    traj_prob += [output]
    traj_actions += [action]
    traj_values += [value]

    obs, reward, done, info = env.step(action.numpy())
    obs = torch.Tensor(obs)

    traj_rewards += [reward]

    #total_reward += reward * pow_gamma
    #pow_gamma = pow_gamma * gamma

    if done:
      loss = 0
      print(len(traj_actions))
      print("Total reward ", total_reward)
      qs = torch.zeros(len(traj_actions))
      _, qval = model(obs)
      for t in reversed(range(len(traj_rewards))):
        qval = traj_rewards[t] + gamma * qval
        qs[t] = qval

      for i in range(len(traj_actions)):
        loss += - F.log_softmax(traj_prob[i])[traj_actions[i]] * (qs[i] - traj_values[i])
        loss += (qs[i] - traj_values[i])*(qs[i] - traj_values[i])
      opt.zero_grad()
      loss.backward()
      opt.step()
      break

torch.save(model.state_dict(), 'model.pt')



env.close()
