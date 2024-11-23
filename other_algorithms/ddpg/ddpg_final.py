import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


random.seed(20)
np.random.seed(20)
torch.manual_seed(20)

# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # action = torch.tanh(self.fc3(x))  # Output actions in range [-1, 1]
        action = torch.sigmoid(self.fc3(x))
        return action

# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    # def forward(self, state, action):
    #     x = torch.cat([state, action], dim=1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     q_value = self.fc3(x)
    #     return q_value

    def forward(self, state, action):
        # Ensure action has the shape (batch_size, action_dim)
        if action.dim() == 1:
            action = action.unsqueeze(1)  # Reshape from (batch_size,) to (batch_size, 1)
        
        # Concatenate state and action along dim=1
        x = torch.cat([state, action], dim=1)  # Now both tensors are 2D
        
        # Forward pass through the critic network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)  # Final Q-value output

        return q_value
    
# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def size(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, buffer_size=100000, batch_size=64, gamma=0.99, tau=0.001, lr_actor=1e-4, lr_critic=1e-3):
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor-Critic Networks
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, batch_size)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action = np.clip(action + noise * np.random.randn(*action.shape), -1, 1)
        # return action
        #discrete_action = 0 if action < 0 else 1  # action < 0 -> move left, action >= 0 -> move right
        discrete_action = 0 if action < 0.5 else 1  # action < 0 -> move left, action >= 0 -> move right
        return discrete_action

#     def update(self):
#         # Sample a batch of experiences from the buffer
#         if self.buffer.size() < self.batch_size:
#             return
#         batch = self.buffer.sample()

#         states, actions, rewards, next_states, dones = zip(*batch)
#         states = torch.FloatTensor(states)
#         actions = torch.FloatTensor(actions)
#         rewards = torch.FloatTensor(rewards).view(-1, 1)
#         next_states = torch.FloatTensor(next_states)
#         dones = torch.FloatTensor(dones).view(-1, 1)

#         # Update Critic
#         next_actions = self.actor_target(next_states)
#         next_q_values = self.critic_target(next_states, next_actions)
#         target_q_values = rewards + self.gamma * (1 - dones) * next_q_values

#         current_q_values = self.critic(states, actions)
#         critic_loss = F.mse_loss(current_q_values, target_q_values)

#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         # Update Actor
#         actor_loss = -self.critic(states, self.actor(states)).mean()

#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # Soft update the target networks
#         self.soft_update(self.actor, self.actor_target)
#         self.soft_update(self.critic, self.critic_target)

#     def update(self):
#         # Sample a batch of experiences from the buffer
#         if self.buffer.size() < self.batch_size:
#             return
#         batch = self.buffer.sample()

#         states, actions, rewards, next_states, dones = zip(*batch)
#         states = torch.FloatTensor(states)
#         actions = torch.FloatTensor(actions)
#         rewards = torch.FloatTensor(rewards).view(-1, 1)  # Ensure rewards are shaped (batch_size, 1)
#         next_states = torch.FloatTensor(next_states)
#         dones = torch.FloatTensor(dones).view(-1, 1)  # Ensure dones are shaped (batch_size, 1)

#         # Update Critic
#         next_actions = self.actor_target(next_states)
#         next_q_values = self.critic_target(next_states, next_actions)
#         target_q_values = rewards + self.gamma * (1 - dones) * next_q_values

#         current_q_values = self.critic(states, actions)

#         # Ensure both target and current Q-values are shaped (batch_size, 1)
#         target_q_values = target_q_values.view(-1, 1)  # Ensure target Q-values are (batch_size, 1)

#         critic_loss = F.mse_loss(current_q_values, target_q_values)

#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         # Update Actor
#         actor_loss = -self.critic(states, self.actor(states)).mean()

#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # Soft update the target networks
#         self.soft_update(self.actor, self.actor_target)
#         self.soft_update(self.critic, self.critic_target)

    def update(self):
        # Sample a batch of experiences from the buffer
        if self.buffer.size() < self.batch_size:
            return
        batch = self.buffer.sample()

        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert them to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).view(-1, 1)  # Ensure rewards are (batch_size, 1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).view(-1, 1)  # Ensure dones are (batch_size, 1)

        # Update Critic Network
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)  # (batch_size, 1)

        # Target Q-values computation: reward + gamma * (1 - done) * next_q_values
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_values

        # Critic loss computation (MSE between current Q-values and target Q-values)
        current_q_values = self.critic(states, actions)  # (batch_size, 1)

        # Ensure target_q_values has the shape (batch_size, 1) before computing loss
        target_q_values = target_q_values.view(-1, 1)  # (batch_size, 1) for loss comparison

        # Compute MSE loss between current Q-values and target Q-values
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        # Backpropagate and update the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor Network
        actor_loss = -self.critic(states, self.actor(states)).mean()  # Maximize Q-value for chosen actions

        # Backpropagate and update the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks (moving averages)
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)


    def soft_update(self, model, target_model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# Main function to train the agent
def train_ddpg(env, agent, episodes=1000, max_timesteps=200):
    reward_records = []
    for episode in range(episodes):
        env.seed(episode)
        state = env.reset()
        env.seed(episode)
        
        episode_reward = 0
        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.add((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            agent.update()

            if done:
                reward_records.append(episode_reward)
                break

        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")

    # Generate recent 150 interval average
    average_reward = []
    for idx in range(len(reward_records)):
        avg_list = np.empty(shape=(1,), dtype=int)
        if idx < 150:
            avg_list = reward_records[:idx+1]
        else:
            avg_list = reward_records[idx-149:idx+1]
        average_reward.append(np.average(avg_list))
    plt.plot(reward_records)
    plt.plot(average_reward)
    plt.axhline(y=195, color='r', linestyle='-')
    plt.show()

    print('Train summary: ')
    print(pd.Series(reward_records).describe())
        



def test_agent(i, print_reward=False):
    env = gym.make('CartPole-v0')
    env.seed(i)
    s = env.reset()
    env.seed(i)
    rewards = []
    obs=[]
    for t in range(200):
        a = agent.select_action(s)
        s, r, term, trunc = env.step(a)
        done = term or trunc
        rewards.append(r)
        #print(sum(rewards))  
        if done:
            rew = sum(rewards)
            if print_reward:
                print("Reward:", rew)
            return rew
    env.close()


def see_test_results(total=100):
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    test_scores = []
    for i in range(total):
        test_scores.append(test_agent(i, print_reward=False))
        
    print(test_scores)

    print('test_score:', np.mean(test_scores))
    print('test_success_percentage:', len([i for i in test_scores if i>=195]))
    
        
# Initialize the environment and agent
env = gym.make('CartPole-v0')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
state_dim = 4
action_dim = 1

agent = DDPGAgent(state_dim, action_dim)

# Train the agent
train_ddpg(env, agent)
see_test_results(100)